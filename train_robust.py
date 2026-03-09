"""
Robust Training script for Loss Estimation Model

On-the-fly adversarial training: for a fraction of batches, the predicted
image is replaced by a pixel-optimized adversarial version.  The model then
learns to predict the *real* metrics of that adversarial image, forcing it
to become robust to pixel-level attacks.

Key difference from train.py:
  - Adversarial samples are generated **per-batch** (on-the-fly), not as a
    separate epoch-level step.
  - The ratio of adversarial-to-normal batches is controlled by --adv_ratio.
  - Uses RobustDataset that also returns GT for real metric computation.
"""
import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn
import imageio
import lpips

from model import LossEstimationNet
from dataset import LossEstimationDataset


# ============================================================
# Dataset that also returns GT focal plane
# ============================================================

class RobustDataset(LossEstimationDataset):
    """
    Extends LossEstimationDataset to also return the GT focal plane image,
    needed for computing real metrics during adversarial training.

    Returns: (x, diopter, targets, gt)
      - gt: (3, H, W) normalized GT focal plane tensor
    """

    def __getitem__(self, idx):
        scene_idx, plane_idx, sample_type = self.samples[idx]
        scene_name = f'seed{scene_idx:04d}'
        scene_dir = os.path.join(self.data_dir, scene_name, '512')

        # Get the standard outputs from parent
        x, diopter, targets = super().__getitem__(idx)

        # Load GT focal plane
        gt_path = os.path.join(scene_dir, f'frame{plane_idx:04d}.exr')
        gt = self._load_exr(gt_path)

        # Same normalization as RGB in parent class
        rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
        rgb_raw = self._load_exr(rgb_path)
        rgb_gamma = np.power(np.abs(rgb_raw), 1.0 / 2.2)
        im_max = np.max(rgb_gamma)

        gt = np.power(np.abs(gt), 1.0 / 2.2)
        gt = gt / (im_max + 1e-8)
        gt = np.clip(gt, 0, 1).astype(np.float32)

        gt_t = torch.from_numpy(gt).permute(2, 0, 1).float()  # (3, H, W)

        return x, diopter, targets, gt_t


# ============================================================
# Per-batch adversarial generation
# ============================================================

def generate_adversarial_batch(model, x, diopter, device,
                               adv_steps=15, adv_lr=0.01):
    """
    Pixel-optimize the predicted image (channels 4:7 of x) to fool the model.

    x layout: [rgb(3), depth(1), pred(3)]  — channels 0-6
    We only perturb pred (channels 4:7).

    Returns:
        x_adv: adversarial input tensor (same shape as x)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rgbd = x[:, :4].detach()           # (B, 4, H, W) — frozen
    pred_init = x[:, 4:].detach()      # (B, 3, H, W) — will be optimized

    pred_opt = pred_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([pred_opt], lr=adv_lr)

    for _ in range(adv_steps):
        opt.zero_grad()
        pred_opt.data.clamp_(0, 1)

        x_cat = torch.cat([rgbd, pred_opt], dim=1)
        out = model(x_cat, diopter)

        # Fool the model: maximize predicted quality
        loss = -out['psnr'].mean() - out['ssim'].mean() + out['lpips'].mean()
        loss.backward()
        opt.step()

    pred_opt.data.clamp_(0, 1)

    x_adv = torch.cat([rgbd, pred_opt.detach()], dim=1)

    for p in model.parameters():
        p.requires_grad = True
    model.train()

    return x_adv


def compute_real_metrics_batch(x_adv, gt_batch, lpips_fn, device):
    """
    Compute real PSNR, SSIM, LPIPS between adversarial pred and actual GT.

    x_adv layout: [rgb(3), depth(1), pred_adv(3)]
    gt_batch: (B, 3, H, W)  — actual GT focal plane

    Returns:
        dict with 'psnr', 'ssim', 'lpips' tensors of shape (B,)
    """
    pred_adv = x_adv[:, 4:].detach()  # (B, 3, H, W)
    B = pred_adv.size(0)

    psnrs, ssims = [], []

    for i in range(B):
        p = pred_adv[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        g = gt_batch[i].cpu().numpy().transpose(1, 2, 0)

        # PSNR
        mse = np.mean((p - g) ** 2)
        psnr = -10 * np.log10(mse + 1e-10) if mse > 0 else 100.0
        psnrs.append(psnr)

        # SSIM
        s = ssim_fn(p, g, data_range=1.0, channel_axis=2)
        ssims.append(s)

    # LPIPS (batched, expects [-1, 1] range)
    with torch.no_grad():
        pred_lpips = pred_adv.to(device) * 2 - 1
        gt_lpips = gt_batch.to(device) * 2 - 1
        lp = lpips_fn(pred_lpips, gt_lpips).view(B)
    lpipss = lp.cpu().tolist()

    return {
        'psnr': torch.tensor(psnrs, dtype=torch.float32),
        'ssim': torch.tensor(ssims, dtype=torch.float32),
        'lpips': torch.tensor([min(v, 1.0) for v in lpipss],
                              dtype=torch.float32),
    }


# ============================================================
# Training
# ============================================================

def train_epoch_robust(model, dataloader, optimizer, criterion, device, epoch,
                       lpips_fn, loss_weights=None, adv_ratio=0.33,
                       adv_steps=15, adv_lr=0.01, adv_mode='replace'):
    """
    Train one epoch with on-the-fly adversarial training.

    Every `1/adv_ratio` batches, the predicted image is adversarially
    perturbed and the model learns the real metrics of the perturbed image.

    adv_mode:
      'replace' — adversarial batch replaces normal training (original)
      'append'  — normal training always runs; adversarial is added on top
    """
    model.train()

    if loss_weights is None:
        loss_weights = {'psnr': 1.0, 'ssim': 1.0, 'lpips': 1.0}

    total_loss = 0
    total_psnr_loss = 0
    total_ssim_loss = 0
    total_lpips_loss = 0
    adv_count = 0
    normal_count = 0

    # How often to do adversarial: e.g. ratio=0.33 → every 3rd batch
    adv_interval = max(1, round(1.0 / adv_ratio)) if adv_ratio > 0 else 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        x, diopter, targets, gt = batch  # RobustDataset returns gt too
        x = x.to(device)
        diopter = diopter.to(device)
        gt = gt.to(device)

        is_adv_batch = (adv_interval > 0 and batch_idx % adv_interval == 0)

        if adv_mode == 'append':
            # === APPEND mode: always do normal, then optionally add adv ===
            # Step 1: Normal training (always)
            optimizer.zero_grad()
            predictions = model(x, diopter)

            psnr_target = targets['psnr'].to(device)
            ssim_target = targets['ssim'].to(device)
            lpips_target = targets['lpips'].to(device)

            psnr_loss = criterion(predictions['psnr'], psnr_target)
            ssim_loss = criterion(predictions['ssim'], ssim_target)
            lpips_loss = criterion(predictions['lpips'], lpips_target)

            loss = (loss_weights['psnr'] * psnr_loss +
                    loss_weights['ssim'] * ssim_loss +
                    loss_weights['lpips'] * lpips_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr_loss += psnr_loss.item()
            total_ssim_loss += ssim_loss.item()
            total_lpips_loss += lpips_loss.item()
            normal_count += 1

            # Step 2: Adversarial training (additional, on selected batches)
            if is_adv_batch:
                x_adv = generate_adversarial_batch(
                    model, x, diopter, device,
                    adv_steps=adv_steps, adv_lr=adv_lr
                )
                real_targets = compute_real_metrics_batch(
                    x_adv, gt, lpips_fn, device
                )

                model.train()
                optimizer.zero_grad()
                predictions = model(x_adv, diopter)

                psnr_target = real_targets['psnr'].to(device)
                ssim_target = real_targets['ssim'].to(device)
                lpips_target = real_targets['lpips'].to(device)

                psnr_loss = criterion(predictions['psnr'], psnr_target)
                ssim_loss = criterion(predictions['ssim'], ssim_target)
                lpips_loss = criterion(predictions['lpips'], lpips_target)

                adv_loss = (loss_weights['psnr'] * psnr_loss +
                            loss_weights['ssim'] * ssim_loss +
                            loss_weights['lpips'] * lpips_loss)

                adv_loss.backward()
                optimizer.step()

                total_loss += adv_loss.item()
                adv_count += 1

            # Skip the shared loss/backward below
            mode_str = 'A+N' if is_adv_batch else 'NRM'
            pbar.set_postfix({
                'mode': mode_str,
                'loss': f'{loss.item():.4f}',
                'adv': adv_count,
            })
            continue  # skip the shared block below

        # === REPLACE mode (original behavior) ===
        if is_adv_batch:
            # --- Adversarial batch (replaces normal) ---
            x_adv = generate_adversarial_batch(
                model, x, diopter, device,
                adv_steps=adv_steps, adv_lr=adv_lr
            )

            real_targets = compute_real_metrics_batch(
                x_adv, gt, lpips_fn, device
            )

            model.train()
            optimizer.zero_grad()
            predictions = model(x_adv, diopter)

            psnr_target = real_targets['psnr'].to(device)
            ssim_target = real_targets['ssim'].to(device)
            lpips_target = real_targets['lpips'].to(device)

            adv_count += 1
        else:
            # --- Normal batch ---
            optimizer.zero_grad()
            predictions = model(x, diopter)

            psnr_target = targets['psnr'].to(device)
            ssim_target = targets['ssim'].to(device)
            lpips_target = targets['lpips'].to(device)

            normal_count += 1

        psnr_loss = criterion(predictions['psnr'], psnr_target)
        ssim_loss = criterion(predictions['ssim'], ssim_target)
        lpips_loss = criterion(predictions['lpips'], lpips_target)

        loss = (loss_weights['psnr'] * psnr_loss +
                loss_weights['ssim'] * ssim_loss +
                loss_weights['lpips'] * lpips_loss)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr_loss += psnr_loss.item()
        total_ssim_loss += ssim_loss.item()
        total_lpips_loss += lpips_loss.item()

        mode_str = 'ADV' if is_adv_batch else 'NRM'
        pbar.set_postfix({
            'mode': mode_str,
            'loss': f'{loss.item():.4f}',
            'adv': adv_count,
        })

    num_batches = len(dataloader)
    total_updates = normal_count + adv_count
    print(f"  Batches: {normal_count} normal + {adv_count} adversarial "
          f"({adv_mode} mode, {total_updates} total updates)")
    return {
        'loss': total_loss / num_batches,
        'psnr_loss': total_psnr_loss / num_batches,
        'ssim_loss': total_ssim_loss / num_batches,
        'lpips_loss': total_lpips_loss / num_batches,
        'adv_batches': adv_count,
        'normal_batches': normal_count,
    }


def validate(model, dataloader, criterion, device, loss_weights=None):
    """Validate on validation set"""
    model.eval()

    if loss_weights is None:
        loss_weights = {'psnr': 1.0, 'ssim': 1.0, 'lpips': 1.0}

    total_loss = 0
    total_psnr_loss = 0
    total_ssim_loss = 0
    total_lpips_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            x, diopter, targets = batch[0], batch[1], batch[2]  # ignore gt
            x = x.to(device)
            diopter = diopter.to(device)
            psnr_target = targets['psnr'].to(device)
            ssim_target = targets['ssim'].to(device)
            lpips_target = targets['lpips'].to(device)

            predictions = model(x, diopter)

            psnr_loss = criterion(predictions['psnr'], psnr_target)
            ssim_loss = criterion(predictions['ssim'], ssim_target)
            lpips_loss = criterion(predictions['lpips'], lpips_target)
            loss = (loss_weights['psnr'] * psnr_loss +
                    loss_weights['ssim'] * ssim_loss +
                    loss_weights['lpips'] * lpips_loss)

            total_loss += loss.item()
            total_psnr_loss += psnr_loss.item()
            total_ssim_loss += ssim_loss.item()
            total_lpips_loss += lpips_loss.item()

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'psnr_loss': total_psnr_loss / num_batches,
        'ssim_loss': total_ssim_loss / num_batches,
        'lpips_loss': total_lpips_loss / num_batches
    }


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'robust_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    loss_weights = {
        'psnr': args.w_psnr,
        'ssim': args.w_ssim,
        'lpips': args.w_lpips,
    }
    print(f"Loss weights: {loss_weights}")
    print(f"Adversarial ratio: {args.adv_ratio:.0%} "
          f"(mode={args.adv_mode}, steps={args.adv_steps}, lr={args.adv_lr})")

    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Datasets — RobustDataset returns GT for adversarial metric computation
    train_dataset = RobustDataset(
        args.data_dir, split='train',
        include_augmented=args.use_augmented,
        include_weak=args.use_weak,
        include_aif=args.use_aif,
        clean_ratio=args.clean_ratio
    )
    val_dataset = RobustDataset(
        args.data_dir, split='val',
        include_augmented=args.use_augmented,
        include_weak=args.use_weak,
        include_aif=args.use_aif,
        clean_ratio=1.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = LossEstimationNet(
        use_spectral_norm=args.use_spectral_norm
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_spectral_norm:
        print("  Spectral Normalization: ENABLED")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from: {args.resume}")

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # LPIPS for real metric computation
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch_robust(
            model, train_loader, optimizer, criterion, device, epoch,
            lpips_fn=lpips_fn,
            loss_weights=loss_weights,
            adv_ratio=args.adv_ratio,
            adv_steps=args.adv_steps,
            adv_lr=args.adv_lr,
            adv_mode=args.adv_mode,
        )

        val_metrics = validate(
            model, val_loader, criterion, device,
            loss_weights=loss_weights
        )

        # Log
        for metric_name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'train/{metric_name}', value, epoch)
        for metric_name, value in val_metrics.items():
            writer.add_scalar(f'val/{metric_name}', value, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nTrain Loss: {train_metrics['loss']:.6f}")
        print(f"Val Loss:   {val_metrics['loss']:.6f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'loss_weights': loss_weights,
                'adv_ratio': args.adv_ratio,
                'use_spectral_norm': args.use_spectral_norm,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")

        # Checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

        scheduler.step(val_metrics['loss'])

    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Robust training with on-the-fly adversarial batches'
    )

    # Paths
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--resume', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)

    # Data
    parser.add_argument('--use_augmented', action='store_true')
    parser.add_argument('--use_weak', action='store_true')
    parser.add_argument('--use_aif', action='store_true')
    parser.add_argument('--clean_ratio', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--w_psnr', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=1.0)
    parser.add_argument('--w_lpips', type=float, default=1.0)

    # Model architecture
    parser.add_argument('--use_spectral_norm', action='store_true',
                        help='Apply spectral normalization to all Conv/Linear layers')

    # On-the-fly adversarial
    parser.add_argument('--adv_mode', type=str, default='append',
                        choices=['replace', 'append'],
                        help='replace: adv replaces normal batch; '
                             'append: normal always runs + adv added on top')
    parser.add_argument('--adv_ratio', type=float, default=0.33,
                        help='Fraction of batches to use adversarial training '
                             '(0.0=none, 0.33=1/3, 0.5=half, 1.0=all)')
    parser.add_argument('--adv_steps', type=int, default=15,
                        help='Pixel optimization steps per adversarial batch')
    parser.add_argument('--adv_lr', type=float, default=0.01,
                        help='Pixel optimization learning rate')

    args = parser.parse_args()
    main(args)
