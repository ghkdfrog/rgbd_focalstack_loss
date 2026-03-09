"""
Focal Distance-Aware Training Script

Key differences from train_robust.py:
  - Uses FocalDataset (match + unmatch focal plane pairs)
  - Unmatch samples re-sampled every epoch for variety
  - No adversarial training
  - Sinusoidal diopter conditioning via model.py `diopter_mode`

Usage:
    python train_focal.py \
        --data_dir ../varifocal/data \
        --diopter_mode sinusoidal \
        --unmatch_ratio 3 \
        --epochs 20
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
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import LossEstimationNet
from dataset_focal import FocalDataset


def train_epoch(model, loader, optimizer, criterion, device, epoch, loss_weights):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False, dynamic_ncols=True)
    for x, diopter, targets in pbar:
        x       = x.to(device)
        diopter = diopter.to(device)
        t_psnr  = targets['psnr'].to(device).view(-1, 1)
        t_ssim  = targets['ssim'].to(device).view(-1, 1)
        t_lpips = targets['lpips'].to(device).view(-1, 1)

        pred = model(x, diopter)

        loss = (loss_weights['psnr']  * criterion(pred['psnr'],  t_psnr / 100.0) +
                loss_weights['ssim']  * criterion(pred['ssim'],  t_ssim) +
                loss_weights['lpips'] * criterion(pred['lpips'], t_lpips))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return {'loss': total_loss / max(n, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, loss_weights, num_match=None):
    """Run validation.  When num_match is given (> 0), also track match-only loss
    by exploiting the fact that the dataset is ordered [match … | unmatch …]."""
    model.eval()
    total_loss = 0.0
    n = 0

    # Track match-only loss when dual tracking is enabled
    track_match = (num_match is not None and num_match > 0)
    match_loss = 0.0
    n_match = 0
    seen = 0  # running sample counter

    pbar = tqdm(loader, desc='           [val] ', leave=False, dynamic_ncols=True)
    for x, diopter, targets in pbar:
        bs = x.size(0)
        x       = x.to(device)
        diopter = diopter.to(device)
        t_psnr  = targets['psnr'].to(device).view(-1, 1)
        t_ssim  = targets['ssim'].to(device).view(-1, 1)
        t_lpips = targets['lpips'].to(device).view(-1, 1)

        pred = model(x, diopter)

        loss = (loss_weights['psnr']  * criterion(pred['psnr'],  t_psnr / 100.0) +
                loss_weights['ssim']  * criterion(pred['ssim'],  t_ssim) +
                loss_weights['lpips'] * criterion(pred['lpips'], t_lpips))

        total_loss += float(loss.item()) * bs
        n += bs

        # Accumulate match-only portion
        if track_match and seen < num_match:
            match_in_batch = min(bs, num_match - seen)
            if match_in_batch == bs:
                # Entire batch is match samples — reuse the already-computed loss
                match_loss += float(loss.item()) * bs
            else:
                # Partial batch: recompute loss for the match slice only
                m = match_in_batch
                ml = (loss_weights['psnr']  * criterion(pred['psnr'][:m],  t_psnr[:m] / 100.0) +
                      loss_weights['ssim']  * criterion(pred['ssim'][:m],  t_ssim[:m]) +
                      loss_weights['lpips'] * criterion(pred['lpips'][:m], t_lpips[:m]))
                match_loss += float(ml.item()) * m
            n_match += match_in_batch
        seen += bs

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    result = {'loss': total_loss / max(n, 1)}
    if track_match:
        result['match_loss'] = match_loss / max(n_match, 1)
    return result


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Output directory ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"focal_{args.diopter_mode}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Dataset ──
    generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    use_coc = (args.diopter_mode == 'coc')
    train_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='train', unmatch_ratio=args.unmatch_ratio,
        use_coc=use_coc
    )
    val_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='val', unmatch_ratio=args.val_unmatch_ratio,
        use_coc=use_coc
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Number of match samples — used to split match/unmatch loss in a single pass
    val_num_match = len(val_ds._match_samples) if args.val_unmatch_ratio > 0 else None

    # ── Model ──
    model = LossEstimationNet(
        use_spectral_norm=args.use_spectral_norm,
        diopter_mode=args.diopter_mode,
        sin_freqs=args.sin_freqs,
        version=args.version
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}  diopter_mode={args.diopter_mode}")

    # ── Resume ──
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    
    criterion = nn.MSELoss() if args.version == 'v2' else nn.L1Loss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    loss_weights = {
        'psnr':  args.w_psnr,
        'ssim':  args.w_ssim,
        'lpips': args.w_lpips,
    }

    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Re-sample unmatch samples every epoch
        if args.unmatch_ratio > 0:
            train_ds.resample_unmatch()
            print(f"  Resampled: {len(train_ds)} total samples "
                  f"({len(train_ds._match_samples)} match + "
                  f"{len(train_ds._unmatch_samples)} unmatch)")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers
        )

        train_metrics = train_epoch(model, train_loader, optimizer, criterion,
                                    device, epoch, loss_weights)
        val_metrics = validate(model, val_loader, criterion, device,
                              loss_weights, num_match=val_num_match)

        if 'match_loss' in val_metrics:
            print(f"Train Loss: {train_metrics['loss']:.4f}  |  "
                  f"Val Loss: {val_metrics['loss']:.4f}  |  "
                  f"Val(match-only) Loss: {val_metrics['match_loss']:.4f}")
        else:
            print(f"Train Loss: {train_metrics['loss']:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")

        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('val/loss',   val_metrics['loss'],   epoch)
        if 'match_loss' in val_metrics:
            writer.add_scalar('val_match_only/loss', val_metrics['match_loss'], epoch)

        scheduler.step(val_metrics['loss'])

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'diopter_mode': args.diopter_mode,
            'use_spectral_norm': args.use_spectral_norm,
            'version': args.version,
            'val_loss': val_metrics['loss'],
        }

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Always keep the best (based on primary val_unmatch_ratio)
        best_path = os.path.join(output_dir, 'best_model.pth')
        if not os.path.exists(best_path):
            torch.save(ckpt, best_path)
        else:
            prev = torch.load(best_path, map_location='cpu')
            if val_metrics['loss'] < prev.get('val_loss', float('inf')):
                torch.save(ckpt, best_path)
                print(f"  ** New best model saved (val_loss={val_metrics['loss']:.4f}) **")

        # Also keep best based on match-only loss (single-pass, no extra validation)
        if 'match_loss' in val_metrics:
            ckpt_mo = {
                **ckpt,
                'val_loss': val_metrics['match_loss'],
                'val_criterion': 'match_only',
            }
            best_mo_path = os.path.join(output_dir, 'best_model_match_only.pth')
            if not os.path.exists(best_mo_path):
                torch.save(ckpt_mo, best_mo_path)
            else:
                prev_mo = torch.load(best_mo_path, map_location='cpu')
                if val_metrics['match_loss'] < prev_mo.get('val_loss', float('inf')):
                    torch.save(ckpt_mo, best_mo_path)
                    print(f"  ** New best match-only model saved "
                          f"(val_match_loss={val_metrics['match_loss']:.4f}) **")

        torch.save(ckpt, os.path.join(output_dir, 'latest.pth'))

    writer.close()
    print(f"\nTraining complete. Results in: {output_dir}")

    # ── Post-training focal perception sweep (all checkpoints) ──
    if args.eval_scenes:
        run_basename = os.path.basename(os.path.normpath(output_dir))
        sweep_base = os.path.join(output_dir, f'focal_sweep_{run_basename}')
        os.makedirs(sweep_base, exist_ok=True)

        ckpt_candidates = [
            ('best_model.pth',            'best'),
            ('best_model_match_only.pth', 'best_match_only'),
            ('latest.pth',                'latest'),
        ]

        for ckpt_fname, ckpt_label in ckpt_candidates:
            ckpt_path = os.path.join(output_dir, ckpt_fname)
            if not os.path.exists(ckpt_path):
                continue
            ckpt_data = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            print(f"\n=== Sweep [{ckpt_label}] epoch={ckpt_data.get('epoch','?')} "
                  f"val_loss={ckpt_data.get('val_loss', float('nan')):.4f} ===")

            sweep_dir = os.path.join(sweep_base, ckpt_label)
            os.makedirs(sweep_dir, exist_ok=True)
            run_focal_sweep(model, args, sweep_dir, device)

        print(f"\nAll sweeps done → {sweep_base}")


def run_focal_sweep(model, args, output_dir, device):
    """After training, sweep all diopters for a fixed input image per test scene.
    Saves one PNG per (scene, fixed_input) combination to output_dir/.
    output_dir is the final destination directory (caller creates it).
    """
    import imageio
    import lpips as lpips_lib
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim_fn

    NUM_FOCAL_PLANES = 40
    DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)

    sweep_dir = output_dir  # output_dir is already the final sweep subdir

    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    model.eval()
    generated_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    for scene_idx in args.eval_scenes:
        scene_name = f'seed{scene_idx:04d}'
        scene_dir  = os.path.join(args.data_dir, scene_name, '512')
        gen_dir    = os.path.join(generated_data_dir, scene_name)

        # Load RGB/Depth
        try:
            rgb = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR')
            if rgb.shape[2] > 3: rgb = rgb[:, :, :3]
            rgb = np.power(np.abs(rgb), 1.0 / 2.2)
            im_max = float(np.max(rgb)) + 1e-8
            rgb_norm = np.clip(rgb / im_max, 0, 1).astype(np.float32)

            depth = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'), format='EXR')
            if len(depth.shape) == 3: depth = depth[:, :, 0]
            depth = (depth / 12.0).astype(np.float32)
        except Exception as e:
            print(f"  Skip scene {scene_idx}: {e}")
            continue

        rgb_t   = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

        # Load GT frames
        gt_frames = {}
        for pi in range(NUM_FOCAL_PLANES):
            try:
                gt = imageio.v2.imread(os.path.join(scene_dir, f'frame{pi:04d}.exr'), format='EXR')
                if gt.shape[2] > 3: gt = gt[:, :, :3]
                gt = np.power(np.abs(gt), 1.0 / 2.2)
                gt = np.clip(gt / im_max, 0, 1).astype(np.float32)
                gt_frames[pi] = gt
            except:
                pass

        # Choose fixed inputs: sample a few evenly-spaced planes + aif
        sample_planes = [0, 9, 19, 29, 39]  # 0.1D, 1.0D, 2.0D, 3.0D, 4.0D

        for fixed_pi in sample_planes:
            pred_path = os.path.join(gen_dir, f'pred_frame{fixed_pi:04d}.exr')
            if not os.path.exists(pred_path):
                continue
            pred = imageio.v2.imread(pred_path, format='EXR')
            if len(pred.shape) == 3 and pred.shape[2] > 3: pred = pred[:, :, :3]
            fixed_np = np.clip(pred, 0, 1).astype(np.float32)
            fixed_t  = torch.from_numpy(fixed_np).permute(2, 0, 1).unsqueeze(0).to(device)
            fixed_d  = float(DP_FOCAL[fixed_pi])

            real_psnr, pred_psnr = [], []
            real_ssim, pred_ssim = [], []
            real_lpips, pred_lpips = [], []

            with torch.no_grad():
                for qi in range(NUM_FOCAL_PLANES):
                    if qi not in gt_frames:
                        continue
                    gt_np = gt_frames[qi]
                    dval  = float(DP_FOCAL[qi])

                    # Real metrics
                    mse = float(np.mean((fixed_np - gt_np) ** 2))
                    rp  = float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100))
                    rs  = float(ssim_fn(fixed_np, gt_np, data_range=1.0, channel_axis=2))
                    pt  = torch.from_numpy(fixed_np).permute(2,0,1).unsqueeze(0).to(device)*2-1
                    gt_t2 = torch.from_numpy(gt_np).permute(2,0,1).unsqueeze(0).to(device)*2-1
                    rl  = float(lpips_fn(pt, gt_t2).item())

                    # Model prediction
                    d_t = torch.tensor([dval], dtype=torch.float32).to(device)
                    x   = torch.cat([rgb_t, depth_t, fixed_t], dim=1)  # (1, 7, H, W)
                    if args.diopter_mode == 'coc':
                        # Compute CoC map (numpy) and append as 8th channel
                        depth_np = depth_t.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                        depth_m  = depth_np * 12.0 + 1e-6
                        dp_dm    = 1.0 / depth_m
                        film_len_dist = 0.017; fov = 0.7854; D = 0.004; cocScale = 30.0
                        dp_fl = 1.0 / film_len_dist + dval
                        coc_np = D * np.abs((dp_fl - dp_dm) / (dp_fl - dval + 1e-8) - 1.0)
                        film_width = 2.0 * film_len_dist * np.tan(fov / 2.0)
                        coc_np = np.clip(coc_np / film_width * 512.0 / cocScale, 0, 1).astype(np.float32)
                        coc_t  = torch.from_numpy(coc_np).unsqueeze(0).unsqueeze(0).to(device)
                        x = torch.cat([x, coc_t], dim=1)  # (1, 8, H, W)
                    out = model(x, d_t)

                    real_psnr.append(rp);  pred_psnr.append(float(out['psnr'].item()) * 100.0)
                    real_ssim.append(rs);  pred_ssim.append(float(out['ssim'].item()))
                    real_lpips.append(rl); pred_lpips.append(float(out['lpips'].item()))

            diopters = [float(DP_FOCAL[qi]) for qi in range(NUM_FOCAL_PLANES) if qi in gt_frames]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Scene {scene_name} | Fixed input: pred @ {fixed_d:.2f}D', fontsize=14)

            for ax, real_v, pred_v, ylabel in zip(
                axes,
                [real_psnr, real_ssim, real_lpips],
                [pred_psnr, pred_ssim, pred_lpips],
                ['PSNR (dB)', 'SSIM', 'LPIPS'],
            ):
                ax.plot(diopters, real_v,  'k--', label='Real', linewidth=2, alpha=0.6)
                ax.plot(diopters, pred_v,  'b-o', label='Model', markersize=3)
                ax.axvline(x=fixed_d, color='green', linestyle=':', alpha=0.8,
                           label=f'Fixed @ {fixed_d:.2f}D')
                ax.set_title(ylabel); ax.set_xlabel('Query Diopter')
                ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f'sweep_{scene_name}_fixed_{fixed_d:.2f}D.png'
            plt.savefig(os.path.join(sweep_dir, fname), dpi=150)
            plt.close()
            print(f"  Saved: focal_sweep/{fname}")

    print(f"Focal sweep done → {sweep_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Focal Distance-Aware Training')

    # Data
    parser.add_argument('--data_dir', default='../varifocal/data')
    parser.add_argument('--output_dir', default='./runs')

    # Training
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--save_every',   type=int,   default=5)
    parser.add_argument('--resume',       type=str,   default=None)

    # Loss weights
    parser.add_argument('--w_psnr',  type=float, default=1.0)
    parser.add_argument('--w_ssim',  type=float, default=1.0)
    parser.add_argument('--w_lpips', type=float, default=1.0)

    # Model
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'],
                        help='Model version: v1 (Linear Head + L1 Loss) or v2 (MLP Head + MSE Loss)')
    parser.add_argument('--diopter_mode',      type=str, default='sinusoidal',
                        choices=['spatial', 'sinusoidal', 'coc'])
    parser.add_argument('--sin_freqs',         type=int, default=8,
                        help='Sinusoidal frequency bands (only for sinusoidal mode)')
    parser.add_argument('--use_spectral_norm', action='store_true')

    # Dataset
    parser.add_argument('--unmatch_ratio', type=int, default=3,
                        help='Unmatch samples per match sample. 0=match only.')
    parser.add_argument('--val_unmatch_ratio', type=int, default=39,
                        help='Unmatch ratio for validation set. 39=all pairs. '
                             'Requires generate_unmatch_labels.py to be run first for speed.')
    parser.add_argument('--eval_scenes', type=int, nargs='+', default=[110, 115, 120],
                        help='Test scene indices to run focal sweep on after training. '
                             'Pass 0 to skip (--eval_scenes 0).')

    args = parser.parse_args()
    main(args)

