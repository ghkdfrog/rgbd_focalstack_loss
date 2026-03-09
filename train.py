"""
Training script for Loss Estimation Model

Supports:
  - Clean / Strong Aug / Weak Aug data loading
  - Per-metric loss weighting
  - Adversarial training (periodic pixel optimization to harden the model)
"""
import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import imageio
from skimage.metrics import structural_similarity as ssim_fn
import lpips

from model import LossEstimationNet
from dataset import LossEstimationDataset

# For adversarial training
DEEPFOCUS_DIR = os.path.join(os.path.dirname(__file__), '..')
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch,
                loss_weights=None):
    """Train for one epoch"""
    model.train()
    
    if loss_weights is None:
        loss_weights = {'psnr': 1.0, 'ssim': 1.0, 'lpips': 1.0}
    
    total_loss = 0
    total_psnr_loss = 0
    total_ssim_loss = 0
    total_lpips_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for x, diopter, targets in pbar:
        x = x.to(device)
        diopter = diopter.to(device)
        psnr_target = targets['psnr'].to(device)
        ssim_target = targets['ssim'].to(device)
        lpips_target = targets['lpips'].to(device)
        
        optimizer.zero_grad()
        predictions = model(x, diopter)
        
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
        
        pbar.set_postfix({
            'loss': loss.item(),
            'psnr': psnr_loss.item(),
            'ssim': ssim_loss.item(),
            'lpips': lpips_loss.item()
        })
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'psnr_loss': total_psnr_loss / num_batches,
        'ssim_loss': total_ssim_loss / num_batches,
        'lpips_loss': total_lpips_loss / num_batches
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
        for x, diopter, targets in tqdm(dataloader, desc='Validation'):
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
# Adversarial Training
# ============================================================

def generate_adversarial_samples(model, data_dir, device, lpips_fn,
                                  num_scenes=3, num_planes=2,
                                  adv_steps=15, adv_lr=0.01):
    """
    Generate adversarial samples by pixel-optimizing images to fool the model,
    then computing their real metrics as training targets.
    
    Args:
        model: LossEstimationNet (will be temporarily frozen)
        data_dir: path to varifocal/data
        device: torch device
        lpips_fn: LPIPS function for real metric computation
        num_scenes: number of random scenes to sample
        num_planes: number of random focal planes per scene
        adv_steps: pixel optimization steps
        adv_lr: pixel optimization learning rate
    
    Returns:
        list of adversarial sample dicts
    """
    model.eval()
    
    # Temporarily freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    adv_samples = []
    
    # Sample random scenes from train split (seed 0-89)
    rng = np.random.default_rng()
    scene_indices = rng.choice(90, size=min(num_scenes, 90), replace=False)
    
    for scene_idx in scene_indices:
        scene_name = f'seed{scene_idx:04d}'
        scene_dir = os.path.join(data_dir, scene_name, '512')
        gen_dir = os.path.join(generated_data_dir, scene_name)
        
        if not os.path.exists(gen_dir):
            continue
        
        # Load RGB + Depth
        rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
        depth_path = os.path.join(scene_dir, 'clean_pass_depth_rgb.exr')
        
        rgb = imageio.v2.imread(rgb_path, format='EXR')
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb = np.power(np.abs(rgb), 1.0/2.2)
        im_max = float(np.max(rgb))
        rgb = rgb / (im_max + 1e-8)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        
        depth = imageio.v2.imread(depth_path, format='EXR')
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = (depth / 12.0).astype(np.float32)
        
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
        
        # Sample random focal planes
        plane_indices = rng.choice(NUM_FOCAL_PLANES, size=min(num_planes, NUM_FOCAL_PLANES), replace=False)
        
        for plane_idx in plane_indices:
            diopter_val = float(DP_FOCAL[plane_idx])
            diopter_t = torch.tensor([diopter_val], dtype=torch.float32).to(device)
            
            # Load GT for real metric computation
            gt_path = os.path.join(scene_dir, f'frame{plane_idx:04d}.exr')
            gt = imageio.v2.imread(gt_path, format='EXR')
            if gt.shape[2] > 3:
                gt = gt[:, :, :3]
            gt = np.power(np.abs(gt), 1.0/2.2)
            gt = gt / (im_max + 1e-8)
            gt = np.clip(gt, 0, 1).astype(np.float32)
            
            # Initialize from clean RGB + noise
            init_img = rgb_t.clone()
            noise = torch.randn_like(init_img) * 0.05
            opt_img = torch.clamp(init_img + noise, 0, 1).requires_grad_(True)
            
            opt = torch.optim.Adam([opt_img], lr=adv_lr)
            
            # Pixel optimization (fool the model)
            for step in range(adv_steps):
                opt.zero_grad()
                with torch.no_grad():
                    opt_img.data.clamp_(0, 1)
                
                x = torch.cat([rgb_t, depth_t, opt_img], dim=1)
                pred = model(x, diopter_t)
                
                # Loss: try to maximize predicted quality
                loss = -pred['psnr'] - pred['ssim'] + pred['lpips']
                loss.backward()
                opt.step()
            
            # Compute REAL metrics of the adversarial image
            with torch.no_grad():
                adv_np = opt_img.data.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Real PSNR
            mse = float(np.mean((adv_np - gt) ** 2))
            real_psnr = -10 * np.log10(mse + 1e-10)
            
            # Real SSIM
            real_ssim = float(ssim_fn(adv_np, gt, multichannel=True, channel_axis=2, data_range=1.0))
            
            # Real LPIPS
            adv_t = torch.from_numpy(adv_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad():
                real_lpips = float(lpips_fn(adv_t, gt_t).item())
            
            adv_samples.append({
                'image': opt_img.data.clamp(0, 1).squeeze(0).cpu(),  # (3, H, W)
                'rgbd': torch.cat([rgb_t.squeeze(0), depth_t.squeeze(0)], dim=0).cpu(),  # (4, H, W)
                'diopter': diopter_val,
                'psnr': real_psnr,
                'ssim': real_ssim,
                'lpips': min(real_lpips, 1.0),
            })
    
    # Unfreeze model
    for param in model.parameters():
        param.requires_grad = True
    
    model.train()
    return adv_samples


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss weights
    loss_weights = {
        'psnr': args.w_psnr,
        'ssim': args.w_ssim,
        'lpips': args.w_lpips
    }
    print(f"Loss weights: PSNR={loss_weights['psnr']}, "
          f"SSIM={loss_weights['ssim']}, LPIPS={loss_weights['lpips']}")
    
    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Datasets
    train_dataset = LossEstimationDataset(
        args.data_dir, split='train',
        include_augmented=args.use_augmented,
        include_weak=args.use_weak,
        include_aif=args.use_aif,
        clean_ratio=args.clean_ratio
    )
    val_dataset = LossEstimationDataset(
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
    model = LossEstimationNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # LPIPS for adversarial training
    lpips_fn = None
    if args.adv_interval > 0:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
        print(f"Adversarial training enabled (every {args.adv_interval} epochs, "
              f"{args.adv_steps} steps, {args.adv_scenes} scenes x {args.adv_planes} planes)")
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # === Adversarial Training Step ===
        if args.adv_interval > 0 and epoch % args.adv_interval == 0 and epoch > 1:
            print(f"\n--- Adversarial Training (Epoch {epoch}) ---")
            adv_samples = generate_adversarial_samples(
                model, args.data_dir, device, lpips_fn,
                num_scenes=args.adv_scenes,
                num_planes=args.adv_planes,
                adv_steps=args.adv_steps,
                adv_lr=0.01
            )
            if adv_samples:
                # Log adversarial sample stats
                adv_psnrs = [s['psnr'] for s in adv_samples]
                adv_lpips_vals = [s['lpips'] for s in adv_samples]
                print(f"  Adversarial PSNR range: {min(adv_psnrs):.1f} ~ {max(adv_psnrs):.1f} dB")
                print(f"  Adversarial LPIPS range: {min(adv_lpips_vals):.4f} ~ {max(adv_lpips_vals):.4f}")
                writer.add_scalar('adv/mean_psnr', np.mean(adv_psnrs), epoch)
                writer.add_scalar('adv/mean_lpips', np.mean(adv_lpips_vals), epoch)
                
                # Train on adversarial samples (repeatedly)
                model.train()
                for adv_epoch in range(args.adv_train_epochs):
                    adv_loss_total = 0
                    for adv in adv_samples:
                        x = torch.cat([adv['rgbd'], adv['image']], dim=0).unsqueeze(0).to(device)
                        diopter = torch.tensor([adv['diopter']], dtype=torch.float32).to(device)
                        
                        optimizer.zero_grad()
                        pred = model(x, diopter)
                        
                        psnr_t = torch.tensor([adv['psnr']], dtype=torch.float32).to(device)
                        ssim_t = torch.tensor([adv['ssim']], dtype=torch.float32).to(device)
                        lpips_t = torch.tensor([adv['lpips']], dtype=torch.float32).to(device)
                        
                        loss = (loss_weights['psnr'] * criterion(pred['psnr'], psnr_t) +
                                loss_weights['ssim'] * criterion(pred['ssim'], ssim_t) +
                                loss_weights['lpips'] * criterion(pred['lpips'], lpips_t))
                        
                        loss.backward()
                        optimizer.step()
                        adv_loss_total += loss.item()
                    
                    if adv_epoch == 0 or adv_epoch == args.adv_train_epochs - 1:
                        avg_adv_loss = adv_loss_total / len(adv_samples)
                        print(f"  Adversarial training loss (iter {adv_epoch+1}/{args.adv_train_epochs}): {avg_adv_loss:.4f}")
                        if adv_epoch == args.adv_train_epochs - 1:
                             writer.add_scalar('adv/loss', avg_adv_loss, epoch)
        
        # === Regular Training ===
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            loss_weights=loss_weights
        )
        
        # === Validation ===
        val_metrics = validate(
            model, val_loader, criterion, device,
            loss_weights=loss_weights
        )
        
        # Log
        for metric_name, value in train_metrics.items():
            writer.add_scalar(f'train/{metric_name}', value, epoch)
        for metric_name, value in val_metrics.items():
            writer.add_scalar(f'val/{metric_name}', value, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nTrain Loss: {train_metrics['loss']:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'loss_weights': loss_weights,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint
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
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--resume', type=str, default=None)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Data
    parser.add_argument('--use_augmented', action='store_true',
                       help='Include strong augmented data')
    parser.add_argument('--use_weak', action='store_true',
                       help='Include weak augmented data')
    parser.add_argument('--use_aif', action='store_true',
                       help='Include all-in-focus data')
    parser.add_argument('--clean_ratio', type=float, default=1.0,
                       help='Fraction of clean samples to use (0.0~1.0)')
    
    # Loss weights
    parser.add_argument('--w_psnr', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=1.0)
    parser.add_argument('--w_lpips', type=float, default=1.0)
    
    # Adversarial training
    parser.add_argument('--adv_interval', type=int, default=1,
                       help='Adversarial training every N epochs (0=disabled)')
    parser.add_argument('--adv_steps', type=int, default=15,
                       help='Pixel optimization steps per adversarial sample')
    parser.add_argument('--adv_scenes', type=int, default=3,
                       help='Number of scenes to sample for adversarial training')
    parser.add_argument('--adv_planes', type=int, default=2,
                       help='Number of focal planes per scene for adversarial training')
    parser.add_argument('--adv_train_epochs', type=int, default=10,
                       help='Number of epochs to train on adversarial samples')
    
    args = parser.parse_args()
    main(args)
