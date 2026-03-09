"""
Score-based Generative Model Training Script (Gradient Matching)

Usage:
    python train_score.py \
        --data_dir ../varifocal/data \
        --diopter_mode sinusoidal \
        --unmatch_ratio 3 \
        --epochs 20 \
        --gm_steps 50 \
        --gm_step_size 0.2 \
        --single_scene_only
"""
import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model_score import EnergyNet
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr
import matplotlib.pyplot as plt

def train_epoch_score(model, loader, optimizer, device, epoch, gm_steps, gm_step_size):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False, dynamic_ncols=True)
    for batch_data in pbar:
        # Load batch
        x, diopter, targets, gt = batch_data
        
        # x is (N, 7, H, W) for spatial/sinusoidal, (N, 8, H, W) for coc
        # Split into condition (RGBD) and current initializing image
        # x has: RGB (3), Depth (1), Pred_RGB (3), [CoC (1)]
        # We will replace Pred_RGB (channels 4,5,6) with our optimizing RGB
        
        N, C, H, W = x.shape
        x = x.to(device)
        diopter = diopter.to(device)
        gt = gt.to(device)  # (N, 3, H, W)

        # 1. Initialize random noise for generation
        current_image = torch.randn_like(gt).to(device)
        
        epoch_loss = 0.0
        
        optimizer.zero_grad()

        # 2. Sequential trajectory steps
        for step in range(gm_steps):
            current_image.requires_grad_(True)
            
            # Construct input by replacing the focal plane channels (idx 4,5,6)
            # with current_image.
            # Slice indices: 0:4 is RGBD, 4:7 is the focal plane, 7: is CoC (if exists)
            input_rgbd = x[:, :4, :, :]
            if C > 7:
                input_tail = x[:, 7:, :, :]
                model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
            else:
                model_input = torch.cat([input_rgbd, current_image], dim=1)
            
            # Forward pass to get energy
            energy = model(model_input, diopter)
            
            # Compute gradient of energy w.r.t current_image
            pred_grad = torch.autograd.grad(
                outputs=energy,
                inputs=current_image,
                grad_outputs=torch.ones_like(energy),
                create_graph=True
            )[0]
            
            # Ground truth direction
            gt_grad = gt - current_image
            
            # Loss computation
            loss = F.mse_loss(pred_grad, gt_grad)
            
            # Accumulate gradients
            loss.backward()
            epoch_loss += loss.item()
            
            # Update current image for next step
            with torch.no_grad():
                current_image = (current_image + gm_step_size * pred_grad).detach()
        
        optimizer.step()
        avg_step_loss = epoch_loss / gm_steps
        total_loss += avg_step_loss * N
        n += N
        pbar.set_postfix(loss=f'{avg_step_loss:.4f}')

    return {'loss': total_loss / max(n, 1)}


@torch.no_grad()
def validate_score(model, loader, device, gm_steps, gm_step_size):
    model.eval()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc='           [val] ', leave=False, dynamic_ncols=True)
    for batch_data in pbar:
        x, diopter, targets, gt = batch_data
        
        N, C, H, W = x.shape
        x = x.to(device)
        diopter = diopter.to(device)
        gt = gt.to(device)

        with torch.enable_grad():
            current_image = torch.randn_like(gt).to(device)
            batch_loss = 0.0
            
            for step in range(gm_steps):
                current_image.requires_grad_(True)
                
                input_rgbd = x[:, :4, :, :]
                if C > 7:
                    input_tail = x[:, 7:, :, :]
                    model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                else:
                    model_input = torch.cat([input_rgbd, current_image], dim=1)
                
                energy = model(model_input, diopter)
                
                pred_grad = torch.autograd.grad(
                    outputs=energy,
                    inputs=current_image,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=False
                )[0]
                
                gt_grad = gt - current_image
                loss = F.mse_loss(pred_grad, gt_grad)
                batch_loss += loss.item()
                
                with torch.no_grad():
                    current_image = (current_image + gm_step_size * pred_grad).detach()
                    
        avg_step_loss = batch_loss / gm_steps
        total_loss += avg_step_loss * N
        n += N
        pbar.set_postfix(loss=f'{avg_step_loss:.4f}')

    return {'loss': total_loss / max(n, 1)}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scene_str = "scene0_" if args.single_scene_only else ""
    run_name = f"score_{scene_str}{args.diopter_mode}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    use_coc = (args.diopter_mode == 'coc')
    
    train_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='train', unmatch_ratio=args.unmatch_ratio,
        use_coc=use_coc, return_gt=True, single_scene_only=args.single_scene_only
    )
    val_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='val', unmatch_ratio=args.val_unmatch_ratio,
        use_coc=use_coc, return_gt=True, single_scene_only=args.single_scene_only
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Model
    model = EnergyNet(
        use_spectral_norm=args.use_spectral_norm,
        diopter_mode=args.diopter_mode,
        sin_freqs=args.sin_freqs
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}  diopter_mode={args.diopter_mode}")

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        if args.unmatch_ratio > 0:
            train_ds.resample_unmatch()
            print(f"  Resampled: {len(train_ds)} total samples")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers
        )

        train_metrics = train_epoch_score(model, train_loader, optimizer,
                                          device, epoch, args.gm_steps, args.gm_step_size)
        val_metrics = validate_score(model, val_loader, device, args.gm_steps, args.gm_step_size)

        print(f"Train Loss: {train_metrics['loss']:.6f}  |  Val Loss: {val_metrics['loss']:.6f}")

        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('val/loss',   val_metrics['loss'],   epoch)

        scheduler.step(val_metrics['loss'])

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'diopter_mode': args.diopter_mode,
            'use_spectral_norm': args.use_spectral_norm,
            'val_loss': val_metrics['loss'],
        }

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

        best_path = os.path.join(output_dir, 'best_model.pth')
        if not os.path.exists(best_path):
            torch.save(ckpt, best_path)
        else:
            prev = torch.load(best_path, map_location='cpu')
            if val_metrics['loss'] < prev.get('val_loss', float('inf')):
                torch.save(ckpt, best_path)
                print(f"  ** New best model saved (val_loss={val_metrics['loss']:.6f}) **")

        torch.save(ckpt, os.path.join(output_dir, 'latest.pth'))

    writer.close()
    print(f"\nTraining complete. Results in: {output_dir}")

    # ── Final generation check (Prototyping) ──
    print("\n[Final Check] Generating representative focal planes (0.1D, 2.0D, 4.0D)...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'))['model_state_dict'])
    model.eval()

    # Target diopters: 0.1D, 2.0D, 4.0D
    # DP_FOCAL = np.linspace(0.1, 4.0, 40)
    # 0.1D -> idx 0, 2.0D -> idx 19 or 20, 4.0D -> idx 39
    target_planes = [0, 20, 39]
    fig, axes = plt.subplots(len(target_planes), 2, figsize=(10, 5 * len(target_planes)))

    for i, q_idx in enumerate(target_planes):
        d_val = float(DP_FOCAL[q_idx])
        print(f"  Generating plane {q_idx} ({d_val:.1f}D)...")
        
        # Find sample
        sample_idx = 0
        for idx, (s, p, q) in enumerate(train_ds._match_samples):
            if q == q_idx:
                sample_idx = idx
                break
        
        x, diopter, targets, gt = train_ds[sample_idx]
        x       = x.unsqueeze(0).to(device)
        diopter = diopter.unsqueeze(0).to(device)
        gt      = gt.unsqueeze(0).to(device)
        
        with torch.enable_grad():
            current_image = torch.randn_like(gt).to(device)
            # Use 50 steps for a clean final evaluation
            for step in range(1, 51):
                current_image.requires_grad_(True)
                N, C, H, W = x.shape
                input_rgbd = x[:, :4, :, :]
                if C > 7:
                    input_tail = x[:, 7:, :, :]
                    model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                else:
                    model_input = torch.cat([input_rgbd, current_image], dim=1)
                
                energy = model(model_input, diopter)
                pred_grad = torch.autograd.grad(energy, current_image, torch.ones_like(energy))[0]
                
                with torch.no_grad():
                    current_image = (current_image + args.gm_step_size * pred_grad).detach()
        
        final_img = torch.clamp(current_image, 0, 1).cpu().squeeze().permute(1, 2, 0).numpy()
        gt_img    = gt.cpu().squeeze().permute(1, 2, 0).numpy()
        psnr_val  = calculate_psnr(current_image.cpu(), gt.cpu()).item()

        axes[i, 0].imshow(gt_img); axes[i, 0].set_title(f"GT {d_val:.1f}D"); axes[i, 0].axis('off')
        axes[i, 1].imshow(final_img); axes[i, 1].set_title(f"Gen {d_val:.1f}D (PSNR: {psnr_val:.2f}dB)"); axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_generation_check.png"))
    print(f"Final generation check image saved to {output_dir}/final_generation_check.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--diopter_mode', type=str, default='sinusoidal', choices=['spatial', 'coc', 'sinusoidal'])
    parser.add_argument('--unmatch_ratio', type=int, default=3)
    parser.add_argument('--val_unmatch_ratio', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)  # Much smaller due to graph retain
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_spectral_norm', action='store_true')
    parser.add_argument('--sin_freqs', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)
    
    # Gradient matching specific args
    parser.add_argument('--gm_steps', type=int, default=50, help='Number of trajectory steps per training sample')
    parser.add_argument('--gm_step_size', type=float, default=0.2, help='Step size mapped to predicted gradient')
    parser.add_argument('--single_scene_only', action='store_true', help='Only use scene 0 for quick prototyping')
    
    args = parser.parse_args()
    main(args)
