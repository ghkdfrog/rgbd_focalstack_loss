"""
Verify Loss Estimation Model as a Differentiable Loss Function

This experiment directly optimizes image pixels using the frozen Loss Estimation Net
as the loss function. If the optimized image converges toward the ground truth,
it proves the model provides valid gradients for generator training.

Usage:
    python verify_as_loss.py --checkpoint runs/run_20260212_143739/best_model.pth
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
import lpips

from model import LossEstimationNet

# Config
DEEPFOCUS_DIR = os.path.join(os.path.dirname(__file__), '..')
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)


def load_scene_data(data_dir, scene_idx, plane_idx):
    """Load RGBD, GT focal plane, and normalization info for one scene+plane"""
    scene_name = f'seed{scene_idx:04d}'
    scene_dir = os.path.join(data_dir, scene_name, '512')
    
    # Load RGB
    rgb = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR')
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0/2.2)
    im_max = float(np.max(rgb))
    rgb = rgb / (im_max + 1e-8)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    
    # Load Depth
    depth = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'), format='EXR')
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    depth = depth / 12.0  # depthScale normalization
    depth = depth.astype(np.float32)
    
    # Load GT focal plane
    gt_path = os.path.join(scene_dir, f'frame{plane_idx:04d}.exr')
    gt = imageio.v2.imread(gt_path, format='EXR')
    if gt.shape[2] > 3:
        gt = gt[:, :, :3]
    gt = np.power(np.abs(gt), 1.0/2.2)
    gt = gt / (im_max + 1e-8)
    gt = np.clip(gt, 0, 1).astype(np.float32)
    
    # Diopter
    diopter = float(DP_FOCAL[plane_idx])
    
    return rgb, depth, gt, diopter


def compute_real_metrics(pred, gt, lpips_fn, device):
    """Compute actual PSNR, SSIM, LPIPS between pred and gt (numpy HWC [0,1])"""
    # PSNR
    mse = float(np.mean((pred - gt) ** 2))
    psnr = -10 * np.log10(mse + 1e-10)
    
    # SSIM
    ssim_val = float(ssim_fn(pred, gt, multichannel=True, channel_axis=2, data_range=1.0))
    
    # LPIPS
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        lpips_val = float(lpips_fn(pred_t, gt_t).item())
    
    return {'psnr': psnr, 'ssim': ssim_val, 'lpips': lpips_val}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory: save under the checkpoint's run folder
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    output_dir = os.path.join(checkpoint_dir, 'verification')
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Load Loss Estimation Net (FROZEN) ==========
    checkpoint = torch.load(args.checkpoint, map_location=device)
    use_sn = checkpoint.get('use_spectral_norm', False)
    if not use_sn:
        use_sn = any('weight_orig' in k for k in checkpoint['model_state_dict'])
    loss_net = LossEstimationNet(use_spectral_norm=use_sn).to(device)
    loss_net.load_state_dict(checkpoint['model_state_dict'])
    loss_net.eval()
    
    # Freeze all parameters
    for param in loss_net.parameters():
        param.requires_grad = False
    
    print(f"Loaded Loss Estimation Net from {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # ========== 2. Load LPIPS for real metric computation ==========
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    # ========== 3. Load scene data ==========
    scene_idx = args.scene_idx
    plane_idx = args.plane_idx
    
    print(f"\nLoading scene seed{scene_idx:04d}, plane {plane_idx} (diopter={DP_FOCAL[plane_idx]:.2f}D)")
    rgb, depth, gt, diopter_val = load_scene_data(args.data_dir, scene_idx, plane_idx)
    H, W = rgb.shape[:2]
    print(f"  Image size: {H}x{W}")
    
    # Prepare fixed inputs (RGBD + diopter) as tensors
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)       # (1, 3, H, W)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)        # (1, 1, H, W)
    diopter_t = torch.tensor([diopter_val], dtype=torch.float32).to(device)       # (1,)
    
    # ========== 4. Initialize optimizable image ==========
    # Start from All-in-Focus RGB (clean_pass_rgb)
    # This is much closer to the natural image manifold than random noise
    # We add small noise to give the optimizer something to "fix"
    init_image = rgb_t.clone()
    noise = torch.randn_like(init_image) * 0.05  # Add 5% noise to start
    init_image = torch.clamp(init_image + noise, 0, 1)
    
    optimizable_image = init_image.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([optimizable_image], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    
    # ========== 5. Optimization loop ==========
    print(f"\nStarting pixel optimization ({args.num_steps} steps, lr={args.lr})...")
    print(f"{'Step':>6} | {'Loss':>10} | {'Pred PSNR':>10} | {'Pred SSIM':>10} | {'Pred LPIPS':>10} | {'Real PSNR':>10} | {'Real SSIM':>10} | {'Real LPIPS':>10}")
    print("-" * 100)
    
    history = {
        'step': [], 'loss': [],
        'pred_psnr': [], 'pred_ssim': [], 'pred_lpips': [],
        'real_psnr': [], 'real_ssim': [], 'real_lpips': []
    }
    
    for step in range(args.num_steps + 1):
        optimizer.zero_grad()
        
        # Clamp image to [0, 1]
        with torch.no_grad():
            optimizable_image.data.clamp_(0, 1)
        
        # Build input: RGBD (4ch) + optimizable_image (3ch) = 7ch
        x = torch.cat([rgb_t, depth_t, optimizable_image], dim=1)  # (1, 7, H, W)
        
        # Forward through frozen Loss Estimation Net
        pred = loss_net(x, diopter_t)
        
        # Loss: maximize PSNR & SSIM, minimize LPIPS
        loss = -args.w_psnr * pred['psnr'] - args.w_ssim * pred['ssim'] + args.w_lpips * pred['lpips']
        
        # Backward (gradients flow to optimizable_image only)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Log every N steps
        if step % args.log_every == 0:
            with torch.no_grad():
                opt_np = optimizable_image.data.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                real_metrics = compute_real_metrics(opt_np, gt, lpips_fn, device)
            
            history['step'].append(step)
            history['loss'].append(float(loss.item()))
            history['pred_psnr'].append(float(pred['psnr'].item()))
            history['pred_ssim'].append(float(pred['ssim'].item()))
            history['pred_lpips'].append(float(pred['lpips'].item()))
            history['real_psnr'].append(real_metrics['psnr'])
            history['real_ssim'].append(real_metrics['ssim'])
            history['real_lpips'].append(real_metrics['lpips'])
            
            print(f"{step:6d} | {loss.item():10.4f} | {pred['psnr'].item():10.2f} | {pred['ssim'].item():10.4f} | {pred['lpips'].item():10.4f} | {real_metrics['psnr']:10.2f} | {real_metrics['ssim']:10.4f} | {real_metrics['lpips']:10.4f}")
    
    # ========== 6. Save results ==========
    # Final optimized image
    final_image = optimizable_image.data.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Save images
    imageio.v2.imwrite(os.path.join(output_dir, 'initial.png'), (init_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    imageio.v2.imwrite(os.path.join(output_dir, 'optimized.png'), (final_image * 255).astype(np.uint8))
    imageio.v2.imwrite(os.path.join(output_dir, 'ground_truth.png'), (gt * 255).astype(np.uint8))
    
    # Save history
    with open(os.path.join(output_dir, 'optimization_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========== 7. Plot convergence ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    steps = history['step']
    
    # Top row: Predicted metrics (from Loss Net)
    axes[0, 0].plot(steps, history['pred_psnr'], 'b-o', markersize=3)
    axes[0, 0].set_title('Predicted PSNR (from Loss Net)', fontsize=12)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(steps, history['pred_ssim'], 'orange', marker='o', markersize=3)
    axes[0, 1].set_title('Predicted SSIM (from Loss Net)', fontsize=12)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].grid(alpha=0.3)
    
    axes[0, 2].plot(steps, history['pred_lpips'], 'g-o', markersize=3)
    axes[0, 2].set_title('Predicted LPIPS (from Loss Net)', fontsize=12)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('LPIPS')
    axes[0, 2].grid(alpha=0.3)
    
    # Bottom row: Real metrics (computed with GT)
    axes[1, 0].plot(steps, history['real_psnr'], 'b-o', markersize=3)
    axes[1, 0].set_title('Real PSNR (vs GT)', fontsize=12)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(steps, history['real_ssim'], 'orange', marker='o', markersize=3)
    axes[1, 1].set_title('Real SSIM (vs GT)', fontsize=12)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].grid(alpha=0.3)
    
    axes[1, 2].plot(steps, history['real_lpips'], 'g-o', markersize=3)
    axes[1, 2].set_title('Real LPIPS (vs GT)', fontsize=12)
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('LPIPS')
    axes[1, 2].grid(alpha=0.3)
    
    fig.suptitle(f'Pixel Optimization Convergence\nScene: seed{scene_idx:04d}, Plane: {plane_idx}, Diopter: {diopter_val:.2f}D', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150)
    print(f"\nSaved convergence plot to {output_dir}/convergence.png")
    
    # ========== 8. Comparison plot ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    init_np = init_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    axes2[0].imshow(np.clip(init_np, 0, 1))
    axes2[0].set_title('Initial (Random)', fontsize=14)
    axes2[0].axis('off')
    
    axes2[1].imshow(np.clip(final_image, 0, 1))
    final_metrics = compute_real_metrics(final_image, gt, lpips_fn, device)
    axes2[1].set_title(f'Optimized\nPSNR: {final_metrics["psnr"]:.1f}dB  SSIM: {final_metrics["ssim"]:.3f}  LPIPS: {final_metrics["lpips"]:.3f}', fontsize=12)
    axes2[1].axis('off')
    
    axes2[2].imshow(np.clip(gt, 0, 1))
    axes2[2].set_title('Ground Truth', fontsize=14)
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    print(f"Saved comparison to {output_dir}/comparison.png")
    
    # ========== 9. Summary ==========
    init_metrics = compute_real_metrics(init_np, gt, lpips_fn, device)
    
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Metric':<10} {'Initial':>12} {'Final':>12} {'Improved?':>12}")
    print(f"{'-'*46}")
    
    psnr_improved = final_metrics['psnr'] > init_metrics['psnr']
    ssim_improved = final_metrics['ssim'] > init_metrics['ssim']
    lpips_improved = final_metrics['lpips'] < init_metrics['lpips']
    
    print(f"{'PSNR':<10} {init_metrics['psnr']:>10.2f}dB {final_metrics['psnr']:>10.2f}dB {'✅ YES' if psnr_improved else '❌ NO':>12}")
    print(f"{'SSIM':<10} {init_metrics['ssim']:>12.4f} {final_metrics['ssim']:>12.4f} {'✅ YES' if ssim_improved else '❌ NO':>12}")
    print(f"{'LPIPS':<10} {init_metrics['lpips']:>12.4f} {final_metrics['lpips']:>12.4f} {'✅ YES' if lpips_improved else '❌ NO':>12}")
    
    all_improved = psnr_improved and ssim_improved and lpips_improved
    print(f"\n{'🎉 VERDICT: Loss Estimation Net provides VALID gradients!' if all_improved else '⚠️  VERDICT: Gradients may not be fully effective.'}")
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify Loss Estimation Net as differentiable loss')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Loss Estimation Net checkpoint')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data',
                       help='Path to varifocal data directory')
    parser.add_argument('--scene_idx', type=int, default=110,
                       help='Scene index (default: 110, first test scene)')
    parser.add_argument('--plane_idx', type=int, default=20,
                       help='Focal plane index 0-39 (default: 20, ~2.0D)')
    parser.add_argument('--num_steps', type=int, default=300,
                       help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for pixel optimization')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log metrics every N steps')
    parser.add_argument('--w_psnr', type=float, default=1.0,
                       help='Weight for PSNR in loss')
    parser.add_argument('--w_ssim', type=float, default=1.0,
                       help='Weight for SSIM in loss')
    parser.add_argument('--w_lpips', type=float, default=1.0,
                       help='Weight for LPIPS in loss')
    
    args = parser.parse_args()
    main(args)
