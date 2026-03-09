"""
Defocus Blur Perception Test

Fix ONE input image (e.g., AiF or a specific focal plane prediction),
then ask the model to score it as if it were the prediction for EACH of the
40 diopter values.  A well-trained model should give a high score (PSNR)
at the matching diopter and lower scores at mismatched diopters.

Usage:
    # Use AiF (all-in-focus / clean RGB) as the fixed input
    python compare_models.py \
        --ckpt_a runs/run_20260214_040344/best_model.pth \
        --ckpt_b runs/robust_20260217_010547/best_model.pth \
        --scene_idx 110 \
        --fixed_input aif

    # Use DeepFocus prediction at 0.1D as the fixed input
    python compare_models.py \
        --ckpt_a runs/run_20260214_040344/best_model.pth \
        --ckpt_b runs/robust_20260217_010547/best_model.pth \
        --scene_idx 110 \
        --fixed_input 0.1
"""
import os
import argparse
import json
import numpy as np
import torch
import imageio
import lpips
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt

from model import LossEstimationNet

# Focal plane diopter values (must match training data)
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)


def load_model(ckpt_path, device):
    """Load model from checkpoint, auto-detecting spectral norm."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    use_sn = checkpoint.get('use_spectral_norm', False)
    model = LossEstimationNet(use_spectral_norm=use_sn).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_scene(data_dir, generated_data_dir, scene_idx):
    """Load scene RGB, depth, GT frames, and DeepFocus predictions."""
    scene_name = f'seed{scene_idx:04d}'
    scene_dir = os.path.join(data_dir, scene_name, '512')
    gen_dir = os.path.join(generated_data_dir, scene_name)

    # RGB
    rgb = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR')
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0 / 2.2)
    im_max = float(np.max(rgb))
    rgb_norm = np.clip(rgb / (im_max + 1e-8), 0, 1).astype(np.float32)

    # Depth
    depth = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'), format='EXR')
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    depth = (depth / 12.0).astype(np.float32)

    # GT frames
    gt_frames = {}
    for pi in range(NUM_FOCAL_PLANES):
        gt_path = os.path.join(scene_dir, f'frame{pi:04d}.exr')
        gt = imageio.v2.imread(gt_path, format='EXR')
        if gt.shape[2] > 3:
            gt = gt[:, :, :3]
        gt = np.power(np.abs(gt), 1.0 / 2.2)
        gt = np.clip(gt / (im_max + 1e-8), 0, 1).astype(np.float32)
        gt_frames[pi] = gt

    # DeepFocus predictions
    pred_frames = {}
    for pi in range(NUM_FOCAL_PLANES):
        pred_path = os.path.join(gen_dir, f'pred_frame{pi:04d}.exr')
        if os.path.exists(pred_path):
            pred = imageio.v2.imread(pred_path, format='EXR')
            if len(pred.shape) == 3 and pred.shape[2] > 3:
                pred = pred[:, :, :3]
            pred = np.clip(pred, 0, 1).astype(np.float32)
            pred_frames[pi] = pred

    return rgb_norm, depth, gt_frames, pred_frames, im_max


def compute_real_metrics(pred_np, gt_np, lpips_fn, device):
    """Compute real PSNR, SSIM, LPIPS between pred and GT."""
    mse = float(np.mean((pred_np - gt_np) ** 2))
    psnr = -10 * np.log10(mse + 1e-10) if mse > 0 else 100.0
    s = float(ssim_fn(pred_np, gt_np, data_range=1.0, channel_axis=2))
    with torch.no_grad():
        p_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        g_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        lp = float(lpips_fn(p_t, g_t).item())
    return {'psnr': psnr, 'ssim': s, 'lpips': min(lp, 1.0)}


def run_model(model, rgb_t, depth_t, input_t, diopter_val, device):
    """Run model on a single input with a given diopter."""
    diopter_t = torch.tensor([diopter_val], dtype=torch.float32).to(device)
    x = torch.cat([rgb_t, depth_t, input_t], dim=1)  # (1, 7, H, W)
    with torch.no_grad():
        out = model(x, diopter_t)
    return {
        'psnr': float(out['psnr'].item()),
        'ssim': float(out['ssim'].item()),
        'lpips': float(out['lpips'].item()),
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = args.data_dir
    generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Load models
    print(f"Loading Model A: {args.label_a} ({args.ckpt_a})")
    model_a = load_model(args.ckpt_a, device)
    print(f"Loading Model B: {args.label_b} ({args.ckpt_b})")
    model_b = load_model(args.ckpt_b, device)

    # LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # Load scene
    print(f"\nLoading scene seed{args.scene_idx:04d} ...")
    rgb_norm, depth, gt_frames, pred_frames, im_max = load_scene(
        data_dir, generated_data_dir, args.scene_idx
    )

    rgb_t = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

    # --- Determine the FIXED input image ---
    fixed_input_str = args.fixed_input.strip().lower()

    if fixed_input_str == 'aif':
        # All-in-Focus = clean RGB itself
        fixed_np = rgb_norm.copy()
        fixed_label = 'AiF (Clean RGB)'
        fixed_diopter = None  # no matching diopter
        print(f"Fixed input: AiF (clean_pass_rgb)")
    else:
        # Parse diopter value
        target_diopter = float(fixed_input_str)
        # Find closest diopter index
        dists = [abs(d - target_diopter) for d in DP_FOCAL]
        fixed_plane_idx = int(np.argmin(dists))
        fixed_diopter = DP_FOCAL[fixed_plane_idx]
        print(f"Fixed input: Diopter {target_diopter} -> matched plane {fixed_plane_idx} (diopter={fixed_diopter})")

        if fixed_plane_idx in pred_frames:
            fixed_np = pred_frames[fixed_plane_idx]
            fixed_label = f'DeepFocus pred @ {fixed_diopter}D'
        else:
            # Fallback to GT
            fixed_np = gt_frames[fixed_plane_idx]
            fixed_label = f'GT @ {fixed_diopter}D (pred not found)'

    fixed_t = torch.from_numpy(fixed_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- Sweep all diopters ---
    results = {
        'scene_idx': args.scene_idx,
        'fixed_input': fixed_input_str,
        'fixed_label': fixed_label,
        'label_a': args.label_a,
        'label_b': args.label_b,
        'planes': [],
    }

    print(f"\nSweeping {NUM_FOCAL_PLANES} diopters with fixed input: {fixed_label}\n")
    header = (f"{'Plane':>5} {'Diopter':>7} | {'Real PSNR':>9} {'Real SSIM':>9} | "
              f"{args.label_a:>12} {args.label_b:>12} | "
              f"{'A SSIM':>7} {'B SSIM':>7}")
    print(header)
    print("-" * len(header))

    for pi in range(NUM_FOCAL_PLANES):
        diopter_val = float(DP_FOCAL[pi])
        gt_np = gt_frames[pi]

        # Real metrics: fixed input vs GT at this diopter
        real = compute_real_metrics(fixed_np, gt_np, lpips_fn, device)

        # Model predictions: "this fixed image is the prediction for diopter_val"
        pred_a = run_model(model_a, rgb_t, depth_t, fixed_t, diopter_val, device)
        pred_b = run_model(model_b, rgb_t, depth_t, fixed_t, diopter_val, device)

        plane_result = {
            'plane_idx': pi,
            'diopter': diopter_val,
            'real': real,
            'pred_a': pred_a,
            'pred_b': pred_b,
        }
        results['planes'].append(plane_result)

        marker = ' <-- match' if fixed_diopter is not None and abs(diopter_val - fixed_diopter) < 0.01 else ''
        print(f"{pi:>5} {diopter_val:>7.2f} | {real['psnr']:>9.2f} {real['ssim']:>9.4f} | "
              f"{pred_a['psnr']:>12.2f} {pred_b['psnr']:>12.2f} | "
              f"{pred_a['ssim']:>7.4f} {pred_b['ssim']:>7.4f}{marker}")

    # Save JSON
    out_name = f"compare_scene{args.scene_idx:04d}_fixed_{fixed_input_str.replace('.','p')}"
    json_path = f"{out_name}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # ---- Plot ----
    planes_data = results['planes']
    diopters = [p['diopter'] for p in planes_data]

    real_psnr = [p['real']['psnr'] for p in planes_data]
    pred_a_psnr = [p['pred_a']['psnr'] for p in planes_data]
    pred_b_psnr = [p['pred_b']['psnr'] for p in planes_data]

    real_ssim = [p['real']['ssim'] for p in planes_data]
    pred_a_ssim = [p['pred_a']['ssim'] for p in planes_data]
    pred_b_ssim = [p['pred_b']['ssim'] for p in planes_data]

    real_lpips = [p['real']['lpips'] for p in planes_data]
    pred_a_lpips = [p['pred_a']['lpips'] for p in planes_data]
    pred_b_lpips = [p['pred_b']['lpips'] for p in planes_data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Scene seed{args.scene_idx:04d} | Fixed input: {fixed_label}', fontsize=14)

    # PSNR
    axes[0].plot(diopters, real_psnr, 'k--', label='Real', linewidth=2, alpha=0.6)
    axes[0].plot(diopters, pred_a_psnr, 'ro-', label=f'{args.label_a}', markersize=4)
    axes[0].plot(diopters, pred_b_psnr, 'bs-', label=f'{args.label_b}', markersize=4)
    if fixed_diopter is not None:
        axes[0].axvline(x=fixed_diopter, color='green', linestyle=':', alpha=0.7, label=f'Fixed @ {fixed_diopter}D')
    axes[0].set_title('PSNR (dB)')
    axes[0].set_xlabel('Query Diopter')
    axes[0].set_ylabel('PSNR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SSIM
    axes[1].plot(diopters, real_ssim, 'k--', label='Real', linewidth=2, alpha=0.6)
    axes[1].plot(diopters, pred_a_ssim, 'ro-', label=f'{args.label_a}', markersize=4)
    axes[1].plot(diopters, pred_b_ssim, 'bs-', label=f'{args.label_b}', markersize=4)
    if fixed_diopter is not None:
        axes[1].axvline(x=fixed_diopter, color='green', linestyle=':', alpha=0.7, label=f'Fixed @ {fixed_diopter}D')
    axes[1].set_title('SSIM')
    axes[1].set_xlabel('Query Diopter')
    axes[1].set_ylabel('SSIM')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LPIPS
    axes[2].plot(diopters, real_lpips, 'k--', label='Real', linewidth=2, alpha=0.6)
    axes[2].plot(diopters, pred_a_lpips, 'ro-', label=f'{args.label_a}', markersize=4)
    axes[2].plot(diopters, pred_b_lpips, 'bs-', label=f'{args.label_b}', markersize=4)
    if fixed_diopter is not None:
        axes[2].axvline(x=fixed_diopter, color='green', linestyle=':', alpha=0.7, label=f'Fixed @ {fixed_diopter}D')
    axes[2].set_title('LPIPS')
    axes[2].set_xlabel('Query Diopter')
    axes[2].set_ylabel('LPIPS')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = f"{out_name}.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Defocus Blur Perception Test: fix one image, sweep all diopters'
    )
    parser.add_argument('--ckpt_a', type=str, required=True,
                        help='Checkpoint path for Model A')
    parser.add_argument('--ckpt_b', type=str, required=True,
                        help='Checkpoint path for Model B')
    parser.add_argument('--label_a', type=str, default='Baseline')
    parser.add_argument('--label_b', type=str, default='Best Robust')
    parser.add_argument('--scene_idx', type=int, default=110,
                        help='Scene index (default: 110, test split)')
    parser.add_argument('--fixed_input', type=str, default='aif',
                        help='"aif" for all-in-focus, or a diopter value like "0.1"')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    args = parser.parse_args()
    main(args)
