"""
Standalone Focal Sweep Script

학습이 끝난 모델의 run 폴더를 지정하면, 사용 가능한 체크포인트
(latest, best, best_match_only)를 자동 감지하여 각각의 서브폴더에
diopter별 PSNR/SSIM/LPIPS 그래프를 저장합니다.

Usage:
    python run_focal_sweep.py \
        --run_dir runs/focal_coc_20260224_154940 \
        --scenes 110 115 120
"""
import os
import argparse
import json
import numpy as np
import torch
import imageio
import lpips as lpips_lib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

from model import LossEstimationNet

NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)
SAMPLE_PLANES = [0, 9, 19, 29, 39]  # ~0.1D, 1.0D, 2.0D, 3.0D, 4.0D

# Checkpoints to auto-detect, in order: (filename, subfolder name)
CHECKPOINT_CANDIDATES = [
    ('best_model.pth',            'best'),
    ('best_model_match_only.pth', 'best_match_only'),
    ('latest.pth',                'latest'),
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    diopter_mode = ckpt.get('diopter_mode', 'spatial')
    use_sn = ckpt.get('use_spectral_norm', False)
    version = ckpt.get('version', 'v1')
    
    model = LossEstimationNet(
        use_spectral_norm=use_sn, 
        diopter_mode=diopter_mode,
        version=version
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    criterion = ckpt.get('val_criterion', '')
    extra = f"  criterion={criterion}" if criterion else ""
    print(f"  Loaded: epoch={epoch}  diopter_mode={diopter_mode}  "
          f"val_loss={val_loss:.4f}{extra}")
    return model


def sweep_scene(model, scene_idx, data_dir, generated_data_dir, lpips_fn, device):
    """Run sweep for all sample planes in one scene. Returns dict of results."""
    scene_name = f'seed{scene_idx:04d}'
    scene_dir  = os.path.join(data_dir, scene_name, '512')
    gen_dir    = os.path.join(generated_data_dir, scene_name)

    # Load RGB / Depth
    rgb = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR')
    if rgb.shape[2] > 3: rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0 / 2.2)
    im_max = float(np.max(rgb)) + 1e-8
    rgb_norm = np.clip(rgb / im_max, 0, 1).astype(np.float32)

    depth = imageio.v2.imread(os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'), format='EXR')
    if len(depth.shape) == 3: depth = depth[:, :, 0]
    depth = (depth / 12.0).astype(np.float32)

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

    scene_results = {}

    for fixed_pi in SAMPLE_PLANES:
        pred_path = os.path.join(gen_dir, f'pred_frame{fixed_pi:04d}.exr')
        if not os.path.exists(pred_path):
            print(f"  [{scene_name}] pred_frame{fixed_pi:04d}.exr not found, skipping")
            continue

        pred = imageio.v2.imread(pred_path, format='EXR')
        if len(pred.shape) == 3 and pred.shape[2] > 3: pred = pred[:, :, :3]
        fixed_np = np.clip(pred, 0, 1).astype(np.float32)
        fixed_t  = torch.from_numpy(fixed_np).permute(2, 0, 1).unsqueeze(0).to(device)
        fixed_d  = float(DP_FOCAL[fixed_pi])

        real_psnr, model_psnr = [], []
        real_ssim, model_ssim = [], []
        real_lpips, model_lpips = [], []

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
                pt   = torch.from_numpy(fixed_np).permute(2,0,1).unsqueeze(0).to(device)*2-1
                gt_t = torch.from_numpy(gt_np).permute(2,0,1).unsqueeze(0).to(device)*2-1
                rl   = float(lpips_fn(pt, gt_t).item())

                # Model prediction
                d_t = torch.tensor([dval], dtype=torch.float32).to(device)
                x   = torch.cat([rgb_t, depth_t, fixed_t], dim=1)  # (1, 7, H, W)
                if model.diopter_mode == 'coc':
                    depth_np = depth_t.squeeze().cpu().numpy()  # (H, W)
                    depth_m = depth_np * 12.0 + 1e-6
                    dp_dm   = 1.0 / depth_m
                    film_len_dist = 0.017; fov = 0.7854; D = 0.004; cocScale = 30.0
                    dp_fl = 1.0 / film_len_dist + dval
                    coc_np = D * np.abs((dp_fl - dp_dm) / (dp_fl - dval + 1e-8) - 1.0)
                    film_width = 2.0 * film_len_dist * np.tan(fov / 2.0)
                    coc_np = np.clip(coc_np / film_width * 512.0 / cocScale, 0, 1).astype(np.float32)
                    coc_t  = torch.from_numpy(coc_np).unsqueeze(0).unsqueeze(0).to(device)
                    x = torch.cat([x, coc_t], dim=1)  # (1, 8, H, W)
                out = model(x, d_t)

                real_psnr.append(rp);  model_psnr.append(float(out['psnr'].item()) * 100.0)
                real_ssim.append(rs);  model_ssim.append(float(out['ssim'].item()))
                real_lpips.append(rl); model_lpips.append(float(out['lpips'].item()))

        diopters = [float(DP_FOCAL[qi]) for qi in range(NUM_FOCAL_PLANES) if qi in gt_frames]
        scene_results[fixed_pi] = {
            'fixed_diopter': fixed_d,
            'diopters': diopters,
            'real_psnr': real_psnr, 'model_psnr': model_psnr,
            'real_ssim': real_ssim, 'model_ssim': model_ssim,
            'real_lpips': real_lpips, 'model_lpips': model_lpips,
        }

    return scene_name, scene_results


def plot_sweep(scene_name, scene_results, sweep_dir):
    for fixed_pi, r in scene_results.items():
        fixed_d = r['fixed_diopter']
        diopters = r['diopters']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Scene {scene_name} | Fixed input: pred @ {fixed_d:.2f}D', fontsize=14)

        for ax, real_v, pred_v, ylabel in zip(
            axes,
            [r['real_psnr'], r['real_ssim'], r['real_lpips']],
            [r['model_psnr'], r['model_ssim'], r['model_lpips']],
            ['PSNR (dB)', 'SSIM', 'LPIPS'],
        ):
            ax.plot(diopters, real_v, 'k--', label='Real', linewidth=2, alpha=0.6)
            ax.plot(diopters, pred_v, 'b-o', label='Model', markersize=3)
            ax.axvline(x=fixed_d, color='green', linestyle=':', alpha=0.8,
                       label=f'Fixed @ {fixed_d:.2f}D')
            ax.set_title(ylabel)
            ax.set_xlabel('Query Diopter')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'sweep_{scene_name}_fixed_{fixed_d:.2f}D.png'
        plt.savefig(os.path.join(sweep_dir, fname), dpi=150)
        plt.close()
        print(f"    Saved: {fname}")


def run_sweep_for_checkpoint(ckpt_path, ckpt_label, sweep_base_dir,
                             scenes, data_dir, generated_data_dir, device, lpips_fn):
    """Load one checkpoint, run sweep, save to sweep_base_dir/ckpt_label/."""
    print(f"\n── [{ckpt_label}] {os.path.basename(ckpt_path)} ──")
    model = load_model(ckpt_path, device)

    sweep_dir = os.path.join(sweep_base_dir, ckpt_label)
    os.makedirs(sweep_dir, exist_ok=True)

    for scene_idx in scenes:
        print(f"  Scene seed{scene_idx:04d} ...")
        try:
            scene_name, results = sweep_scene(
                model, scene_idx, data_dir, generated_data_dir, lpips_fn, device
            )
            plot_sweep(scene_name, results, sweep_dir)
        except Exception as e:
            print(f"    Error: {e}")

    print(f"  → {sweep_dir}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    run_dir = args.run_dir
    generated_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Create top-level sweep folder: focal_sweep_{run_basename}
    run_basename = os.path.basename(os.path.normpath(run_dir))
    sweep_base_dir = os.path.join(run_dir, f'focal_sweep_{run_basename}')
    os.makedirs(sweep_base_dir, exist_ok=True)
    print(f"Sweep output: {sweep_base_dir}")

    # Init LPIPS once (shared across all checkpoints)
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    if args.ckpt:
        # Manual checkpoint override: single run
        label = os.path.splitext(os.path.basename(args.ckpt))[0]
        run_sweep_for_checkpoint(
            args.ckpt, label, sweep_base_dir,
            args.scenes, args.data_dir, generated_data_dir, device, lpips_fn
        )
    else:
        # Auto-detect all available checkpoints
        found = []
        for fname, label in CHECKPOINT_CANDIDATES:
            path = os.path.join(run_dir, fname)
            if os.path.exists(path):
                found.append((path, label))

        if not found:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")

        print(f"Found {len(found)} checkpoint(s): {[l for _, l in found]}")

        for ckpt_path, label in found:
            run_sweep_for_checkpoint(
                ckpt_path, label, sweep_base_dir,
                args.scenes, args.data_dir, generated_data_dir, device, lpips_fn
            )

    print(f"\nAll done → {sweep_base_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Focal Perception Sweep (multi-checkpoint)')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run folder (e.g. runs/focal_coc_20260224_154940)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Single checkpoint path override. '
                             'Default: auto-detect best, best_match_only, latest.')
    parser.add_argument('--scenes', type=int, nargs='+', default=[110, 115, 120],
                        help='Scene indices to evaluate')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    args = parser.parse_args()
    main(args)
