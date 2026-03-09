"""
Generate augmented dataset for Loss Estimation Model

Supports four modes:
  --mode strong : Strongly degraded data (PSNR 15~35dB)
  --mode weak   : Weakly degraded data   (PSNR 30~40dB)
  --mode aif    : All-in-Focus labels only (no image files generated)
  --mode all    : Generate all three types

Files generated:
  Strong: aug_pred_frameXXXX.exr + aug_labels.json
  Weak:   weak_pred_frameXXXX.exr + weak_labels.json
  AiF:    aif_labels.json only (uses existing clean_pass_rgb.exr)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips

from augmentations import augment_image, augment_image_weak

DEEPFOCUS_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, DEEPFOCUS_DIR)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)

# LPIPS bounds
MAX_LPIPS = 1.0
MIN_LPIPS_WEAK = 0.02  # Weak must be distinguishable from Clean
MAX_RETRIES = 5

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)


def compute_metrics(pred, gt, device=DEVICE):
    """Compute MSE, SSIM, LPIPS"""
    mse = float(np.mean((pred - gt) ** 2))
    ssim_val = float(ssim(pred, gt, multichannel=True, channel_axis=2, data_range=1.0))
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        lpips_val = float(lpips_fn(pred_tensor, gt_tensor).item())
    return {
        'mse': mse,
        'psnr': float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100)),
        'ssim': ssim_val,
        'lpips': lpips_val
    }


def load_and_normalize_exr(path, im_max):
    img = imageio.imread(path, format='EXR')
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    img = np.power(np.abs(img), 1.0/2.2)
    img = img / (im_max + 1e-8)
    return np.clip(img, 0, 1).astype(np.float32)


def get_im_max(data_dir, scene_name):
    scene_dir = os.path.join(data_dir, scene_name, '512')
    rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
    rgb = imageio.imread(rgb_path, format='EXR')
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0/2.2)
    return float(np.max(rgb))


def process_scene_augmented(data_dir, scene_idx, rng, mode='strong'):
    """Generate Strong or Weak augmented data for one scene."""
    scene_name = f'seed{scene_idx:04d}'
    scene_dir = os.path.join(data_dir, scene_name, '512')
    generated_dir = os.path.join(OUTPUT_DIR, scene_name)

    if not os.path.exists(generated_dir):
        return None

    im_max = get_im_max(data_dir, scene_name)

    if mode == 'weak':
        prefix = 'weak_pred_frame'
        label_file = 'weak_labels.json'
        aug_fn = augment_image_weak
        min_lpips = MIN_LPIPS_WEAK
        max_lpips = MAX_LPIPS
    else:
        prefix = 'aug_pred_frame'
        label_file = 'aug_labels.json'
        aug_fn = augment_image
        min_lpips = 0.0
        max_lpips = MAX_LPIPS

    metrics_list = []

    for i in range(NUM_FOCAL_PLANES):
        pred_path = os.path.join(generated_dir, f'pred_frame{i:04d}.exr')
        if not os.path.exists(pred_path):
            continue

        pred = imageio.imread(pred_path, format='EXR')
        if len(pred.shape) == 3 and pred.shape[2] > 3:
            pred = pred[:, :, :3]
        pred = np.clip(pred, 0, 1).astype(np.float32)

        gt_path = os.path.join(scene_dir, f'frame{i:04d}.exr')
        gt = load_and_normalize_exr(gt_path, im_max)

        # Apply augmentation with LPIPS bounds check
        for retry in range(MAX_RETRIES):
            aug_pred, aug_type_name = aug_fn(pred, rng=rng)
            metrics = compute_metrics(aug_pred, gt)

            if min_lpips <= metrics['lpips'] <= max_lpips:
                break
            if retry == MAX_RETRIES - 1:
                metrics['lpips'] = np.clip(metrics['lpips'], min_lpips, max_lpips)

        out_path = os.path.join(generated_dir, f'{prefix}{i:04d}.exr')
        imageio.imwrite(out_path, aug_pred.astype(np.float32), format='EXR')

        metrics['diopter'] = float(DP_FOCAL[i])
        metrics['aug_type'] = aug_type_name
        metrics_list.append(metrics)

    label_path = os.path.join(generated_dir, label_file)
    with open(label_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)

    return metrics_list


def process_scene_aif(data_dir, scene_idx):
    """
    Generate All-in-Focus labels for one scene.
    No image files are created — uses existing clean_pass_rgb.exr.
    Computes metrics: AiF vs GT for each of the 40 focal planes.
    """
    scene_name = f'seed{scene_idx:04d}'
    scene_dir = os.path.join(data_dir, scene_name, '512')
    generated_dir = os.path.join(OUTPUT_DIR, scene_name)

    if not os.path.exists(generated_dir):
        return None

    im_max = get_im_max(data_dir, scene_name)

    # Load All-in-Focus image (normalized same way as GT)
    aif_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
    aif = load_and_normalize_exr(aif_path, im_max)

    metrics_list = []

    for i in range(NUM_FOCAL_PLANES):
        gt_path = os.path.join(scene_dir, f'frame{i:04d}.exr')
        if not os.path.exists(gt_path):
            continue

        gt = load_and_normalize_exr(gt_path, im_max)
        metrics = compute_metrics(aif, gt)
        metrics['diopter'] = float(DP_FOCAL[i])
        metrics['aug_type'] = 'all_in_focus'
        metrics_list.append(metrics)

    label_path = os.path.join(generated_dir, 'aif_labels.json')
    with open(label_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)

    return metrics_list


def run_mode(mode, data_dir, seed_start, seed_end, rng):
    """Run a single generation mode."""
    print(f"\n{'='*60}")
    print(f"Generating {mode.upper()} data...")
    print(f"{'='*60}")

    all_labels = {}

    for scene_idx in tqdm(range(seed_start, seed_end + 1), desc=f"{mode}"):
        scene_name = f'seed{scene_idx:04d}'
        try:
            if mode == 'aif':
                metrics = process_scene_aif(data_dir, scene_idx)
            else:
                metrics = process_scene_augmented(data_dir, scene_idx, rng, mode=mode)

            if metrics is not None:
                all_labels[scene_name] = metrics
        except Exception as e:
            print(f"\nError processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Statistics
    total = sum(len(v) for v in all_labels.values())
    print(f"\n{mode.upper()} complete! Total samples: {total}")

    if total > 0:
        all_mse = [m['mse'] for v in all_labels.values() for m in v]
        all_lpips_vals = [m['lpips'] for v in all_labels.values() for m in v]
        psnr_vals = [-10 * np.log10(m + 1e-10) for m in all_mse]
        print(f"PSNR range:  {min(psnr_vals):.1f} ~ {max(psnr_vals):.1f} dB")
        print(f"LPIPS range: {min(all_lpips_vals):.4f} ~ {max(all_lpips_vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Generate augmented dataset')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--seed_end', type=int, default=129)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='all',
                       choices=['strong', 'weak', 'aif', 'all'],
                       help='Generation mode')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Processing seeds {args.seed_start} to {args.seed_end}")
    print(f"Mode: {args.mode}")

    rng = np.random.default_rng(args.random_seed)

    if args.mode == 'all':
        modes = ['strong', 'weak', 'aif']
    else:
        modes = [args.mode]

    for mode in modes:
        run_mode(mode, args.data_dir, args.seed_start, args.seed_end, rng)

    print("\nAll done!")


if __name__ == "__main__":
    main()
