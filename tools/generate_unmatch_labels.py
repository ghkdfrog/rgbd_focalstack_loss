"""
generate_unmatch_labels.py
─────────────────────────
One-time precomputation of unmatch metrics for all (scene, pred_plane, query_plane) pairs.

For each scene:
  - Load all 40 pred frames + 40 GT frames
  - Compute MSE, SSIM, LPIPS for every (pred, GT) pair where pred_plane != query_plane
  - Save to data/seed{XXXX}/labels_unmatch.json

After running this, dataset_focal.py __getitem__ reads labels directly (no on-the-fly AlexNet).
Typical runtime: ~20 min (GPU) or ~6 hours (CPU).

Usage:
    python generate_unmatch_labels.py --data_dir ../varifocal/data
"""

import os
import json
import argparse
import numpy as np
import torch
import imageio
import lpips as lpips_lib
import contextlib
import io as _io
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

NUM_FOCAL_PLANES = 40
LPIPS_BATCH = 16   # pairs per GPU batch for LPIPS; reduce if OOM


def load_frame(path):
    """Load a pred EXR frame, return (H,W,3) float32 numpy array normalized [0,1].
    Pred frames are already gamma corrected and scaled, so no additional processing is needed."""
    img = imageio.v2.imread(path, format='EXR')
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return np.clip(img, 0, 1).astype(np.float32)


def load_gt(path, im_max):
    """Load a GT focal-stack EXR frame, return (H,W,3) float32 [0,1]."""
    img = imageio.v2.imread(path, format='EXR')
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    img = np.power(np.abs(img), 1.0 / 2.2)
    return np.clip(img / im_max, 0, 1).astype(np.float32)


def process_scene(scene_idx, data_dir, gen_dir, lpips_fn, device):
    """Compute and return unmatch labels dict for one scene."""
    scene_name = f'seed{scene_idx:04d}'
    scene_dir  = os.path.join(data_dir, scene_name, '512')
    scene_gen  = os.path.join(gen_dir, scene_name)

    # ── Load all 40 pred frames ──────────────────────────────────────────────
    pred_frames = {}
    for pi in range(NUM_FOCAL_PLANES):
        p = os.path.join(scene_gen, f'pred_frame{pi:04d}.exr')
        if os.path.exists(p):
            pred_frames[pi] = load_frame(p)

    if len(pred_frames) == 0:
        return None

    # ── Load all 40 GT frames ────────────────────────────────────────────────
    # We need the global im_max from clean_pass_rgb.exr to accurately normalize the gt frames
    rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
    if not os.path.exists(rgb_path):
        return None
    rgb = imageio.v2.imread(rgb_path, format='EXR')
    if len(rgb.shape) == 3 and rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0 / 2.2)
    im_max = float(np.max(rgb)) + 1e-8

    gt_frames = {}
    for pi in range(NUM_FOCAL_PLANES):
        p = os.path.join(scene_dir, f'frame{pi:04d}.exr')
        if os.path.exists(p):
            gt_frames[pi] = load_gt(p, im_max)

    if len(gt_frames) == 0:
        return None

    # ── Enumerate all unmatch pairs ──────────────────────────────────────────
    pairs = []
    for pred_pi, pred_img in pred_frames.items():
        for gt_pi, gt_img in gt_frames.items():
            if pred_pi != gt_pi:
                pairs.append((pred_pi, gt_pi, pred_img, gt_img))

    if len(pairs) == 0:
        return None

    # ── Compute MSE, PSNR, SSIM (CPU numpy) ──────────────────────────────────────
    mse_map  = {}
    psnr_map = {}
    ssim_map = {}
    for pred_pi, gt_pi, pred_img, gt_img in pairs:
        mse  = float(np.mean((pred_img - gt_img) ** 2))
        psnr = float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100))
        ssim = float(ssim_fn(pred_img, gt_img, data_range=1.0, channel_axis=2))
        mse_map[ (pred_pi, gt_pi)] = mse
        psnr_map[(pred_pi, gt_pi)] = psnr
        ssim_map[(pred_pi, gt_pi)] = ssim

    # ── Compute LPIPS in batches on GPU ──────────────────────────────────────
    # Convert all pairs to [-1,1] tensors
    pred_ts = [torch.from_numpy(p).permute(2,0,1).unsqueeze(0) * 2 - 1 for _,_,p,_ in pairs]
    gt_ts   = [torch.from_numpy(g).permute(2,0,1).unsqueeze(0) * 2 - 1 for _,_,_,g in pairs]
    pair_keys = [(pred_pi, gt_pi) for pred_pi, gt_pi, _, _ in pairs]

    lpips_map = {}
    with torch.no_grad():
        for i in range(0, len(pairs), LPIPS_BATCH):
            batch_p = torch.cat(pred_ts[i:i+LPIPS_BATCH], dim=0).to(device)
            batch_g = torch.cat(gt_ts[  i:i+LPIPS_BATCH], dim=0).to(device)
            vals = lpips_fn(batch_p, batch_g).squeeze().cpu().tolist()
            if isinstance(vals, float):
                vals = [vals]
            for j, v in enumerate(vals):
                key = pair_keys[i + j]
                lpips_map[key] = float(np.clip(v, 0.0, 1.0))

    # ── Assemble result dict ─────────────────────────────────────────────────
    labels = {}
    for pred_pi, gt_pi, _, _ in pairs:
        key = f'{pred_pi}_{gt_pi}'
        labels[key] = {
            'mse':   mse_map[ (pred_pi, gt_pi)],
            'psnr':  psnr_map[(pred_pi, gt_pi)],
            'ssim':  ssim_map[(pred_pi, gt_pi)],
            'lpips': lpips_map[(pred_pi, gt_pi)],
        }
    return labels


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    with contextlib.redirect_stdout(_io.StringIO()):
        lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    print('LPIPS model loaded.')

    gen_dir = os.path.join(os.path.dirname(__file__), 'data')

    scene_range = range(args.start_scene, args.end_scene)
    skipped = done = 0

    for scene_idx in tqdm(scene_range, desc='Scenes'):
        scene_name = f'seed{scene_idx:04d}'
        out_path = os.path.join(gen_dir, scene_name, 'labels_unmatch.json')

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        labels = process_scene(scene_idx, args.data_dir, gen_dir, lpips_fn, device)
        if labels is None:
            continue

        os.makedirs(os.path.join(gen_dir, scene_name), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(labels, f)
        done += 1

    print(f'Done. Processed {done} scenes, skipped {skipped} (already exist).')
    print(f'Re-run with --overwrite to force recompute.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='../varifocal/data')
    parser.add_argument('--start_scene', type=int, default=0,
                        help='Start scene index (inclusive)')
    parser.add_argument('--end_scene',   type=int, default=130,
                        help='End scene index (exclusive)')
    parser.add_argument('--overwrite',   action='store_true',
                        help='Recompute even if labels_unmatch.json already exists')
    args = parser.parse_args()
    main(args)
