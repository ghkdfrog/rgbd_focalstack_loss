"""
Focal Distance-Aware Dataset for Loss Estimation

Loads only DeepFocus predictions (clean DeepFocus output, no S/W/A augmentation).
Each sample is a (pred_image, query_diopter, real_metrics) triple, where:

  - Match sample:   pred[i] queried at diopter[i]  → model should give high score
  - Unmatch sample: pred[i] queried at diopter[j≠i] → model should give low score

Unmatch samples are re-sampled every epoch for variety.
Match samples are always included.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim_fn
import imageio
import lpips as lpips_lib

# Focal plane diopter values (must match data generation)
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)

def calculate_psnr(pred, gt):
    """Calculates PSNR between two tensors [0, 1]"""
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return torch.tensor(100.0).to(pred.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class FocalDataset(Dataset):
    """
    Args:
        data_dir:           Path to varifocal/data (raw EXR files)
        generated_data_dir: Path to rgbd_focalstack_loss/data (pred frames + labels)
        split:              'train' | 'val' | 'test'
        unmatch_ratio:      K unmatch samples per match sample (0 = match only)
        seed:               Random seed for reproducibility of initial unmatch draw
    """

    # Split boundaries
    SPLIT_RANGES = {
        'train': (0, 90),
        'val':   (90, 110),
        'test':  (110, 130),
    }

    def __init__(self, data_dir, generated_data_dir,
                 split='train', unmatch_ratio=3, seed=42,
                 use_coc=False, return_gt=False, single_scene_only=False):
        self.data_dir = data_dir
        self.generated_data_dir = generated_data_dir
        self.unmatch_ratio = unmatch_ratio
        self.use_coc = use_coc
        self.return_gt = return_gt
        self.single_scene_only = single_scene_only
        self.rng = np.random.default_rng(seed)

        # LPIPS must run on CPU (DataLoader workers are forked → no CUDA)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            self._lpips_fn = lpips_lib.LPIPS(net='alex')  # CPU
        self._lpips_fn.eval()
        for p in self._lpips_fn.parameters():
            p.requires_grad_(False)

        start, end = self.SPLIT_RANGES[split]
        self.scenes = list(range(start, end))
        


        # ── Pre-load all labels & check which (scene, plane) have pred frames ──
        self._valid = []   # list of (scene_idx, plane_idx)
        for scene_idx in self.scenes:
            scene_name = f'seed{scene_idx:04d}'
            gen_dir = os.path.join(generated_data_dir, scene_name)
            label_path = os.path.join(gen_dir, 'labels.json')
            if not os.path.exists(label_path):
                continue
            with open(label_path) as f:
                labels = json.load(f)
            for plane_idx in range(NUM_FOCAL_PLANES):
                pred_path = os.path.join(gen_dir, f'pred_frame{plane_idx:04d}.exr')
                if os.path.exists(pred_path) and plane_idx < len(labels):
                    self._valid.append((scene_idx, plane_idx))
            if self.single_scene_only and len(self._valid) > 0:
                self.scenes = [scene_idx]
                break

        # ── Build fixed match samples ──
        self._match_samples = [
            (scene_idx, plane_idx, plane_idx)  # (scene, pred_plane, query_plane)
            for (scene_idx, plane_idx) in self._valid
        ]

        # ── Build initial unmatch samples ──
        self._unmatch_samples = []
        if unmatch_ratio > 0:
            self._build_unmatch()

        self._samples = self._match_samples + self._unmatch_samples

        # ── Cache labels for fast lookup ──
        self._label_cache = {}
        self._unmatch_label_cache = {}  # loaded from labels_unmatch.json
        self._load_labels()

        # ── For RGB max normalization (per scene) ──
        self._scene_max_cache = {}

        print(f"[{split}] {len(self._match_samples)} match + "
              f"{len(self._unmatch_samples)} unmatch = {len(self._samples)} samples total")

    def _build_unmatch(self):
        """Sample K random unmatch query planes per valid (scene, pred_plane)."""
        self._unmatch_samples = []
        for (scene_idx, pred_plane) in self._valid:
            candidates = [p for p in range(NUM_FOCAL_PLANES) if p != pred_plane]
            chosen = self.rng.choice(candidates, size=self.unmatch_ratio, replace=False)
            for query_plane in chosen:
                self._unmatch_samples.append((scene_idx, pred_plane, int(query_plane)))

    def resample_unmatch(self):
        """Call at the start of each epoch to re-draw unmatch samples."""
        if self.unmatch_ratio > 0:
            self._build_unmatch()
            self._samples = self._match_samples + self._unmatch_samples

    def _load_labels(self):
        """Cache match labels and (if available) unmatch labels in memory."""
        for scene_idx in self.scenes:
            scene_name = f'seed{scene_idx:04d}'
            gen_dir = os.path.join(self.generated_data_dir, scene_name)

            # Match labels
            label_path = os.path.join(gen_dir, 'labels.json')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    self._label_cache[scene_idx] = json.load(f)

            # Unmatch labels (precomputed by generate_unmatch_labels.py)
            ul_path = os.path.join(gen_dir, 'labels_unmatch.json')
            if os.path.exists(ul_path):
                with open(ul_path) as f:
                    self._unmatch_label_cache[scene_idx] = json.load(f)

    def _get_scene_max(self, scene_idx):
        if scene_idx not in self._scene_max_cache:
            scene_name = f'seed{scene_idx:04d}'
            scene_dir = os.path.join(self.data_dir, scene_name, '512')
            rgb = imageio.v2.imread(
                os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR'
            )
            if rgb.shape[2] > 3:
                rgb = rgb[:, :, :3]
            rgb = np.power(np.abs(rgb), 1.0 / 2.2)
            self._scene_max_cache[scene_idx] = float(np.max(rgb)) + 1e-8
        return self._scene_max_cache[scene_idx]

    def _load_rgb_depth(self, scene_idx):
        """Load and normalize RGBD for a scene (always the same per scene)."""
        scene_name = f'seed{scene_idx:04d}'
        scene_dir = os.path.join(self.data_dir, scene_name, '512')
        im_max = self._get_scene_max(scene_idx)

        rgb = imageio.v2.imread(
            os.path.join(scene_dir, 'clean_pass_rgb.exr'), format='EXR'
        )
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb = np.power(np.abs(rgb), 1.0 / 2.2)
        rgb = np.clip(rgb / im_max, 0, 1).astype(np.float32)

        depth = imageio.v2.imread(
            os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'), format='EXR'
        )
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = (depth / 12.0).astype(np.float32)

        return rgb, depth

    def _load_pred(self, scene_idx, pred_plane):
        """Load DeepFocus prediction for a given (scene, plane)."""
        scene_name = f'seed{scene_idx:04d}'
        gen_dir = os.path.join(self.generated_data_dir, scene_name)
        pred_path = os.path.join(gen_dir, f'pred_frame{pred_plane:04d}.exr')
        pred = imageio.v2.imread(pred_path, format='EXR')
        if len(pred.shape) == 3 and pred.shape[2] > 3:
            pred = pred[:, :, :3]
        return np.clip(pred, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        scene_idx, pred_plane, query_plane = self._samples[idx]

        # Load RGBD
        rgb, depth = self._load_rgb_depth(scene_idx)

        # Load DeepFocus prediction (image being evaluated)
        pred = self._load_pred(scene_idx, pred_plane)

        # Diopter of the QUERY plane (what we're asking about)
        diopter = float(DP_FOCAL[query_plane])

        # ── Label: real metrics of pred[pred_plane] vs GT[query_plane] ──
        gt_loaded = False
        if pred_plane == query_plane:
            # Match: use pre-computed labels (fast path)
            label = self._label_cache[scene_idx][query_plane]
            mse  = float(label['mse'])
            ssim = float(label['ssim'])
            lpips_val = float(min(label['lpips'], 1.0))
            psnr = float(label.get('psnr', np.clip(-10 * np.log10(mse + 1e-10), 0, 100)))
        else:
            # Unmatch: GT[query_plane] != pred[pred_plane]
            key = f'{pred_plane}_{query_plane}'
            if (scene_idx in self._unmatch_label_cache and
                    key in self._unmatch_label_cache[scene_idx]):
                # Fast path: use precomputed labels
                ul = self._unmatch_label_cache[scene_idx][key]
                mse       = float(ul['mse'])
                ssim      = float(ul['ssim'])
                lpips_val = float(min(ul['lpips'], 1.0))
                psnr      = float(ul.get('psnr', np.clip(-10 * np.log10(mse + 1e-10), 0, 100)))
            else:
                # Slow fallback: compute on-the-fly (pre-compute not yet available)
                gt = self._load_gt(scene_idx, query_plane)
                gt_loaded = True
                mse  = float(np.mean((pred - gt) ** 2))
                psnr = float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100))
                ssim = float(ssim_fn(pred, gt, data_range=1.0, channel_axis=2))
                pred_t2 = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                gt_t2   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                with torch.no_grad():
                    lpips_val = float(self._lpips_fn(pred_t2, gt_t2).item())
                lpips_val = float(np.clip(lpips_val, 0.0, 1.0))

        if self.return_gt and not gt_loaded:
            gt = self._load_gt(scene_idx, query_plane)

        # Build input tensor
        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1).float()    # (3, H, W)
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()       # (1, H, W)
        pred_t  = torch.from_numpy(pred).permute(2, 0, 1).float()    # (3, H, W)

        if self.use_coc:
            # Compute CoC map on CPU (numpy) — avoid GPU overhead in forward pass
            # depth is already normalized (d/12.0 → [0,1]), recover meters
            depth_m = depth * 12.0 + 1e-6          # (H, W)
            dp_dm   = 1.0 / depth_m                 # depth diopter map
            dp_focal_val = float(diopter)
            film_len_dist = 0.017
            fov = 0.7854
            D = 0.004
            cocScale = 30.0
            dp_fl = 1.0 / film_len_dist + dp_focal_val
            coc_np = D * np.abs(
                (dp_fl - dp_dm) / (dp_fl - dp_focal_val + 1e-8) - 1.0
            )
            film_width = 2.0 * film_len_dist * np.tan(fov / 2.0)
            W = depth.shape[1]
            coc_np = np.clip(coc_np / film_width * W / cocScale, 0.0, 1.0).astype(np.float32)
            coc_t = torch.from_numpy(coc_np).unsqueeze(0).float()   # (1, H, W)
            x = torch.cat([rgb_t, depth_t, pred_t, coc_t], dim=0)  # (8, H, W)
        else:
            x = torch.cat([rgb_t, depth_t, pred_t], dim=0)          # (7, H, W)

        diopter_t = torch.tensor(diopter, dtype=torch.float32)

        targets = {
            'psnr':  torch.tensor(psnr,      dtype=torch.float32),
            'ssim':  torch.tensor(ssim,      dtype=torch.float32),
            'lpips': torch.tensor(lpips_val, dtype=torch.float32),
        }

        if self.return_gt:
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).float()
            return x, diopter_t, targets, gt_t

        return x, diopter_t, targets

    def _load_gt(self, scene_idx, plane_idx):
        """Load and normalize GT focal plane for unmatch metric computation."""
        scene_name = f'seed{scene_idx:04d}'
        scene_dir  = os.path.join(self.data_dir, scene_name, '512')
        im_max = self._get_scene_max(scene_idx)
        gt = imageio.v2.imread(
            os.path.join(scene_dir, f'frame{plane_idx:04d}.exr'), format='EXR'
        )
        if len(gt.shape) == 3 and gt.shape[2] > 3:
            gt = gt[:, :, :3]
        gt = np.power(np.abs(gt), 1.0 / 2.2)
        return np.clip(gt / im_max, 0, 1).astype(np.float32)


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'varifocal', 'data')
    gen_dir  = os.path.join(os.path.dirname(__file__), 'data')

    ds = FocalDataset(data_dir, gen_dir, split='train', unmatch_ratio=3)
    result = ds[0]
    x, d, t = result[:3]
    print(f"Sample 0: x={x.shape}, diopter={d:.3f}, "
          f"PSNR={t['psnr']:.2f}, SSIM={t['ssim']:.4f}, LPIPS={t['lpips']:.4f}")

    print("\nResample unmatch...")
    ds.resample_unmatch()
    print(f"Total samples after resample: {len(ds)}")
