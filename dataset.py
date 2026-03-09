"""
Dataset class for Loss Estimation Model training

Supports 4 data types: Clean, Strong Aug, Weak Aug, All-in-Focus (AiF)
AiF samples load clean_pass_rgb.exr directly (no duplicated files)
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio


class LossEstimationDataset(Dataset):
    def __init__(self, data_dir, seed_start=0, seed_end=129, split='train',
                 include_augmented=False, include_weak=False,
                 include_aif=False, clean_ratio=1.0):
        """
        Args:
            data_dir: Path to varifocal/data directory
            include_augmented: Load strong augmented samples
            include_weak: Load weak augmented samples
            include_aif: Load all-in-focus samples
            clean_ratio: Fraction of clean samples to keep (default 1.0)
        """
        self.data_dir = data_dir
        self.generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Split scenes
        all_scenes = list(range(seed_start, seed_end + 1))
        if split == 'train':
            self.scenes = all_scenes[:90]
        elif split == 'val':
            self.scenes = all_scenes[90:110]
        elif split == 'test':
            self.scenes = all_scenes[110:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Build sample list: (scene_idx, focal_plane_idx, sample_type)
        # sample_type: 'clean', 'strong', 'weak', 'aif'
        self.samples = []
        rng = np.random.default_rng(42)
        
        for scene_idx in self.scenes:
            scene_name = f'seed{scene_idx:04d}'
            
            # Clean
            label_path = os.path.join(self.generated_data_dir, scene_name, 'labels.json')
            if os.path.exists(label_path):
                for plane_idx in range(40):
                    if clean_ratio >= 1.0 or rng.random() < clean_ratio:
                        self.samples.append((scene_idx, plane_idx, 'clean'))
            
            # Strong augmented
            if include_augmented:
                aug_path = os.path.join(self.generated_data_dir, scene_name, 'aug_labels.json')
                if os.path.exists(aug_path):
                    with open(aug_path, 'r') as f:
                        aug_labels = json.load(f)
                    for plane_idx in range(len(aug_labels)):
                        self.samples.append((scene_idx, plane_idx, 'strong'))
            
            # Weak augmented
            if include_weak:
                weak_path = os.path.join(self.generated_data_dir, scene_name, 'weak_labels.json')
                if os.path.exists(weak_path):
                    with open(weak_path, 'r') as f:
                        weak_labels = json.load(f)
                    for plane_idx in range(len(weak_labels)):
                        self.samples.append((scene_idx, plane_idx, 'weak'))
            
            # All-in-Focus
            if include_aif:
                aif_path = os.path.join(self.generated_data_dir, scene_name, 'aif_labels.json')
                if os.path.exists(aif_path):
                    with open(aif_path, 'r') as f:
                        aif_labels = json.load(f)
                    for plane_idx in range(len(aif_labels)):
                        self.samples.append((scene_idx, plane_idx, 'aif'))
        
        # Count by type
        counts = {}
        for s in self.samples:
            counts[s[2]] = counts.get(s[2], 0) + 1
        count_str = ', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))
        print(f"[{split}] {len(self.samples)} samples ({count_str})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        scene_idx, plane_idx, sample_type = self.samples[idx]
        scene_name = f'seed{scene_idx:04d}'
        
        scene_dir = os.path.join(self.data_dir, scene_name, '512')
        generated_dir = os.path.join(self.generated_data_dir, scene_name)
        
        # Load RGBD (same for all types)
        rgb = self._load_exr(os.path.join(scene_dir, 'clean_pass_rgb.exr'))
        depth = self._load_exr(os.path.join(scene_dir, 'clean_pass_depth_rgb.exr'))
        
        # Normalize RGB
        rgb = np.power(np.abs(rgb), 1.0/2.2)
        im_max = np.max(rgb)
        rgb_norm = rgb / (im_max + 1e-8)
        
        # Depth
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = depth / 12.0
        
        # Load predicted focal plane based on type
        if sample_type == 'aif':
            # AiF: use normalized clean_pass_rgb directly
            pred = np.clip(rgb_norm, 0, 1).astype(np.float32)
            label_file = 'aif_labels.json'
        elif sample_type == 'weak':
            pred = self._load_exr(os.path.join(generated_dir, f'weak_pred_frame{plane_idx:04d}.exr'))
            label_file = 'weak_labels.json'
        elif sample_type == 'strong':
            pred = self._load_exr(os.path.join(generated_dir, f'aug_pred_frame{plane_idx:04d}.exr'))
            label_file = 'aug_labels.json'
        else:  # clean
            pred = self._load_exr(os.path.join(generated_dir, f'pred_frame{plane_idx:04d}.exr'))
            label_file = 'labels.json'
        
        # Load labels
        with open(os.path.join(generated_dir, label_file), 'r') as f:
            labels = json.load(f)
        plane_label = labels[plane_idx]
        
        # Convert MSE to PSNR
        mse = plane_label['mse']
        psnr = -10 * np.log10(mse + 1e-10) if mse > 0 else 100.0
        
        # Convert to tensors (HWC -> CHW)
        rgb_t = torch.from_numpy(np.clip(rgb_norm, 0, 1)).permute(2, 0, 1).float()
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        pred_t = torch.from_numpy(np.clip(pred, 0, 1)).permute(2, 0, 1).float()
        
        x = torch.cat([rgb_t, depth_t, pred_t], dim=0)  # (7, H, W)
        
        diopter = torch.tensor(plane_label['diopter'], dtype=torch.float32)
        
        targets = {
            'psnr': torch.tensor(psnr, dtype=torch.float32),
            'ssim': torch.tensor(plane_label['ssim'], dtype=torch.float32),
            'lpips': torch.tensor(min(plane_label['lpips'], 1.0), dtype=torch.float32),
        }
        
        return x, diopter, targets
    
    def _load_exr(self, path):
        img = imageio.imread(path, format='EXR')
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        return img.astype(np.float32)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'varifocal', 'data')
    
    print("=== Full 4-way dataset ===")
    ds = LossEstimationDataset(
        data_dir, split='train',
        include_augmented=True, include_weak=True, include_aif=True
    )
    print(f"Total: {len(ds)}")
    
    if len(ds) > 0:
        x, diopter, targets = ds[0]
        print(f"Sample: x={x.shape}, diopter={diopter:.2f}, "
              f"PSNR={targets['psnr']:.2f}, SSIM={targets['ssim']:.4f}, LPIPS={targets['lpips']:.4f}")
