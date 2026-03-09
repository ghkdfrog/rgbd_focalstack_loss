"""
Generate training dataset for RGBD Focal Stack Loss Estimation Model

This script:
1. Loads DeepFocus LFS model
2. Runs inference on all seed scenes
3. Saves predicted focal stacks as EXR
4. Computes MSE, SSIM, LPIPS against ground truth
5. Saves labels as JSON
"""
import os
import sys
import json
import numpy as np
import torch
import imageio
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips

# Add DeepFocus to path
DEEPFOCUS_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, DEEPFOCUS_DIR)

from deepfocus_pytorch import LFS, DeepFocusDataset, load_tf_weights_to_pytorch

# Paths
VARIFOCAL_DATA_DIR = os.path.join(DEEPFOCUS_DIR, 'varifocal/data')
CKPT_LFS = os.path.join(DEEPFOCUS_DIR, 'varifocal/model/saved_models-lfs/select/model-best.ckpt-309089')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED_START = 0
SEED_END = 129  # 130 scenes total
NUM_FOCAL_PLANES = 40
DP_FOCAL = np.linspace(0.1, 4.0, NUM_FOCAL_PLANES)

# Initialize LPIPS
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

def compute_metrics(pred, gt):
    """
    Compute MSE, SSIM, LPIPS between predicted and ground truth images
    
    Args:
        pred: (H, W, 3) float32 [0, 1]
        gt: (H, W, 3) float32 [0, 1]
    
    Returns:
        dict with mse, ssim, lpips values
    """
    # MSE
    mse = np.mean((pred - gt) ** 2)
    
    # SSIM (computed per channel, then averaged)
    ssim_val = ssim(pred, gt, multichannel=True, channel_axis=2, data_range=1.0)
    
    # LPIPS (requires tensors in [-1, 1] range, NCHW format)
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(DEVICE) * 2 - 1
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(DEVICE) * 2 - 1
    
    with torch.no_grad():
        lpips_val = lpips_fn(pred_tensor, gt_tensor).item()
    
    return {
        'mse': float(mse),
        'psnr': float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100)),
        'ssim': float(ssim_val),
        'lpips': float(lpips_val)
    }

def load_and_normalize_exr(path, im_max):
    """Load EXR and normalize using provided im_max"""
    img = imageio.imread(path, format='EXR')
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    img = np.power(np.abs(img), 1.0/2.2)
    img = img / (im_max + 1e-8)
    return np.clip(img, 0, 1)

def process_scene(model, scene_idx):
    """
    Process a single scene: run LFS inference and compute metrics
    
    Returns:
        metrics_per_plane: list of 40 dicts with mse/ssim/lpips
    """
    scene_name = f'seed{scene_idx:04d}'
    scene_dir = os.path.join(VARIFOCAL_DATA_DIR, scene_name, '512')
    output_scene_dir = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(output_scene_dir, exist_ok=True)
    
    # Load RGBD
    rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
    depth_path = os.path.join(scene_dir, 'clean_pass_depth_rgb.exr')
    
    rgb = imageio.imread(rgb_path, format='EXR')
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0/2.2)
    im_max = np.max(rgb)
    rgb = rgb / (im_max + 1e-8)
    
    depth = imageio.imread(depth_path, format='EXR')
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    # Compute diopter map
    diopter_map = 1.0 / (depth + 1e-6) / 4.0
    
    # Run LFS inference (3 times for R, G, B)
    H, W = depth.shape
    diopter_tensor = torch.from_numpy(diopter_map).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    outputs = []
    model.eval()
    with torch.no_grad():
        for c in range(3):
            intensity = rgb[:, :, c]
            intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            x = torch.cat([intensity_tensor, diopter_tensor], dim=1)  # (1, 2, H, W)
            out = model(x)  # (1, 40, H, W)
            outputs.append(out)
    
    # Combine: (1, 40, 3, H, W)
    outputs = torch.cat([o.unsqueeze(2) for o in outputs], dim=2)
    
    # Save predictions and compute metrics
    metrics_per_plane = []
    
    for i in range(NUM_FOCAL_PLANES):
        pred_plane = outputs[0, i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        pred_plane = np.clip(pred_plane, 0, 1)
        
        # Save as EXR
        pred_path = os.path.join(output_scene_dir, f'pred_frame{i:04d}.exr')
        imageio.imwrite(pred_path, pred_plane.astype(np.float32), format='EXR')
        
        # Load GT
        gt_path = os.path.join(scene_dir, f'frame{i:04d}.exr')
        gt_plane = load_and_normalize_exr(gt_path, im_max)
        
        # Compute metrics
        metrics = compute_metrics(pred_plane, gt_plane)
        metrics['diopter'] = float(DP_FOCAL[i])
        metrics_per_plane.append(metrics)
    
    return metrics_per_plane

def main():
    print(f"Using device: {DEVICE}")
    
    # Load DeepFocus LFS model
    print("Loading DeepFocus LFS model...")
    model = LFS().to(DEVICE)
    load_tf_weights_to_pytorch(model, CKPT_LFS)
    print("Model loaded successfully")
    
    # Process all scenes
    all_labels = {}
    
    for scene_idx in tqdm(range(SEED_START, SEED_END + 1), desc="Processing scenes"):
        scene_name = f'seed{scene_idx:04d}'
        
        try:
            metrics = process_scene(model, scene_idx)
            all_labels[scene_name] = metrics
            
            # Save individual scene labels
            scene_label_path = os.path.join(OUTPUT_DIR, scene_name, 'labels.json')
            with open(scene_label_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"\nError processing {scene_name}: {e}")
            continue
    
    # Save combined labels
    combined_label_path = os.path.join(OUTPUT_DIR, 'all_labels.json')
    with open(combined_label_path, 'w') as f:
        json.dump(all_labels, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Total scenes processed: {len(all_labels)}")
    print(f"Total samples: {len(all_labels) * NUM_FOCAL_PLANES}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
