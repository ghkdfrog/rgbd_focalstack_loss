import sys
import os
import re
import json
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Configuration (Assumed relative to this script position)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIFOCAL_DATA_DIR = os.path.join(BASE_DIR, '..', 'varifocal', 'data')

def load_exr(path, gamma=2.2):
    """Load and normalize EXR"""
    img = imageio.imread(path, format='EXR')
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return np.power(np.abs(img), 1.0/gamma)

def get_im_max(scene_dir):
    """Get max intensity from original clean pass for GT normalization"""
    rgb_path = os.path.join(scene_dir, 'clean_pass_rgb.exr')
    if not os.path.exists(rgb_path):
        return 1.0
    rgb = imageio.imread(rgb_path, format='EXR')
    if rgb.ndim == 3 and rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    rgb = np.power(np.abs(rgb), 1.0/2.2)
    return float(np.max(rgb))

def view_comparison(pred_path):
    print(f"Loading {pred_path}...")
    
    # 1. Parse path
    # Expected formats: 
    #   .../data/seedXXXX/pred_frameYYYY.exr 
    #   .../data/seedXXXX/aug_pred_frameYYYY.exr
    
    abs_path = os.path.abspath(pred_path)
    parts = abs_path.split(os.sep)
    
    try:
        # Find 'seedXXXX' in path
        scene_idx = next(i for i, part in enumerate(parts) if part.startswith('seed'))
        scene_name = parts[scene_idx]
        filename = parts[-1]
    except StopIteration:
        print("Error: Could not find 'seedXXXX' folder in path.")
        return

    # Check augmentation
    is_augmented = 'aug_' in filename
    
    # Extract frame number
    match = re.search(r'frame(\d{4})', filename)
    if not match:
        print("Error: Could not extract frame number from filename.")
        return
    frame_idx = int(match.group(1))
    
    # 2. Load Prediction
    pred_img = load_exr(abs_path)
    pred_img = np.clip(pred_img, 0, 1)
    
    # 3. Load Ground Truth
    gt_path = os.path.join(VARIFOCAL_DATA_DIR, scene_name, '512', f'frame{frame_idx:04d}.exr')
    
    if os.path.exists(gt_path):
        # Normalize GT using im_max logic matching training
        scene_dir = os.path.dirname(gt_path)
        im_max = get_im_max(scene_dir)
        
        gt_raw = imageio.imread(gt_path, format='EXR')
        if gt_raw.ndim == 3 and gt_raw.shape[2] > 3:
            gt_raw = gt_raw[:, :, :3]
        gt_img = np.power(np.abs(gt_raw), 1.0/2.2)
        gt_img = gt_img / (im_max + 1e-8)
        gt_img = np.clip(gt_img, 0, 1)
    else:
        print(f"Warning: GT file not found at {gt_path}")
        gt_img = np.zeros_like(pred_img)

    # 4. Load Metrics from labels.json
    labels_file = 'aug_labels.json' if is_augmented else 'labels.json'
    labels_path = os.path.join(os.path.dirname(abs_path), labels_file)
    
    metrics_str = "Metrics not found"
    
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            try:
                labels = json.load(f)
                # Find label for this frame
                # aug_labels might be list or dict, handled implicitly by index usually but augment logic saves list
                if frame_idx < len(labels):
                    m = labels[frame_idx]
                    
                    # MSE -> PSNR conversion if needed
                    psnr = m.get('psnr')
                    if psnr is None and 'mse' in m:
                        mse = m['mse']
                        psnr = -10 * np.log10(mse + 1e-10) if mse > 0 else 100.0
                        
                    metrics_str = (f"PSNR: {psnr:.2f} dB\n"
                                   f"SSIM: {m.get('ssim', 0):.4f}\n"
                                   f"LPIPS: {m.get('lpips', 0):.4f}")
                    if 'aug_type' in m:
                        metrics_str += f"\nAug: {m['aug_type']}"
            except Exception as e:
                print(f"Error reading labels: {e}")

    # 5. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prediction
    axes[0].imshow(pred_img)
    title_prefix = "[Augmented] " if is_augmented else "[Clean] "
    axes[0].set_title(f"{title_prefix}Prediction\n{filename}", fontsize=10)
    axes[0].axis('off')
    
    # GT
    axes[1].imshow(gt_img)
    axes[1].set_title(f"Ground Truth\nframe{frame_idx:04d}.exr", fontsize=10)
    axes[1].axis('off')
    
    # Metrics Text
    fig.suptitle(f"Scene: {scene_name} / Frame: {frame_idx}\n{metrics_str}", fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_exr.py <path_to_pred_exr>")
        sys.exit(1)
        
    view_comparison(sys.argv[1])
