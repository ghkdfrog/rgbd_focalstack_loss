"""
Check Data Distribution (Clean vs Strong vs Weak)

Visualizes the metric distribution of the generated dataset to verify
if the Weak Augmentation strategy successfully fills the gap between
Clean and Strong data.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_data(data_dir, output_dir):
    data = {'Clean': [], 'Strong': [], 'Weak': [], 'AiF': []}
    
    # Iterate through scenes
    for scene_idx in tqdm(range(130), desc="Loading labels"):
        scene_name = f'seed{scene_idx:04d}'
        gen_dir = os.path.join(output_dir, scene_name)
        
        if not os.path.exists(gen_dir):
            continue
            
        # 1. Clean
        label_path = os.path.join(gen_dir, 'labels.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = json.load(f)
            data['Clean'].extend(labels)
            
        # 2. Strong
        aug_path = os.path.join(gen_dir, 'aug_labels.json')
        if os.path.exists(aug_path):
            with open(aug_path, 'r') as f:
                labels = json.load(f)
            data['Strong'].extend(labels)
            
        # 3. Weak
        weak_path = os.path.join(gen_dir, 'weak_labels.json')
        if os.path.exists(weak_path):
            with open(weak_path, 'r') as f:
                labels = json.load(f)
            data['Weak'].extend(labels)
            
        # 4. AiF
        aif_path = os.path.join(gen_dir, 'aif_labels.json')
        if os.path.exists(aif_path):
            with open(aif_path, 'r') as f:
                labels = json.load(f)
            data['AiF'].extend(labels)
            
    return data

def plot_distribution(data, output_path='data_distribution.png'):
    metrics = {
        'PSNR': lambda x: -10 * np.log10(x['mse'] + 1e-10) if x['mse'] > 0 else 100,
        'SSIM': lambda x: x['ssim'],
        'LPIPS': lambda x: x['lpips']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Clean': 'green', 'Strong': 'red', 'Weak': 'orange', 'AiF': 'blue'}
    
    for idx, (metric_name, extractor) in enumerate(metrics.items()):
        ax = axes[idx]
        
        for dtype, items in data.items():
            if not items:
                continue
            values = [extractor(item) for item in items]
            
            # Clip outlier PSNR for better visualization
            if metric_name == 'PSNR':
                values = [v for v in values if 0 <= v <= 60]
            elif metric_name == 'LPIPS':
                values = [v for v in values if v <= 1.2]
                
            sns.kdeplot(values, ax=ax, label=f"{dtype} ({len(values)})", 
                        fill=True, alpha=0.3, color=colors[dtype])
            
        ax.set_title(f'{metric_name} Distribution')
        ax.set_xlabel(metric_name)
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved distribution plot to {output_path}")

if __name__ == "__main__":
    # Assuming standard path structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'data') # where labels are saved
    data_dir = os.path.join(script_dir, '../varifocal/data')
    
    print(f"Loading data from {output_dir}...")
    data = load_data(data_dir, output_dir)
    
    print("\nData counts:")
    for k, v in data.items():
        print(f"  {k}: {len(v)}")
        
    plot_distribution(data)
