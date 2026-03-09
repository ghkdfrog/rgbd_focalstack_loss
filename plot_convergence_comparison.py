
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Root directory
root = r'C:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\collected_results'

# Selected runs
target_runs = {
    'run_20260214_040344': 'Baseline (No Adv)',
    'run_20260214_040522': 'Epoch-based (Rep=20)',
    'robust_20260217_010547': 'Best Robust (Append, Ratio=0.33)'
}

def plot_combined_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['r', 'g', 'b']
    
    for idx, (run_id, label) in enumerate(target_runs.items()):
        json_path = os.path.join(root, run_id, 'verification', 'optimization_history.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: {run_id} verification data not found.")
            continue
            
        with open(json_path, 'r') as f:
            history = json.load(f)
            
        pred_psnr = history['pred_psnr']
        real_psnr = history['real_psnr']
        steps = range(len(pred_psnr))
        
        color = colors[idx]
        
        # Plot Pred PSNR
        ax1.plot(steps, pred_psnr, label=label, color=color, linewidth=2)
        
        # Plot Real PSNR
        ax2.plot(steps, real_psnr, label=label, color=color, linewidth=2)

    # Styling
    ax1.set_title('Predicted PSNR (Hallucination)', fontsize=14)
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Real PSNR (Actual Quality)', fontsize=14)
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('PSNR (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_convergence_comparison.png', dpi=150)
    print("Saved plot to optimization_convergence_comparison.png")

if __name__ == "__main__":
    plot_combined_convergence()
