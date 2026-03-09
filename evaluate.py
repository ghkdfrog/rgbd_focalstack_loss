"""
Evaluate trained Loss Estimation Model on test set

Supports evaluating with augmented data for comprehensive assessment.
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from model import LossEstimationNet
from dataset import LossEstimationDataset

def evaluate(model, dataloader, device):
    """Evaluate model and collect predictions"""
    model.eval()
    
    predictions = {'psnr': [], 'ssim': [], 'lpips': []}
    targets = {'psnr': [], 'ssim': [], 'lpips': []}
    diopters = []
    
    with torch.no_grad():
        for x, diopter, target in tqdm(dataloader, desc='Evaluating'):
            x = x.to(device)
            diopter = diopter.to(device)
            
            # Forward
            pred = model(x, diopter)
            
            # Collect (move to CPU)
            for metric in ['psnr', 'ssim', 'lpips']:
                predictions[metric].extend(pred[metric].cpu().numpy().tolist())
                targets[metric].extend(target[metric].numpy().tolist())
            diopters.extend(diopter.cpu().numpy().tolist())
    
    # Convert to numpy
    for metric in ['psnr', 'ssim', 'lpips']:
        predictions[metric] = np.array(predictions[metric])
        targets[metric] = np.array(targets[metric])
    diopters = np.array(diopters)
    
    return predictions, targets, diopters

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    results = {}
    
    for metric in ['psnr', 'ssim', 'lpips']:
        pred = predictions[metric]
        targ = targets[metric]
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred - targ))
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((pred - targ) ** 2))
        
        # Pearson correlation
        pearson, _ = pearsonr(pred, targ)
        
        # Spearman correlation
        spearman, _ = spearmanr(pred, targ)
        
        # Relative error
        relative_error = np.mean(np.abs(pred - targ) / (np.abs(targ) + 1e-8)) * 100
        
        results[metric] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'pearson': float(pearson),
            'spearman': float(spearman),
            'relative_error_%': float(relative_error)
        }
    
    return results

def compute_gt_stats(targets):
    """Compute statistics for ground truth values"""
    stats = {}
    
    for metric in ['psnr', 'ssim', 'lpips']:
        targ = targets[metric]
        stats[metric] = {
            'min': float(np.min(targ)),
            'max': float(np.max(targ)),
            'mean': float(np.mean(targ)),
            'std': float(np.std(targ))
        }
    
    return stats

def plot_predictions(predictions, targets, output_dir):
    """Plot predicted vs actual for each metric"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', 'tab:blue'),
        ('ssim', 'SSIM', 'tab:orange'),
        ('lpips', 'LPIPS', 'tab:green')
    ]
    
    for ax, (metric, title, color) in zip(axes, metrics_info):
        pred = predictions[metric]
        targ = targets[metric]
        
        # Scatter plot
        ax.scatter(targ, pred, alpha=0.5, s=10, c=color)
        
        # Ideal line (y=x)
        min_val = min(targ.min(), pred.min())
        max_val = max(targ.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Labels
        ax.set_xlabel(f'True {title}', fontsize=12)
        ax.set_ylabel(f'Predicted {title}', fontsize=12)
        ax.set_title(f'{title} Prediction', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_scatter.png'), dpi=150)
    print(f"Saved scatter plot to {output_dir}/predictions_scatter.png")
    plt.close()

def plot_gt_distribution(targets, output_dir):
    """Plot distribution of ground truth metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', 'tab:blue'),
        ('ssim', 'SSIM', 'tab:orange'),
        ('lpips', 'LPIPS', 'tab:green')
    ]
    
    for ax, (metric, title, color) in zip(axes, metrics_info):
        targ = targets[metric]
        
        ax.hist(targ, bins=50, alpha=0.7, color=color, edgecolor='black')
        
        # Add stats text
        t_min, t_max, t_mean = targ.min(), targ.max(), targ.mean()
        stats_text = f"Min: {t_min:.2f}\nMax: {t_max:.2f}\nMean: {t_mean:.2f}"
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'GT {title}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'GT {title} Distribution', fontsize=14)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gt_distribution.png'), dpi=150)
    print(f"Saved GT distribution to {output_dir}/gt_distribution.png")
    plt.close()

def plot_diopter_analysis(predictions, targets, diopters, output_dir):
    """Plot prediction error vs diopter for each metric"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Unique diopter values (should be 40 focal planes: 0.1 to 4.0)
    unique_diopters = np.unique(diopters)
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', 'tab:blue'),
        ('ssim', 'SSIM', 'tab:orange'),
        ('lpips', 'LPIPS', 'tab:green')
    ]
    
    for col, (metric, title, color) in enumerate(metrics_info):
        pred = predictions[metric]
        targ = targets[metric]
        errors = pred - targ
        abs_errors = np.abs(errors)
        
        # Top row: MAE per diopter
        ax_mae = axes[0, col]
        mae_per_diopter = []
        for d in unique_diopters:
            mask = diopters == d
            mae_per_diopter.append(np.mean(abs_errors[mask]))
        
        ax_mae.plot(unique_diopters, mae_per_diopter, 'o-', color=color, linewidth=2, markersize=6)
        ax_mae.set_xlabel('Diopter (D)', fontsize=12)
        ax_mae.set_ylabel(f'MAE ({title})', fontsize=12)
        ax_mae.set_title(f'{title} Error vs Focal Distance', fontsize=14)
        ax_mae.grid(alpha=0.3)
        
        # Bottom row: Scatter of predictions colored by diopter
        ax_scatter = axes[1, col]
        scatter = ax_scatter.scatter(targ, pred, c=diopters, s=10, alpha=0.5, cmap='viridis')
        
        # Ideal line
        min_val = min(targ.min(), pred.min())
        max_val = max(targ.max(), pred.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax_scatter.set_xlabel(f'True {title}', fontsize=12)
        ax_scatter.set_ylabel(f'Predicted {title}', fontsize=12)
        ax_scatter.set_title(f'{title} by Diopter', fontsize=14)
        ax_scatter.grid(alpha=0.3)
        
        # Colorbar for bottom row
        if col == 2:
            cbar = plt.colorbar(scatter, ax=axes[1, :], location='right', pad=0.02)
            cbar.set_label('Diopter (D)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diopter_analysis.png'), dpi=150)
    print(f"Saved diopter analysis to {output_dir}/diopter_analysis.png")
    plt.close()

def plot_error_distribution(predictions, targets, output_dir):
    """Plot error distribution for each metric"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', 'tab:blue'),
        ('ssim', 'SSIM', 'tab:orange'),
        ('lpips', 'LPIPS', 'tab:green')
    ]
    
    for ax, (metric, title, color) in zip(axes, metrics_info):
        errors = predictions[metric] - targets[metric]
        
        ax.hist(errors, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Prediction Error ({title})', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title} Error Distribution', fontsize=14)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
    print(f"Saved error distribution to {output_dir}/error_distribution.png")
    plt.close()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (auto-detect spectral normalization)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    use_sn = checkpoint.get('use_spectral_norm', False)
    if not use_sn:
        # Fallback: check state_dict keys for SN signature
        use_sn = any('weight_orig' in k for k in checkpoint['model_state_dict'])
    model = LossEstimationNet(use_spectral_norm=use_sn).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    if use_sn:
        print("  Spectral Normalization: detected")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'loss_weights' in checkpoint:
        print(f"Loss weights: {checkpoint['loss_weights']}")
    
    # Load dataset
    dataset = LossEstimationDataset(
        args.data_dir, split='test',
        include_augmented=args.use_augmented
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} test samples...")
    predictions, targets, diopters = evaluate(model, dataloader, device)
    
    # Compute metrics
    results = compute_metrics(predictions, targets)
    
    # Compute GT statistics
    gt_stats = compute_gt_stats(targets)
    
    # Merge GT stats into results
    final_results = {}
    for metric in ['psnr', 'ssim', 'lpips']:
        final_results[metric] = results[metric]
        final_results[metric]['gt_stats'] = gt_stats[metric]
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric in ['psnr', 'ssim', 'lpips']:
        print(f"\n{metric.upper()}:")
        for key, value in results[metric].items():
            if isinstance(value, dict):
                continue
            print(f"  {key}: {value:.6f}")
        print(f"  GT Range: {gt_stats[metric]['min']:.4f} ~ {gt_stats[metric]['max']:.4f}")
        print(f"  GT Mean: {gt_stats[metric]['mean']:.4f} (std: {gt_stats[metric]['std']:.4f})")
    
    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved results to {output_dir}/test_results.json")
    
    # Plot
    plot_predictions(predictions, targets, output_dir)
    plot_error_distribution(predictions, targets, output_dir)
    plot_gt_distribution(targets, output_dir)
    plot_diopter_analysis(predictions, targets, diopters, output_dir)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (best_model.pth)')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data',
                       help='Path to varifocal data directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_augmented', action='store_true',
                       help='Include augmented data in evaluation')
    
    args = parser.parse_args()
    main(args)
