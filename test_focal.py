import os
import glob
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import LossEstimationNet
from dataset_focal import FocalDataset

CHECKPOINT_CANDIDATES = [
    'latest.pth',
    'best_model_match_only.pth',
    'best_model.pth',
]

def load_model(ckpt_path, device):
    global version
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
    print(f"  Loaded model from {os.path.basename(ckpt_path)}")
    print(f"  > Epoch: {epoch} | Version: {version} | Mode: {diopter_mode} | Val Loss: {val_loss:.4f}")
    return model, version, diopter_mode

def run_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory not found: {args.run_dir}")
        return

    # Find valid checkpoints
    found_checkpoints = []
    for ckpt_name in CHECKPOINT_CANDIDATES:
        ckpt_path = os.path.join(args.run_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            found_checkpoints.append(ckpt_path)
            
    if not found_checkpoints:
        print(f"Error: No valid checkpoints found in {args.run_dir}")
        print(f"Looked for: {CHECKPOINT_CANDIDATES}")
        return
        
    print(f"Found {len(found_checkpoints)} checkpoint(s) to evaluate.")
    
    # We need to initialize dataset. We'll peek at the first model to get 'diopter_mode'
    # assuming all checkpoints in a run_dir have the same mode.
    dummy_model, dummy_version, dummy_mode = load_model(found_checkpoints[0], device)
    del dummy_model
    
    generated_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    use_coc = (dummy_mode == 'coc')
    
    test_ds = FocalDataset(
        data_dir=args.data_dir, 
        generated_data_dir=generated_data_dir,
        split='test', 
        unmatch_ratio=args.unmatch_ratio,
        use_coc=use_coc
    )
    
    # We want to measure match and unmatch separately, so we enforce shuffle=False
    # FocalDataset groups matches first when shuffling is disabled.
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )
    
    num_match = len(test_ds._match_samples)
    print(f"\nTest Dataset built with {dummy_mode} mode.")
    print(f"Total test samples: {len(test_ds)} (Match: {num_match}, Unmatch: {len(test_ds) - num_match})\n")
    
    all_results = {}
    
    for ckpt_path in found_checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print("=" * 60)
        print(f"Evaluating: {ckpt_name}")
        model, version, _ = load_model(ckpt_path, device)
        
        criterion = nn.MSELoss() if version == 'v2' else nn.L1Loss()
        
        test_metrics = {
            'total_loss': 0.0,
            'match_loss': 0.0,
            'unmatch_loss': 0.0,
            
            # Absolute metric errors (L1 distance to actual targets) regardless of whether V1/V2 is used
            'psnr_mae_match': 0.0, 'psnr_mae_total': 0.0,
            'ssim_mae_match': 0.0, 'ssim_mae_total': 0.0,
            'lpips_mae_match': 0.0, 'lpips_mae_total': 0.0,
        }
        
        seen = 0
        pbar = tqdm(test_loader, desc='Testing', leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for x, diopter, targets in pbar:
                bs = x.size(0)
                x = x.to(device)
                diopter = diopter.to(device)
                
                # Targets are already shaped (Batch, 1) in the dataset but let's be safe
                t_psnr  = targets['psnr'].to(device).view(-1, 1)
                t_ssim  = targets['ssim'].to(device).view(-1, 1)
                t_lpips = targets['lpips'].to(device).view(-1, 1)
                
                pred = model(x, diopter)
                
                # Forward predictions
                p_psnr = pred['psnr'].view(-1, 1)
                p_ssim = pred['ssim'].view(-1, 1)
                p_lpips = pred['lpips'].view(-1, 1)
                
                # Compute objective loss logic
                loss = (args.w_psnr  * criterion(p_psnr,  t_psnr / 100.0) +
                        args.w_ssim  * criterion(p_ssim,  t_ssim) +
                        args.w_lpips * criterion(p_lpips, t_lpips))
                
                # Compute absolute MAE differences for readable reporting
                psnr_mae = torch.abs(p_psnr * 100.0 - t_psnr).sum().item()
                ssim_mae = torch.abs(p_ssim - t_ssim).sum().item()
                lpips_mae = torch.abs(p_lpips - t_lpips).sum().item()
                
                test_metrics['total_loss'] += float(loss.item()) * bs
                test_metrics['psnr_mae_total'] += psnr_mae
                test_metrics['ssim_mae_total'] += ssim_mae
                test_metrics['lpips_mae_total'] += lpips_mae
                
                # Split logic for match vs unmatch
                end_idx = seen + bs
                if seen < num_match: # We are still processing match samples
                    match_in_batch = min(bs, num_match - seen)
                    
                    if match_in_batch == bs:
                        ms = slice(0, bs)
                        match_loss = loss
                    else:
                        ms = slice(0, match_in_batch)
                        match_loss = (args.w_psnr  * criterion(p_psnr[ms],  t_psnr[ms] / 100.0) +
                                      args.w_ssim  * criterion(p_ssim[ms],  t_ssim[ms]) +
                                      args.w_lpips * criterion(p_lpips[ms], t_lpips[ms]))
                        
                    test_metrics['match_loss'] += float(match_loss.item()) * match_in_batch
                    test_metrics['psnr_mae_match'] += torch.abs(p_psnr[ms] * 100.0 - t_psnr[ms]).sum().item()
                    test_metrics['ssim_mae_match'] += torch.abs(p_ssim[ms] - t_ssim[ms]).sum().item()
                    test_metrics['lpips_mae_match'] += torch.abs(p_lpips[ms] - t_lpips[ms]).sum().item()
                    
                    unmatch_in_batch = bs - match_in_batch
                    if unmatch_in_batch > 0:
                        us = slice(match_in_batch, bs)
                        unmatch_loss = (args.w_psnr  * criterion(p_psnr[us],  t_psnr[us] / 100.0) +
                                      args.w_ssim  * criterion(p_ssim[us],  t_ssim[us]) +
                                      args.w_lpips * criterion(p_lpips[us], t_lpips[us]))
                        test_metrics['unmatch_loss'] += float(unmatch_loss.item()) * unmatch_in_batch
                else: 
                    test_metrics['unmatch_loss'] += float(loss.item()) * bs
                    
                seen += bs

        # Average out
        n_total = len(test_ds)
        n_match = num_match
        n_unmatch = n_total - num_match
        
        avg_total_loss = test_metrics['total_loss'] / max(n_total, 1)
        avg_match_loss = test_metrics['match_loss'] / max(n_match, 1)
        avg_unmatch_loss = test_metrics['unmatch_loss'] / max(n_unmatch, 1)
        
        print("\n--- RESULTS ---")
        print(f"Model Objective Loss ({'MSE' if version == 'v2' else 'L1'}):")
        print(f"  Overall: {avg_total_loss:.4f} | Match: {avg_match_loss:.4f} | Unmatch: {avg_unmatch_loss:.4f}")
        
        print(f"\nMean Absolute Error (Real World Units):")
        
        # Total
        p_mae_total = test_metrics['psnr_mae_total'] / max(n_total, 1)
        s_mae_total = test_metrics['ssim_mae_total'] / max(n_total, 1)
        l_mae_total = test_metrics['lpips_mae_total'] / max(n_total, 1)
        
        # Match
        p_mae_match = test_metrics['psnr_mae_match'] / max(n_match, 1)
        s_mae_match = test_metrics['ssim_mae_match'] / max(n_match, 1)
        l_mae_match = test_metrics['lpips_mae_match'] / max(n_match, 1)
        
        print(f"  [Total]   PSNR: {p_mae_total:>5.2f} dB | SSIM: {s_mae_total:>6.3f} | LPIPS: {l_mae_total:>6.3f}")
        print(f"  [Match]   PSNR: {p_mae_match:>5.2f} dB | SSIM: {s_mae_match:>6.3f} | LPIPS: {l_mae_match:>6.3f}")
        
        # Unmatch
        if n_unmatch > 0:
            p_mae_unmatch = (test_metrics['psnr_mae_total'] - test_metrics['psnr_mae_match']) / n_unmatch
            s_mae_unmatch = (test_metrics['ssim_mae_total'] - test_metrics['ssim_mae_match']) / n_unmatch
            l_mae_unmatch = (test_metrics['lpips_mae_total'] - test_metrics['lpips_mae_match']) / n_unmatch
            print(f"  [Unmatch] PSNR: {p_mae_unmatch:>5.2f} dB | SSIM: {s_mae_unmatch:>6.3f} | LPIPS: {l_mae_unmatch:>6.3f}")
        print("=" * 60 + "\n")
        
        result_dict = {
            'objective_loss_type': 'MSE' if version == 'v2' else 'L1',
            'objective_loss': {
                'overall': avg_total_loss,
                'match': avg_match_loss,
                'unmatch': avg_unmatch_loss
            },
            'mae_total': {
                'psnr': p_mae_total,
                'ssim': s_mae_total,
                'lpips': l_mae_total
            },
            'mae_match': {
                'psnr': p_mae_match,
                'ssim': s_mae_match,
                'lpips': l_mae_match
            }
        }
        if n_unmatch > 0:
            result_dict['mae_unmatch'] = {
                'psnr': p_mae_unmatch,
                'ssim': s_mae_unmatch,
                'lpips': l_mae_unmatch
            }
        all_results[ckpt_name] = result_dict

    json_path = os.path.join(args.run_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved results to {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Focal Model Quantitatively')
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory (e.g. runs/focal_spatial_xxxx)')
    parser.add_argument('--data_dir', default='../varifocal/data', help='Path to varifocal EXR data')
    parser.add_argument('--unmatch_ratio', type=int, default=39, help='Unmatch ratio for the test set (39 = test all frames)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Needs to match the loss weighting of the run for the objective output to match
    parser.add_argument('--w_psnr',  type=float, default=1.0)
    parser.add_argument('--w_ssim',  type=float, default=0.5)
    parser.add_argument('--w_lpips', type=float, default=0.5)
    
    args = parser.parse_args()
    run_test(args)
