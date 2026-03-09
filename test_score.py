"""
Score-based Generative Model Inference Script

Usage:
    python test_score.py \
        --run_dir runs/score_sinusoidal_XXXX \
        --gm_steps 50 \
        --gm_step_size 0.2 \
        --scene_idx 0 \
        --plane_idx 20
"""

import os
import argparse
import glob
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from model_score import EnergyNet
from dataset_focal import FocalDataset, DP_FOCAL

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    diopter_mode = ckpt.get('diopter_mode', 'sinusoidal')
    use_spectral_norm = ckpt.get('use_spectral_norm', False)
    
    model = EnergyNet(
        use_spectral_norm=use_spectral_norm,
        diopter_mode=diopter_mode
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, diopter_mode

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = os.path.join(args.run_dir, 'latest.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")
        
    print(f"Loading checkpoint: {ckpt_path}")
    model, diopter_mode = load_model(ckpt_path, device)

    data_dir = args.data_dir
    generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    use_coc = (diopter_mode == 'coc')

    # Load 1 sample
    ds = FocalDataset(
        data_dir, generated_data_dir,
        split='val', unmatch_ratio=0,
        use_coc=use_coc, return_gt=True, single_scene_only=True
    )
    
    # Force the specific target diopter
    # Load RGB, Depth
    rgb, depth = ds._load_rgb_depth(args.scene_idx)
    target_diopter = float(DP_FOCAL[args.plane_idx])
    gt = ds._load_gt(args.scene_idx, args.plane_idx)
    
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    depth_t = torch.from_numpy(depth).unsqueeze(0).float().unsqueeze(0).to(device)
    diopter_t = torch.tensor([target_diopter], dtype=torch.float32).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    C = 4
    if use_coc:
        # CoC stub
        depth_m = depth * 12.0 + 1e-6
        dp_dm = 1.0 / depth_m
        dp_fl = 1.0 / 0.017 + target_diopter
        coc_np = 0.004 * np.abs((dp_fl - dp_dm) / (dp_fl - target_diopter + 1e-8) - 1.0)
        coc_np = np.clip(coc_np / 0.028 * depth.shape[1] / 30.0, 0.0, 1.0).astype(np.float32)
        coc_t = torch.from_numpy(coc_np).unsqueeze(0).unsqueeze(0).float().to(device)
        input_rgbd = torch.cat([rgb_t, depth_t], dim=1)
        input_tail = coc_t
        C = 8
    else:
        input_rgbd = torch.cat([rgb_t, depth_t], dim=1)

    print("Starting generation trajectory from random noise...")
    
    with torch.enable_grad():
        current_image = torch.randn_like(gt_t).to(device)
        
        history = [current_image.detach().cpu()]
        
        for step in range(1, args.gm_steps + 1):
            current_image.requires_grad_(True)
            
            if use_coc:
                model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
            else:
                model_input = torch.cat([input_rgbd, current_image], dim=1)
                
            energy = model(model_input, diopter_t)
            
            pred_grad = torch.autograd.grad(
                outputs=energy,
                inputs=current_image,
                grad_outputs=torch.ones_like(energy),
                create_graph=False
            )[0]
            
            with torch.no_grad():
                current_image = (current_image + args.gm_step_size * pred_grad).detach()
                
            if step % 10 == 0 or step == args.gm_steps:
                history.append(current_image.cpu())
                print(f"Step {step}/{args.gm_steps} Energy: {energy.item():.4f}")

    final_image = torch.clamp(current_image, 0.0, 1.0).cpu().squeeze()
    gt_disp = gt_t.cpu().squeeze()

    from dataset_focal import calculate_psnr
    final_psnr = calculate_psnr(final_image, gt_disp).item()
    print(f"Final PSNR against GT: {final_psnr:.2f} dB")

    # Plot
    num_plots = min(len(history), 6)
    indices = np.linspace(0, len(history)-1, num_plots, dtype=int)
    
    fig, axes = plt.subplots(1, num_plots + 1, figsize=(18, 4))
    for i, idx in enumerate(indices):
        im = history[idx].squeeze().permute(1, 2, 0).numpy()
        im = np.clip(im, 0, 1)
        axes[i].imshow(im)
        axes[i].set_title(f"Step {idx}")
        axes[i].axis('off')
        
    axes[-1].imshow(np.clip(gt_disp.permute(1, 2, 0).numpy(), 0, 1))
    axes[-1].set_title("GT")
    axes[-1].axis('off')
    
    out_img = os.path.join(args.run_dir, f'inference_scene{args.scene_idx}_plane{args.plane_idx}.png')
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"Generation history saved to {out_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Path to run directory')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data')
    parser.add_argument('--gm_steps', type=int, default=50)
    parser.add_argument('--gm_step_size', type=float, default=0.2)
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--plane_idx', type=int, default=20)
    
    args = parser.parse_args()
    run_inference(args)
