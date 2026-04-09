"""
Gradient Matching Training Script (SimpleCNN)

Usage:
    python -m gm.train --epochs 50 --single_scene_only
    python -m gm.train --diopter_mode spatial --gm_steps 30
    python -m gm.train --energy_head conv1x1 --eta_schedule cosine --langevin_noise --noise_method constant_scale --noise_scale 0.1
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure parent directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gm.model import SimpleCNN, SimpleCNNDeep, SimpleCNNStride, SimpleResNet, SimpleResNetFiLM, ResUNet, SimpleConvNeXt, ConvNeXtUNet, DilatedNet, InterleaveResNet, save_model_architecture
from gm.config import parse_args, get_parser
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr

try:
    from tensorboardX import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

import math


# ──────────────────────────────────────────────────────────────
# Eta scheduling & Langevin step helpers
# ──────────────────────────────────────────────────────────────
def get_eta(step, total_steps, eta_max, eta_min, schedule='constant'):
    """η 스케줄러: step index(0-based)에 따라 현재 스텝 사이즈를 반환.

    Args:
        step:        현재 스텝 (0-based)
        total_steps: 전체 스텝 수
        eta_max:     최대 η (초반 보폭, = --gm_step_size)
        eta_min:     최소 η (후반 보폭, = --eta_min)
        schedule:    'constant' | 'cosine' | 'linear'

    Returns:
        float: 현재 스텝의 η 값
    """
    if schedule == 'constant':
        return eta_max
    t = step / max(total_steps - 1, 1)  # 0.0 ~ 1.0 정규화
    if schedule == 'cosine':
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t))
    elif schedule == 'linear':
        return eta_max - (eta_max - eta_min) * t
    return eta_max


def langevin_step(current_image, pred_grad, eta, use_noise=False,
                  noise_method='constant_scale', noise_scale=0.1):
    """η 스케줄 + 선택적 Langevin 노이즈를 적용한 업데이트.

    x_{t+1} = x_t + η * ∇E  [+ noise_term]

    Noise methods:
        - constant_scale: noise_term = noise_scale * eta * z
    """
    with torch.no_grad():
        new_image = current_image + eta * pred_grad
        if use_noise:
            noise = torch.randn_like(current_image)
            if noise_method == 'constant_scale':
                new_image = new_image + noise_scale * eta * noise
            else:
                # fallback: original sqrt(2*eta) for unknown methods
                new_image = new_image + math.sqrt(2 * eta) * noise
        return new_image.detach()


def compute_bypass_grad(x, current_image, bypass_lambda, bypass_gamma):
    """Stop-gradient bypass: analytical sharp prior gradient.

    ∂E_sharp/∂y = -λ · w(coc) · (y - rgb)
    w(coc) = exp(-γ · |coc|)

    Computed inside torch.no_grad() → no autograd graph created.
    Returns None if CoC channel is not available (C <= 7).
    """
    N, C, H, W = x.shape
    if C <= 7:
        return None
    with torch.no_grad():
        rgb_in = x[:, :3, :, :]
        coc_map = x[:, 7:8, :, :]
        coc_abs = torch.abs(coc_map)
        w = torch.exp(-bypass_gamma * coc_abs)
        return -bypass_lambda * w * (current_image.detach() - rgb_in)


# ──────────────────────────────────────────────────────────────
# Training (one epoch)
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, epoch,
                gm_steps, gm_step_size, eta_min=0.002,
                eta_schedule='constant', langevin_noise=False,
                noise_method='constant_scale', noise_scale=0.1,
                bypass_lambda=0.0, bypass_gamma=30.0, bypass_alpha=0.0,
                scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False, dynamic_ncols=True)
    for batch_data in pbar:
        x, diopter, targets, gt = batch_data
        N, C, H, W = x.shape
        x = x.to(device)
        diopter = diopter.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        # 1. 랜덤 노이즈에서 시작
        current_image = torch.randn_like(gt).to(device)
        batch_loss = 0.0

        # 2. Trajectory steps
        for step in range(gm_steps):
            eta = get_eta(step, gm_steps, gm_step_size, eta_min, eta_schedule)
            current_image.requires_grad_(True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 모델 입력 조립: RGBD(4ch) + 생성중인 이미지(3ch) + [CoC(1ch)]
                input_rgbd = x[:, :4, :, :]
                if C > 7:
                    input_tail = x[:, 7:, :, :]
                    model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                else:
                    model_input = torch.cat([input_rgbd, current_image], dim=1)

                energy = model(model_input, diopter)

                pred_grad = torch.autograd.grad(
                    outputs=energy,
                    inputs=current_image,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True
                )[0]

                # Stop-gradient bypass: analytical sharp prior gradient
                if bypass_alpha > 0:
                    bg = compute_bypass_grad(x, current_image, bypass_lambda, bypass_gamma)
                    if bg is not None:
                        pred_grad = pred_grad + bypass_alpha * bg

                gt_grad = gt - current_image
                loss = F.mse_loss(pred_grad, gt_grad)
                
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            batch_loss += loss.item()

            current_image = langevin_step(current_image, pred_grad, eta, langevin_noise,
                                          noise_method, noise_scale)

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        avg_step_loss = batch_loss / gm_steps
        total_loss += avg_step_loss * N
        n += N
        pbar.set_postfix(loss=f'{avg_step_loss:.4f}')

    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, gm_steps, gm_step_size,
            eta_min=0.002, eta_schedule='constant', langevin_noise=False,
            noise_method='constant_scale', noise_scale=0.1,
            bypass_lambda=0.0, bypass_gamma=30.0, bypass_alpha=0.0, use_amp=False):
    model.eval()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc='           [val] ', leave=False, dynamic_ncols=True)
    for batch_data in pbar:
        x, diopter, targets, gt = batch_data
        N, C, H, W = x.shape
        x = x.to(device)
        diopter = diopter.to(device)
        gt = gt.to(device)

        with torch.enable_grad():
            current_image = torch.randn_like(gt).to(device)
            batch_loss = 0.0

            for step in range(gm_steps):
                eta = get_eta(step, gm_steps, gm_step_size, eta_min, eta_schedule)
                current_image.requires_grad_(True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    input_rgbd = x[:, :4, :, :]
                    if C > 7:
                        input_tail = x[:, 7:, :, :]
                        model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                    else:
                        model_input = torch.cat([input_rgbd, current_image], dim=1)

                    energy = model(model_input, diopter)

                    pred_grad = torch.autograd.grad(
                        outputs=energy,
                        inputs=current_image,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=False
                    )[0]

                    # Stop-gradient bypass
                    if bypass_alpha > 0:
                        bg = compute_bypass_grad(x, current_image, bypass_lambda, bypass_gamma)
                        if bg is not None:
                            pred_grad = pred_grad + bypass_alpha * bg

                    gt_grad = gt - current_image
                    loss = F.mse_loss(pred_grad, gt_grad)
                
                batch_loss += loss.item()

                current_image = langevin_step(current_image, pred_grad, eta, langevin_noise,
                                              noise_method, noise_scale)

        avg_step_loss = batch_loss / gm_steps
        total_loss += avg_step_loss * N
        n += N
        pbar.set_postfix(loss=f'{avg_step_loss:.4f}')

    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────
# PSNR Evaluation (실제 생성 품질 측정)
# ──────────────────────────────────────────────────────────────
def compute_val_psnr(model, dataset, device, gm_steps, gm_step_size,
                    eta_min=0.002, eta_schedule='constant', langevin_noise=False,
                    noise_method='constant_scale', noise_scale=0.1,
                    eval_plane=20,
                    bypass_lambda=0.0, bypass_gamma=30.0, bypass_alpha=0.0, use_amp=False):
    """Val set의 모든 scene에 대해 특정 plane을 생성하고 평균 PSNR을 반환.
    
    val_loss와 달리 실제 이미지 생성 품질을 측정하므로,
    best model 판정에 더 신뢰할 수 있는 지표입니다.
    """
    model.eval()
    psnr_list = []

    # eval_plane에 해당하는 모든 match sample 찾기
    eval_indices = []
    for idx, (s, pp, qp) in enumerate(dataset._match_samples):
        if qp == eval_plane:
            eval_indices.append(idx)

    if not eval_indices:
        print(f"  WARNING: plane {eval_plane} not found in dataset")
        return 0.0

    for sample_idx in eval_indices:
        x, diopter, targets, gt = dataset[sample_idx]
        x = x.unsqueeze(0).to(device)
        diopter = diopter.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)
        gt_cpu = gt.cpu().squeeze()

        with torch.enable_grad():
            current_image = torch.randn_like(gt).to(device)
            N, C, H, W = x.shape

            for step in range(gm_steps):
                eta = get_eta(step, gm_steps, gm_step_size, eta_min, eta_schedule)
                current_image.requires_grad_(True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    input_rgbd = x[:, :4, :, :]
                    if C > 7:
                        input_tail = x[:, 7:, :, :]
                        model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                    else:
                        model_input = torch.cat([input_rgbd, current_image], dim=1)

                    energy = model(model_input, diopter)
                    pred_grad = torch.autograd.grad(
                        energy, current_image, torch.ones_like(energy),
                        create_graph=False
                    )[0]

                    # Stop-gradient bypass
                    if bypass_alpha > 0:
                        bg = compute_bypass_grad(x, current_image, bypass_lambda, bypass_gamma)
                        if bg is not None:
                            pred_grad = pred_grad + bypass_alpha * bg

                use_noise = langevin_noise and (step < gm_steps - 1)
                current_image = langevin_step(current_image, pred_grad, eta, use_noise,
                                              noise_method, noise_scale)

        final_img = torch.clamp(current_image, 0, 1).cpu().squeeze()
        psnr_val = calculate_psnr(final_img, gt_cpu).item()
        psnr_list.append(psnr_val)

    avg_psnr = np.mean(psnr_list)
    return avg_psnr

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Auto-resume from run directory ──
    if args.resume_dir:
        resume_args_path = os.path.join(args.resume_dir, 'args.json')
        resume_ckpt_path = os.path.join(args.resume_dir, 'latest.pth')

        if not os.path.exists(resume_args_path):
            print(f"ERROR: args.json not found in {args.resume_dir}")
            return
        if not os.path.exists(resume_ckpt_path):
            print(f"ERROR: latest.pth not found in {args.resume_dir}")
            return

        # 저장된 설정 로드
        with open(resume_args_path) as f:
            saved_args = json.load(f)

        # CLI 기본값과 다른 인수만 override로 간주
        parser = get_parser()
        defaults = vars(parser.parse_args([]))

        # saved_args에서 복원할 키 목록 (학습 설정 전체)
        restore_keys = [
            'arch', 'diopter_mode', 'energy_head', 'channels',
            'use_film', 'long_skip', 'sharp_prior', 'sharp_lambda_learnable', 'sharp_gamma_learnable', 'sharp_lambda', 'sharp_gamma',
            'activation', 'interleave_rate',
            'epochs', 'batch_size', 'lr', 'weight_decay',
            'gm_steps', 'gm_step_size', 'eta_schedule', 'eta_min', 'langevin_noise', 'noise_method', 'noise_scale',
            'train_bypass', 'bypass_lambda', 'bypass_gamma', 'bypass_warmup', 'bypass_ramp',
            'single_scene_only', 'num_scenes', 'unmatch_ratio', 'save_every',
            'data_dir', 'generated_data_dir', 'output_dir'
        ]

        for key in restore_keys:
            if key in saved_args:
                cli_val = getattr(args, key, defaults.get(key))
                default_val = defaults.get(key)
                # CLI에서 명시적으로 지정하지 않은 경우에만 복원
                if cli_val == default_val:
                    setattr(args, key, saved_args[key])

        # resume 체크포인트 자동 설정
        args.resume = resume_ckpt_path
        output_dir = args.resume_dir

        print(f"\n{'='*50}")
        print(f"  AUTO-RESUME from: {args.resume_dir}")
        print(f"  Checkpoint: latest.pth")
        print(f"  Config restored from args.json")
        print(f"{'='*50}\n")
    else:
        # ── 새 run: Output directory 생성 ──
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scene_str = "scene0_" if args.single_scene_only else ""
        head_str = f"_{args.energy_head}" if args.energy_head != 'fc' else ""
        eta_str = f"_{args.eta_schedule}" if args.eta_schedule != 'constant' else ""
        run_name = f"gm_{scene_str}{args.diopter_mode}{head_str}{eta_str}_{timestamp}"
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)

    # ── Save args.json (항상 최신 설정 저장) ──
    args_dict = vars(args)
    args_dict['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(f"Args saved to {output_dir}/args.json")

    # ── Dataset ──
    generated_data_dir = args.generated_data_dir
    if generated_data_dir is None:
        generated_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    diopter_mode = args.diopter_mode

    train_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='train', unmatch_ratio=args.unmatch_ratio,
        diopter_mode=diopter_mode, return_gt=True,
        single_scene_only=args.single_scene_only,
        num_scenes=args.num_scenes
    )
    val_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='val', unmatch_ratio=0,
        diopter_mode=diopter_mode, return_gt=True,
        single_scene_only=args.single_scene_only,
        num_scenes=args.num_scenes
    )

    if args.single_scene_only:
        print("[Prototype Mode] Overfitting to a single SCENE (all focal planes)!")
        val_ds = train_ds

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ── Model init ──
    if args.arch == 'deep':
        model = SimpleCNNDeep(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head
        ).to(device)
    elif args.arch == 'stride':
        model = SimpleCNNStride(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head
        ).to(device)
    elif args.arch == 'resnet':
        model = SimpleResNet(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            num_blocks=4,
            channels=args.channels,
            use_film=args.use_film,
            long_skip=args.long_skip,
            use_sharp_prior=args.sharp_prior,
            activation=args.activation,
            sharp_lambda_init=args.sharp_lambda,
            sharp_gamma_init=args.sharp_gamma
        ).to(device)
    elif args.arch == 'resnet_film':
        model = SimpleResNetFiLM(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            num_blocks=4,
            channels=args.channels,
            long_skip=args.long_skip,
            use_sharp_prior=args.sharp_prior,
            sharp_lambda_learnable=args.sharp_lambda_learnable,
            sharp_gamma_learnable=args.sharp_gamma_learnable,
            activation=args.activation,
            sharp_lambda_init=args.sharp_lambda,
            sharp_gamma_init=args.sharp_gamma
        ).to(device)
    elif args.arch == 'resunet':
        model = ResUNet(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            base_channels=args.channels,
            num_bottleneck_blocks=3,
            use_film=args.use_film
        ).to(device)
    elif args.arch == 'convnext':
        model = SimpleConvNeXt(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            num_blocks=4,
            channels=args.channels,
            use_film=args.use_film
        ).to(device)
    elif args.arch == 'convnext_unet':
        model = ConvNeXtUNet(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            num_blocks=9,
            channels=args.channels,
            use_film=args.use_film
        ).to(device)
    elif args.arch == 'dilated':
        model = DilatedNet(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head
        ).to(device)
    elif args.arch == 'interleave_resnet':
        model = InterleaveResNet(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head,
            num_blocks=4,
            channels=args.channels,
            use_film=args.use_film,
            interleave_rate=args.interleave_rate
        ).to(device)
    else:
        model = SimpleCNN(
            input_channels=7,
            diopter_mode=args.diopter_mode,
            energy_head=args.energy_head
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}  diopter_mode={args.diopter_mode}  energy_head={args.energy_head}")
    print(f"Eta schedule: {args.eta_schedule}  eta_max={args.gm_step_size}  eta_min={args.eta_min}")
    print(f"Langevin noise: {args.langevin_noise}  method={args.noise_method}  scale={args.noise_scale}")
    if args.train_bypass:
        print(f"Train bypass: lambda={args.bypass_lambda}  gamma={args.bypass_gamma}  warmup={args.bypass_warmup}  ramp={args.bypass_ramp}")

    # 모델 구조 .txt 저장
    save_model_architecture(model, os.path.join(output_dir, 'model_architecture.txt'), args)
    print(f"Model architecture saved to {output_dir}/model_architecture.txt")

    # ── Resume ──
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Compile the model after loading weights ──
    if args.compile:
        print(f"Applying torch.compile(mode={args.compile_mode})...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}. Proceeding without compilation.")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    writer = None
    if HAS_TB:
        writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    best_val_loss = float('inf')
    best_val_psnr = 0.0  # PSNR 기반 best 모델 추적

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Bypass alpha: warmup → ramp → full
        if args.train_bypass:
            if args.bypass_ramp > 0:
                bypass_alpha = min(1.0, max(0.0, (epoch - args.bypass_warmup) / args.bypass_ramp))
            else:
                bypass_alpha = 1.0 if epoch > args.bypass_warmup else 0.0
            if epoch <= args.bypass_warmup + args.bypass_ramp:
                print(f"  Bypass alpha: {bypass_alpha:.3f}")
        else:
            bypass_alpha = 0.0

        if args.unmatch_ratio > 0:
            train_ds.resample_unmatch()
            print(f"  Resampled: {len(train_ds)} total samples")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers
        )

        train_loss = train_epoch(model, train_loader, optimizer,
                                 device, epoch, args.gm_steps, args.gm_step_size,
                                 args.eta_min, args.eta_schedule, args.langevin_noise,
                                 args.noise_method, args.noise_scale,
                                 args.bypass_lambda, args.bypass_gamma, bypass_alpha,
                                 scaler=scaler, use_amp=args.amp)
        val_loss = validate(model, val_loader, device, args.gm_steps, args.gm_step_size,
                            args.eta_min, args.eta_schedule, args.langevin_noise,
                            args.noise_method, args.noise_scale,
                            args.bypass_lambda, args.bypass_gamma, bypass_alpha, use_amp=args.amp)

        print(f"Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")

        # ── Val PSNR 측정 (실제 생성 품질) ──
        val_psnr = compute_val_psnr(
            model, val_ds, device, args.gm_steps, args.gm_step_size,
            args.eta_min, args.eta_schedule, args.langevin_noise,
            args.noise_method, args.noise_scale,
            eval_plane=20,
            bypass_lambda=args.bypass_lambda, bypass_gamma=args.bypass_gamma,
            bypass_alpha=bypass_alpha, use_amp=args.amp
        )
        print(f"Val PSNR (plane 20 avg): {val_psnr:.2f} dB")

        if writer:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/psnr', val_psnr, epoch)

        scheduler.step(val_loss)

        # ── Checkpoint ──
        state_dict = model.state_dict()
        # Remove _orig_mod. prefix if compiled
        if args.compile:
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        ckpt = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'arch': args.arch,
            'diopter_mode': args.diopter_mode,
            'energy_head': args.energy_head,
            'eta_schedule': args.eta_schedule,
            'eta_min': args.eta_min,
            'langevin_noise': args.langevin_noise,
            'noise_method': args.noise_method,
            'noise_scale': args.noise_scale,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'channels': args.channels,
            'use_film': args.use_film,
            'long_skip': args.long_skip,
            'interleave_rate': args.interleave_rate,
            'sharp_prior': args.sharp_prior,
            'sharp_lambda_learnable': args.sharp_lambda_learnable,
            'sharp_gamma_learnable': args.sharp_gamma_learnable,
            'activation': args.activation,
            'sharp_lambda': args.sharp_lambda,
            'sharp_gamma': args.sharp_gamma,
            'train_bypass': args.train_bypass,
            'bypass_lambda': args.bypass_lambda,
            'bypass_gamma': args.bypass_gamma,
            'bypass_warmup': args.bypass_warmup,
            'bypass_ramp': args.bypass_ramp
        }

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(output_dir, 'best_model.pth'))
            print(f"  ** New best (loss) saved (val_loss={val_loss:.6f}) **")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(ckpt, os.path.join(output_dir, 'best_psnr_model.pth'))
            print(f"  ** New best (PSNR) saved (val_psnr={val_psnr:.2f} dB) **")

        torch.save(ckpt, os.path.join(output_dir, 'latest.pth'))

    if writer:
        writer.close()
    print(f"\nTraining complete. Results in: {output_dir}")

    # ── Auto inference: best + latest 두 모델 모두 자동 생성 ──
    # 추론 시에는 최소 50 스텝 사용 (학습 시 gm_steps가 작아도)
    # prototype(single_scene_only)이면 학습 데이터로 검증, 아니면 val 데이터
    infer_ds = train_ds if args.single_scene_only else val_ds
    infer_steps = max(args.gm_steps, 50)
    print(f"Auto inference using {'train' if args.single_scene_only else 'val'} dataset ({len(infer_ds)} samples)")

    # Bypass 학습 시 auto-inference에도 bypass 적용 (alpha=1.0)
    infer_bypass_alpha = 1.0 if args.train_bypass else 0.0

    for tag in ['best', 'best_psnr', 'latest']:
        ckpt_path = os.path.join(output_dir, f'{tag}_model.pth' if tag != 'latest' else f'{tag}.pth')
        if os.path.exists(ckpt_path):
            final_generation_check(model, infer_ds, device, output_dir, args,
                                   ckpt_path, tag, infer_steps,
                                   bypass_lambda=args.bypass_lambda,
                                   bypass_gamma=args.bypass_gamma,
                                   bypass_alpha=infer_bypass_alpha, use_amp=args.amp)

# ──────────────────────────────────────────────────────────────
# Final Generation Check (자동 추론)
# ──────────────────────────────────────────────────────────────
def final_generation_check(model, dataset, device, output_dir, args, ckpt_path, tag, infer_steps,
                           bypass_lambda=0.0, bypass_gamma=30.0, bypass_alpha=0.0, use_amp=False):
    """체크포인트를 로드하여 대표 focal plane 3장 생성 + 저장"""
    print(f"\n[Final Check - {tag}] Loading {ckpt_path}...")
    print(f"  Using {infer_steps} inference steps (train was {args.gm_steps})")
    print(f"  Eta schedule: {args.eta_schedule}  noise: {args.langevin_noise}  method: {args.noise_method}  scale: {args.noise_scale}")
    if bypass_alpha > 0:
        print(f"  Bypass: lambda={bypass_lambda}  gamma={bypass_gamma}  alpha={bypass_alpha:.3f}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    ckpt_epoch = ckpt.get('epoch', '?')
    print(f"  Checkpoint epoch: {ckpt_epoch}")

    target_planes = [0, 20, 39]  # 0.1D, 2.0D, 4.0D

    fig, axes = plt.subplots(len(target_planes), 2, figsize=(10, 5 * len(target_planes)))
    psnr_results = []

    for i, q_idx in enumerate(target_planes):
        d_val = float(DP_FOCAL[q_idx])
        print(f"  Generating plane {q_idx} ({d_val:.1f}D)...")

        # 해당 plane의 match sample 찾기
        sample_idx = 0
        for idx, (s, p, q) in enumerate(dataset._match_samples):
            if q == q_idx:
                sample_idx = idx
                break

        x, diopter, targets, gt = dataset[sample_idx]
        x = x.unsqueeze(0).to(device)
        diopter = diopter.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        with torch.enable_grad():
            current_image = torch.randn_like(gt).to(device)
            N, C, H, W = x.shape

            for step in range(infer_steps):
                eta = get_eta(step, infer_steps, args.gm_step_size,
                              args.eta_min, args.eta_schedule)
                current_image.requires_grad_(True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    input_rgbd = x[:, :4, :, :]
                    if C > 7:
                        input_tail = x[:, 7:, :, :]
                        model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
                    else:
                        model_input = torch.cat([input_rgbd, current_image], dim=1)

                    energy = model(model_input, diopter)
                    pred_grad = torch.autograd.grad(
                        energy, current_image, torch.ones_like(energy)
                    )[0]

                    # Stop-gradient bypass
                    if bypass_alpha > 0:
                        bg = compute_bypass_grad(x, current_image, bypass_lambda, bypass_gamma)
                        if bg is not None:
                            pred_grad = pred_grad + bypass_alpha * bg

                # 추론 시 마지막 스텝에서는 노이즈를 끌어서 깨끗한 결과 보장
                use_noise = args.langevin_noise and (step < infer_steps - 1)
                current_image = langevin_step(current_image, pred_grad, eta, use_noise,
                                              args.noise_method, args.noise_scale)

        final_img = torch.clamp(current_image, 0, 1).cpu().squeeze().permute(1, 2, 0).numpy()
        gt_img = gt.cpu().squeeze().permute(1, 2, 0).numpy()
        psnr_val = calculate_psnr(current_image.cpu().squeeze(), gt.cpu().squeeze()).item()
        psnr_results.append({'plane': q_idx, 'diopter': d_val, 'psnr': psnr_val})

        axes[i, 0].imshow(np.clip(gt_img, 0, 1))
        axes[i, 0].set_title(f"GT {d_val:.1f}D")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(np.clip(final_img, 0, 1))
        axes[i, 1].set_title(f"Gen {d_val:.1f}D (PSNR: {psnr_val:.2f}dB)")
        axes[i, 1].axis('off')

    plt.suptitle(f"[{tag}] epoch {ckpt_epoch}", fontsize=14)
    plt.tight_layout()
    out_img = os.path.join(output_dir, f'generation_check_{tag}.png')
    plt.savefig(out_img, dpi=100)
    plt.close()
    print(f"  Saved: {out_img}")

    # PSNR 결과 JSON
    json_path = os.path.join(output_dir, f'psnr_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump({'tag': tag, 'epoch': ckpt_epoch, 'results': psnr_results}, f, indent=2)
    for r in psnr_results:
        print(f"  Plane {r['plane']} ({r['diopter']:.1f}D): PSNR = {r['psnr']:.2f} dB")


if __name__ == '__main__':
    main()

