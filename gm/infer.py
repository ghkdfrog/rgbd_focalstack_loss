"""
Gradient Matching Inference Script (SimpleCNN)

Usage:
    # 기본: best, best_psnr, latest 모델 전부 추론 (planes 0,20,39)
    python -m gm.infer --run_dir runs_gm/<run_name>

    # 특정 체크포인트만
    python -m gm.infer --run_dir runs_gm/<run_name> --ckpt_tag best_psnr

    # 전체 40장
    python -m gm.infer --run_dir runs_gm/<run_name> --plane_idx -1

    # 커스텀 plane
    python -m gm.infer --run_dir runs_gm/<run_name> --plane_idx 0,10,20,30,39
"""

import os
import sys
import json
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gm.model import (SimpleCNN, SimpleCNNDeep, SimpleCNNStride, SimpleResNet, SimpleResNetFiLM, DWTResNetFiLM, ResUNet,
                      SimpleConvNeXt, ConvNeXtUNet, DilatedNet, InterleaveResNet)
from gm.config import parse_args
from gm.train import get_eta, langevin_step
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr


def load_model_from_ckpt(ckpt_path, diopter_mode, energy_head, device, arch='simple', channels=256, use_film=False, long_skip=False, interleave_rate=2):
    """체크포인트 파일에서 모델 로드"""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 체크포인트에 저장된 설정 우선
    diopter_mode = ckpt.get('diopter_mode', diopter_mode)
    energy_head = ckpt.get('energy_head', energy_head)
    channels = ckpt.get('channels', channels)
    use_film = ckpt.get('use_film', use_film)
    long_skip = ckpt.get('long_skip', long_skip)
    interleave_rate = ckpt.get('interleave_rate', interleave_rate)
    sharp_prior = ckpt.get('sharp_prior', False)
    activation = ckpt.get('activation', 'relu')
    sharp_lambda_init = ckpt.get('sharp_lambda', 10.0)
    sharp_gamma_init = ckpt.get('sharp_gamma', 30.0)
    # 하위 호환: 기존 sharp_learnable → 둘 다 learnable로 해석
    old_learnable = ckpt.get('sharp_learnable', False)
    sharp_lambda_learnable = ckpt.get('sharp_lambda_learnable', old_learnable)
    sharp_gamma_learnable = ckpt.get('sharp_gamma_learnable', old_learnable)

    # 하위 호환: 기존 film_resnet → resnet + use_film=True
    if arch == 'film_resnet':
        arch = 'resnet'
        use_film = True

    compositional_ebm = ckpt.get('compositional_ebm', False)

    if arch == 'deep':
        model = SimpleCNNDeep(diopter_mode=diopter_mode, energy_head=energy_head).to(device)
    elif arch == 'stride':
        model = SimpleCNNStride(diopter_mode=diopter_mode, energy_head=energy_head).to(device)
    elif arch == 'resnet':
        model = SimpleResNet(diopter_mode=diopter_mode, energy_head=energy_head, num_blocks=4, channels=channels, use_film=use_film, long_skip=long_skip, use_sharp_prior=sharp_prior, activation=activation, sharp_lambda_init=sharp_lambda_init, sharp_gamma_init=sharp_gamma_init).to(device)
    elif arch == 'resnet_film':
        model = SimpleResNetFiLM(diopter_mode=diopter_mode, energy_head=energy_head, num_blocks=4, channels=channels, long_skip=long_skip, use_sharp_prior=sharp_prior, sharp_lambda_learnable=sharp_lambda_learnable, sharp_gamma_learnable=sharp_gamma_learnable, activation=activation, sharp_lambda_init=sharp_lambda_init, sharp_gamma_init=sharp_gamma_init, compositional_ebm=compositional_ebm).to(device)
    elif arch == 'dwt_resnet_film':
        model = DWTResNetFiLM(diopter_mode=diopter_mode, energy_head=energy_head, num_blocks=4, channels=channels, long_skip=long_skip, activation=activation).to(device)
    elif arch == 'resunet':
        model = ResUNet(diopter_mode=diopter_mode, energy_head=energy_head, base_channels=channels, num_bottleneck_blocks=3, use_film=use_film).to(device)
    elif arch == 'convnext':
        model = SimpleConvNeXt(diopter_mode=diopter_mode, energy_head=energy_head, num_blocks=4, channels=channels, use_film=use_film).to(device)
    elif arch == 'convnext_unet':
        model = ConvNeXtUNet(diopter_mode=diopter_mode, energy_head=energy_head, num_blocks=9, channels=channels, use_film=use_film).to(device)
    elif arch == 'dilated':
        model = DilatedNet(diopter_mode=diopter_mode, energy_head=energy_head).to(device)
    elif arch == 'interleave_resnet':
        model = InterleaveResNet(diopter_mode=diopter_mode, energy_head=energy_head,
                                num_blocks=4, channels=channels, use_film=use_film,
                                interleave_rate=interleave_rate).to(device)
    else:
        model = SimpleCNN(diopter_mode=diopter_mode, energy_head=energy_head).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    return model, epoch


def resolve_ckpt_paths(run_dir, ckpt_tag):
    """ckpt_tag에 따라 추론할 체크포인트 경로 목록 반환"""
    tag_to_filename = {
        'best':      'best_model.pth',
        'best_psnr': 'best_psnr_model.pth',
        'latest':    'latest.pth',
    }

    if ckpt_tag == 'all':
        tags_to_try = ['best', 'best_psnr', 'latest']
    else:
        tags_to_try = [ckpt_tag]

    result = []
    for tag in tags_to_try:
        path = os.path.join(run_dir, tag_to_filename[tag])
        if os.path.exists(path):
            result.append((tag, path))
        else:
            print(f"  [SKIP] {tag_to_filename[tag]} not found")

    return result


def generate_one_plane(model, x, diopter, gt, device, gm_steps, gm_step_size,
                       eta_min=0.002, eta_schedule='constant', use_langevin_noise=False,
                       noise_method='constant_scale', noise_scale=0.1,
                       infer_sharp=False, infer_sharp_lambda=5.0,
                       infer_sharp_gamma=30.0, infer_sharp_start=0.5, use_amp=False,
                       enable_struct=True, enable_percep=True, enable_phys=True):
    """한 장의 focal plane을 생성하고 PSNR, 히스토리, step별 PSNR/energy를 반환

    infer_sharp: True이면 inference-time sharpening 적용
        모델 gradient에 해석적 sharp prior gradient를 추가:
        ∇E_sharp = -λ · w(coc) · (y - rgb)
        w(coc) = exp(-γ · |coc|)
        초점 영역(CoC≈0)에서 y → rgb 방향으로 당기는 효과
    infer_sharp_start: 전체 step 대비 sharpening 시작 비율 (0.0~1.0)
        초반에는 모델 gradient만으로 이미지 구조를 잡고,
        후반부터 선명화를 적용하여 안정적인 수렴 보장
    """
    N, C, H, W = x.shape
    step_psnr_history = []
    gt_cpu = gt.cpu().squeeze()  # step별 PSNR 계산용 (미리 CPU로)

    # Sharp prior용 원본 RGB & CoC 맵 추출
    rgb_in = x[:, :3, :, :]  # 원본 RGB (3ch)
    coc_map = x[:, 7:8, :, :] if C > 7 else None  # CoC (1ch)
    sharp_start_step = int(gm_steps * infer_sharp_start)

    with torch.enable_grad():
        current_image = torch.randn_like(gt).to(device)
        history = [current_image.detach().cpu()]

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

                if getattr(model, 'compositional_ebm', False):
                    eng_struct, eng_percep, eng_phys = model(model_input, diopter)
                    energy_val = (eng_struct.item() + eng_percep.item() + eng_phys.item())

                    pred_gs = torch.autograd.grad(eng_struct, current_image, torch.ones_like(eng_struct), create_graph=False, retain_graph=True)[0]
                    pred_gp = torch.autograd.grad(eng_percep, current_image, torch.ones_like(eng_percep), create_graph=False, retain_graph=True)[0]
                    pred_gph = torch.autograd.grad(eng_phys, current_image, torch.ones_like(eng_phys), create_graph=False)[0]

                    grad = torch.zeros_like(current_image)
                    if enable_struct: grad = grad + pred_gs
                    if enable_percep: grad = grad + pred_gp
                    if enable_phys: grad = grad + pred_gph
                else:
                    energy = model(model_input, diopter)
                    energy_val = energy.item()

                    grad = torch.autograd.grad(
                        outputs=energy,
                        inputs=current_image,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=False
                    )[0]

            # ── Inference-time Sharpening ──
            # 해석적 gradient: ∇E_sharp/∂y = -λ · w(coc) · (y - rgb)
            # autograd 없이 직접 계산 → NaN 위험 없음
            if infer_sharp and coc_map is not None and step >= sharp_start_step:
                with torch.no_grad():
                    coc_abs = torch.abs(coc_map)  # (N, 1, H, W)
                    w = torch.exp(-infer_sharp_gamma * coc_abs)  # CoC≈0 → 1, CoC↑ → 0
                    # ∂E_sharp/∂y = -λ · w · (y - rgb)  →  y를 rgb 방향으로 당김
                    sharp_grad = -infer_sharp_lambda * w * (current_image.detach() - rgb_in)
                    grad = grad + sharp_grad

            # 마지막 스텝에서는 노이즈 없이 깨끗하게 마무리
            noise = use_langevin_noise and (step < gm_steps - 1)
            current_image = langevin_step(current_image, grad, eta, noise, noise_method, noise_scale)

            # step별 PSNR + energy 기록
            with torch.no_grad():
                step_img = torch.clamp(current_image, 0.0, 1.0).cpu().squeeze()
                step_psnr = calculate_psnr(step_img, gt_cpu).item()
                step_psnr_history.append({
                    'step': step + 1,
                    'eta': round(eta, 6),
                    'psnr': round(step_psnr, 4),
                    'energy': round(energy_val, 6)
                })

            if (step + 1) % 10 == 0 or step == gm_steps - 1:
                history.append(current_image.cpu())

    final_image = torch.clamp(current_image, 0.0, 1.0).cpu().squeeze()
    psnr = calculate_psnr(final_image, gt_cpu).item()

    return final_image, psnr, history, step_psnr_history


def visualize(history, gt, psnr, save_path, plane_idx, diopter_val):
    """생성 과정 히스토리 + GT 비교 시각화"""
    num_plots = min(len(history), 6)
    indices = np.linspace(0, len(history) - 1, num_plots, dtype=int)

    fig, axes = plt.subplots(1, num_plots + 1, figsize=(18, 4))
    for i, idx in enumerate(indices):
        im = history[idx].squeeze().permute(1, 2, 0).numpy()
        axes[i].imshow(np.clip(im, 0, 1))
        step_label = 0 if idx == 0 else idx * 10
        axes[i].set_title(f"Step {step_label}")
        axes[i].axis('off')

    axes[-1].imshow(np.clip(gt.squeeze().permute(1, 2, 0).numpy(), 0, 1))
    axes[-1].set_title("GT")
    axes[-1].axis('off')

    plt.suptitle(f"Plane {plane_idx} ({diopter_val:.2f}D)  PSNR: {psnr:.2f} dB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved: {save_path}  (PSNR: {psnr:.2f} dB)")


def save_step_psnr_csv(all_step_data, save_dir):
    """모든 plane의 step별 PSNR + energy를 하나의 CSV 표로 저장"""
    csv_path = os.path.join(save_dir, 'step_psnr_table.csv')
    if not all_step_data:
        return csv_path

    plane_keys = sorted(all_step_data.keys())
    num_steps = len(all_step_data[plane_keys[0]])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step', 'eta']
        for pk in plane_keys:
            dv = float(DP_FOCAL[pk])
            header.append(f'plane{pk:02d}_{dv:.1f}D_psnr')
            header.append(f'plane{pk:02d}_{dv:.1f}D_energy')
        writer.writerow(header)

        for i in range(num_steps):
            row = [
                all_step_data[plane_keys[0]][i]['step'],
                all_step_data[plane_keys[0]][i]['eta']
            ]
            for pk in plane_keys:
                row.append(all_step_data[pk][i]['psnr'])
                row.append(all_step_data[pk][i].get('energy', ''))
            writer.writerow(row)

    print(f"  Step PSNR table saved: {csv_path}")
    return csv_path


def plot_psnr_convergence(all_step_data, save_dir):
    """step별 PSNR 수렴 그래프 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for p_idx in sorted(all_step_data.keys()):
        step_data = all_step_data[p_idx]
        steps = [d['step'] for d in step_data]
        psnrs = [d['psnr'] for d in step_data]
        dv = float(DP_FOCAL[p_idx])
        ax.plot(steps, psnrs, label=f'Plane {p_idx} ({dv:.1f}D)', linewidth=1.5)

    ax.set_xlabel('GM Step', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR Convergence per GM Step', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'psnr_convergence.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  PSNR convergence plot saved: {plot_path}")


def plot_energy_per_plane(all_step_data, save_dir):
    """plane별 energy 그래프 생성: 개별 plane 그래프 + 전체 오버레이 그래프"""
    if not all_step_data:
        return

    energy_dir = os.path.join(save_dir, 'energy_plots')
    os.makedirs(energy_dir, exist_ok=True)

    # ── 1) 개별 plane 그래프 ──
    for p_idx in sorted(all_step_data.keys()):
        step_data = all_step_data[p_idx]
        steps = [d['step'] for d in step_data]
        energies = [d.get('energy', 0) for d in step_data]
        dv = float(DP_FOCAL[p_idx])

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(steps, energies, color='#e74c3c', linewidth=1.8, marker='o', markersize=2.5)
        ax.set_xlabel('GM Step', fontsize=12)
        ax.set_ylabel('Model Output (Energy)', fontsize=12)
        ax.set_title(f'Energy per Step — Plane {p_idx} ({dv:.1f}D)', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(energy_dir, f'energy_plane{p_idx:02d}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

    # ── 2) 전체 오버레이 그래프 ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for p_idx in sorted(all_step_data.keys()):
        step_data = all_step_data[p_idx]
        steps = [d['step'] for d in step_data]
        energies = [d.get('energy', 0) for d in step_data]
        dv = float(DP_FOCAL[p_idx])
        ax.plot(steps, energies, label=f'Plane {p_idx} ({dv:.1f}D)', linewidth=1.5)

    ax.set_xlabel('GM Step', fontsize=12)
    ax.set_ylabel('Model Output (Energy)', fontsize=12)
    ax.set_title('Energy per Step (All Planes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    overlay_path = os.path.join(save_dir, 'energy_convergence.png')
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"  Energy plots saved: {energy_dir}/  +  {overlay_path}")


def run_inference_for_tag(tag, ckpt_path, args, saved_args, device,
                          ds, plane_indices, gm_steps, gm_step_size,
                          eta_min, eta_schedule, langevin_noise,
                          noise_method='constant_scale', noise_scale=0.1,
                          channels=256, use_film=False, long_skip=False, interleave_rate=2, use_amp=False):
    """하나의 체크포인트 태그에 대해 추론 실행"""
    diopter_mode = saved_args.get('diopter_mode', 'coc')
    energy_head = saved_args.get('energy_head', 'fc')
    arch = saved_args.get('arch', 'simple')

    model, ckpt_epoch = load_model_from_ckpt(ckpt_path, diopter_mode, energy_head, device, arch, channels=channels, use_film=use_film, long_skip=long_skip, interleave_rate=interleave_rate)

    if args.compile:
        print(f"  Applying torch.compile(mode={args.compile_mode})...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"  Warning: torch.compile failed: {e}")

    print(f"\n{'='*50}")
    print(f"  [{tag}] epoch={ckpt_epoch}")
    print(f"{'='*50}")

    # 출력 폴더: inference/<tag>/
    out_subdir = os.path.join(args.run_dir, 'inference', tag)
    os.makedirs(out_subdir, exist_ok=True)

    # 추론 설정 저장
    infer_config = {
        'tag': tag,
        'epoch': ckpt_epoch,
        'gm_steps': gm_steps,
        'gm_step_size': gm_step_size,
        'eta_schedule': eta_schedule,
        'eta_min': eta_min,
        'langevin_noise': langevin_noise,
        'noise_method': noise_method,
        'noise_scale': noise_scale,
        'diopter_mode': diopter_mode,
        'plane_indices': plane_indices,
        'infer_sharp': args.infer_sharp,
        'infer_sharp_lambda': args.infer_sharp_lambda,
        'infer_sharp_gamma': args.infer_sharp_gamma,
        'infer_sharp_start': args.infer_sharp_start
    }
    with open(os.path.join(out_subdir, 'infer_config.json'), 'w') as f:
        json.dump(infer_config, f, indent=2)

    results = []
    all_step_data = {}  # plane별 step PSNR 히스토리

    for p_idx in plane_indices:
        diopter_val = float(DP_FOCAL[p_idx])
        print(f"\nGenerating plane {p_idx} ({diopter_val:.2f}D)...")

        # 해당 plane의 데이터 찾기
        sample_idx = None
        for idx, (s, pp, qp) in enumerate(ds._match_samples):
            if qp == p_idx:
                sample_idx = idx
                break

        if sample_idx is None:
            print(f"  WARNING: plane {p_idx} not found in dataset, skipping.")
            continue

        x, diopter, targets, gt = ds[sample_idx]
        x = x.unsqueeze(0).to(device)
        diopter = diopter.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        # Compositional head enable flags from saved_args
        enable_struct = saved_args.get('enable_struct', True)
        enable_percep = saved_args.get('enable_percep', True)
        enable_phys = saved_args.get('enable_phys', True)

        final_image, psnr, history, step_psnr_history = generate_one_plane(
            model, x, diopter, gt, device, gm_steps, gm_step_size,
            eta_min, eta_schedule, langevin_noise, noise_method, noise_scale,
            infer_sharp=args.infer_sharp, infer_sharp_lambda=args.infer_sharp_lambda,
            infer_sharp_gamma=args.infer_sharp_gamma, infer_sharp_start=args.infer_sharp_start, use_amp=use_amp,
            enable_struct=enable_struct, enable_percep=enable_percep, enable_phys=enable_phys
        )

        all_step_data[p_idx] = step_psnr_history

        save_path = os.path.join(out_subdir,
                                 f'scene{args.scene_idx}_plane{p_idx:02d}.png')
        visualize(history, gt.cpu(), psnr, save_path, p_idx, diopter_val)

        results.append({
            'plane_idx': p_idx,
            'diopter': diopter_val,
            'psnr': psnr
        })

    # 결과 요약 JSON
    results_path = os.path.join(out_subdir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # step별 PSNR/energy 표 & 수렴 그래프 저장
    if all_step_data:
        save_step_psnr_csv(all_step_data, out_subdir)
        plot_psnr_convergence(all_step_data, out_subdir)
        plot_energy_per_plane(all_step_data, out_subdir)

    if len(results) > 0:
        avg_psnr = np.mean([r['psnr'] for r in results])
        print(f"\n  [{tag}] Average PSNR: {avg_psnr:.2f} dB")
    else:
        avg_psnr = 0.0

    return results, avg_psnr


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.run_dir is None:
        print("ERROR: --run_dir is required for inference.")
        return

    # args.json 읽기
    args_path = os.path.join(args.run_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path) as f:
            saved_args = json.load(f)
    else:
        saved_args = {}

    diopter_mode = saved_args.get('diopter_mode', 'coc')

    # args.json에서 학습 시 설정 복원 → CLI로 명시하면 override
    train_steps = saved_args.get('gm_steps', 50)
    train_step_size = saved_args.get('gm_step_size', 0.2)
    train_channels = saved_args.get('channels', 256)

    # CLI 기본값과 같으면 args.json 값 사용 (명시적 override 감지)
    gm_steps = args.gm_steps if args.gm_steps != 50 else train_steps
    gm_step_size = args.gm_step_size if args.gm_step_size != 0.2 else train_step_size
    channels = args.channels if args.channels != 256 else train_channels

    train_use_film = saved_args.get('use_film', False)
    use_film = args.use_film or train_use_film

    train_long_skip = saved_args.get('long_skip', False)
    long_skip = args.long_skip or train_long_skip

    train_interleave_rate = saved_args.get('interleave_rate', 2)
    interleave_rate = args.interleave_rate if args.interleave_rate != 2 else train_interleave_rate

    # 새 파라미터: 학습 시 설정 복원, CLI override 가능
    train_eta_schedule = saved_args.get('eta_schedule', 'constant')
    train_eta_min = saved_args.get('eta_min', 0.002)
    train_langevin_noise = saved_args.get('langevin_noise', False)
    train_noise_method = saved_args.get('noise_method', 'constant_scale')
    train_noise_scale = saved_args.get('noise_scale', 0.1)

    eta_schedule = args.eta_schedule if args.eta_schedule != 'constant' else train_eta_schedule
    eta_min = args.eta_min if args.eta_min != 0.002 else train_eta_min
    langevin_noise = args.langevin_noise or train_langevin_noise
    noise_method = args.noise_method if args.noise_method != 'constant_scale' else train_noise_method
    noise_scale = args.noise_scale if args.noise_scale != 0.1 else train_noise_scale

    single_scene_only = saved_args.get('single_scene_only', False)

    # ── Auto-enable infer_sharp if model was trained with bypass ──
    if saved_args.get('train_bypass', False) and not args.infer_sharp:
        args.infer_sharp = True
        args.infer_sharp_lambda = saved_args.get('bypass_lambda', 5.0)
        args.infer_sharp_gamma = saved_args.get('bypass_gamma', 30.0)
        print(f"  [AUTO] infer_sharp enabled from train_bypass "
              f"(lambda={args.infer_sharp_lambda}, gamma={args.infer_sharp_gamma})")

    # single_scene_only이면 학습 데이터(train)로 검증, 아니면 test 데이터
    infer_split = 'train' if single_scene_only else 'test'

    print(f"\n{'='*50}")
    print(f"  Inference Configuration")
    print(f"{'='*50}")
    print(f"  diopter_mode : {diopter_mode}")
    print(f"  energy_head  : {saved_args.get('energy_head', 'fc')}")
    print(f"  gm_steps     : {gm_steps}  (train was {train_steps})")
    print(f"  gm_step_size : {gm_step_size}  (train was {train_step_size})")
    print(f"  eta_schedule : {eta_schedule}  (train was {train_eta_schedule})")
    print(f"  eta_min      : {eta_min}  (train was {train_eta_min})")
    print(f"  langevin_noise: {langevin_noise}  (train was {train_langevin_noise})")
    print(f"  noise_method : {noise_method}  (train was {train_noise_method})")
    print(f"  noise_scale  : {noise_scale}  (train was {train_noise_scale})")
    print(f"  channels     : {channels}  (train was {train_channels})")
    print(f"  prototype    : {single_scene_only} → split='{infer_split}'")
    print(f"  ckpt_tag     : {args.ckpt_tag}")
    print(f"  plane_idx    : {args.plane_idx}")
    print(f"  infer_sharp  : {args.infer_sharp}")
    if args.infer_sharp:
        print(f"    lambda     : {args.infer_sharp_lambda}")
        print(f"    gamma      : {args.infer_sharp_gamma}")
        print(f"    start      : {args.infer_sharp_start} ({int(gm_steps * args.infer_sharp_start)}/{gm_steps} step)")
    print(f"  compile      : {args.compile} (mode: {args.compile_mode})")
    print(f"  amp          : {args.amp}")
    print(f"{'='*50}\n")

    # 데이터셋
    generated_data_dir = args.generated_data_dir
    if generated_data_dir is None:
        generated_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split=infer_split, unmatch_ratio=0,
        diopter_mode=diopter_mode, return_gt=True,
        single_scene_only=single_scene_only,
    )

    # 생성할 plane 목록 결정
    if args.plane_idx.strip() == '-1':
        plane_indices = list(range(40))
    else:
        plane_indices = [int(x.strip()) for x in args.plane_idx.split(',')]

    # 추론할 체크포인트 목록
    ckpt_list = resolve_ckpt_paths(args.run_dir, args.ckpt_tag)
    if not ckpt_list:
        print("ERROR: No checkpoint files found. Aborting.")
        return

    # 각 체크포인트에 대해 추론
    all_summaries = {}
    for tag, ckpt_path in ckpt_list:
        results, avg_psnr = run_inference_for_tag(
            tag, ckpt_path, args, saved_args, device,
            ds, plane_indices, gm_steps, gm_step_size,
            eta_min, eta_schedule, langevin_noise,
            noise_method, noise_scale,
            channels=channels, use_film=use_film, long_skip=long_skip, interleave_rate=interleave_rate, use_amp=args.amp
        )
        all_summaries[tag] = {
            'avg_psnr': avg_psnr,
            'results': results
        }

    # 전체 요약 출력
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    for tag, summary in all_summaries.items():
        print(f"  [{tag}] Avg PSNR: {summary['avg_psnr']:.2f} dB")
    print(f"{'='*50}")

    # 전체 요약 JSON
    summary_path = os.path.join(args.run_dir, 'inference', 'summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
