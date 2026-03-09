"""
Gradient Matching Inference Script (SimpleCNN)

Usage:
    python -m gm.infer --run_dir runs_gm/<run_name> --plane_idx 20
    python -m gm.infer --run_dir runs_gm/<run_name> --plane_idx -1   # 전체 40장
    python -m gm.infer --run_dir runs_gm/<run_name> --plane_idx 20 --eta_schedule cosine
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gm.model import SimpleCNN
from gm.config import parse_args
from gm.train import get_eta, langevin_step
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr


def load_model(run_dir, device):
    """체크포인트에서 모델 로드 + args.json에서 설정 복원"""
    # args.json 읽기
    args_path = os.path.join(run_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path) as f:
            saved_args = json.load(f)
    else:
        saved_args = {}

    diopter_mode = saved_args.get('diopter_mode', 'coc')
    energy_head = saved_args.get('energy_head', 'fc')

    # 체크포인트 로드
    ckpt_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, 'latest.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 체크포인트 우선
    diopter_mode = ckpt.get('diopter_mode', diopter_mode)
    energy_head = ckpt.get('energy_head', energy_head)

    model = SimpleCNN(diopter_mode=diopter_mode, energy_head=energy_head).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, diopter_mode, saved_args


def generate_one_plane(model, x, diopter, gt, device, gm_steps, gm_step_size,
                       eta_min=0.002, eta_schedule='constant', use_langevin_noise=False):
    """한 장의 focal plane을 생성하고 PSNR과 히스토리를 반환"""
    N, C, H, W = x.shape

    with torch.enable_grad():
        current_image = torch.randn_like(gt).to(device)
        history = [current_image.detach().cpu()]

        for step in range(gm_steps):
            eta = get_eta(step, gm_steps, gm_step_size, eta_min, eta_schedule)
            current_image.requires_grad_(True)

            input_rgbd = x[:, :4, :, :]
            if C > 7:
                input_tail = x[:, 7:, :, :]
                model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
            else:
                model_input = torch.cat([input_rgbd, current_image], dim=1)

            energy = model(model_input, diopter)

            grad = torch.autograd.grad(
                outputs=energy,
                inputs=current_image,
                grad_outputs=torch.ones_like(energy),
                create_graph=False
            )[0]

            # 마지막 스텝에서는 노이즈 없이 깨끗하게 마무리
            noise = use_langevin_noise and (step < gm_steps - 1)
            current_image = langevin_step(current_image, grad, eta, noise)

            if (step + 1) % 10 == 0 or step == gm_steps - 1:
                history.append(current_image.cpu())

    final_image = torch.clamp(current_image, 0.0, 1.0).cpu().squeeze()
    gt_disp = gt.cpu().squeeze()
    psnr = calculate_psnr(final_image, gt_disp).item()

    return final_image, psnr, history


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


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.run_dir is None:
        print("ERROR: --run_dir is required for inference.")
        return

    model, diopter_mode, saved_args = load_model(args.run_dir, device)
    use_coc = (diopter_mode == 'coc')

    # args.json에서 학습 시 설정 복원 → CLI로 명시하면 override
    train_steps = saved_args.get('gm_steps', 50)
    train_step_size = saved_args.get('gm_step_size', 0.2)

    # CLI 기본값과 같으면 args.json 값 사용 (명시적 override 감지)
    gm_steps = args.gm_steps if args.gm_steps != 50 else train_steps
    gm_step_size = args.gm_step_size if args.gm_step_size != 0.2 else train_step_size

    # 새 파라미터: 학습 시 설정 복원, CLI override 가능
    train_eta_schedule = saved_args.get('eta_schedule', 'constant')
    train_eta_min = saved_args.get('eta_min', 0.002)
    train_langevin_noise = saved_args.get('langevin_noise', False)

    eta_schedule = args.eta_schedule if args.eta_schedule != 'constant' else train_eta_schedule
    eta_min = args.eta_min if args.eta_min != 0.002 else train_eta_min
    langevin_noise = args.langevin_noise or train_langevin_noise

    single_scene_only = saved_args.get('single_scene_only', False)

    # prototype(single_scene_only)이면 학습 데이터로 검증, 아니면 val 데이터
    infer_split = 'train' if single_scene_only else 'val'

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
    print(f"  prototype    : {single_scene_only} → split='{infer_split}'")
    print(f"  plane_idx    : {args.plane_idx}")
    print(f"{'='*50}\n")

    # 데이터셋
    generated_data_dir = args.generated_data_dir
    if generated_data_dir is None:
        generated_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split=infer_split, unmatch_ratio=0,
        use_coc=use_coc, return_gt=True,
        single_scene_only=single_scene_only
    )

    # 생성할 plane 목록 결정
    if args.plane_idx.strip() == '-1':
        plane_indices = list(range(40))
    else:
        plane_indices = [int(x.strip()) for x in args.plane_idx.split(',')]

    # 출력 폴더: inference/steps{N}_ss{size}/
    ss_str = f"{gm_step_size}".replace('.', 'p')
    out_subdir = os.path.join(args.run_dir, 'inference', f'steps{gm_steps}_ss{ss_str}')
    os.makedirs(out_subdir, exist_ok=True)

    # 추론 설정 저장
    infer_config = {
        'gm_steps': gm_steps,
        'gm_step_size': gm_step_size,
        'train_gm_steps': train_steps,
        'train_gm_step_size': train_step_size,
        'eta_schedule': eta_schedule,
        'eta_min': eta_min,
        'langevin_noise': langevin_noise,
        'diopter_mode': diopter_mode,
        'split': infer_split,
        'prototype': single_scene_only,
        'plane_indices': plane_indices
    }
    with open(os.path.join(out_subdir, 'infer_config.json'), 'w') as f:
        json.dump(infer_config, f, indent=2)

    results = []

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

        final_image, psnr, history = generate_one_plane(
            model, x, diopter, gt, device, gm_steps, gm_step_size,
            eta_min, eta_schedule, langevin_noise
        )

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
    print(f"\nResults saved to {results_path}")

    if len(results) > 1:
        avg_psnr = np.mean([r['psnr'] for r in results])
        print(f"Average PSNR: {avg_psnr:.2f} dB")


if __name__ == '__main__':
    main()
