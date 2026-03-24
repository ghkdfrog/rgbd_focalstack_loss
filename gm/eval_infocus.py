"""
eval_infocus.py — In-Focus Region Quality Evaluation

초점이 맞는 영역(CoC ≈ 0)의 픽셀만 추출하여 MSE/PSNR을 측정합니다.
- 모든 40개 focal plane에 대해 측정
- 대표 plane (0, 20, 39)은 마스킹 시각화 출력
- best, best_psnr, latest 체크포인트 전부 평가
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gm.model import (SimpleCNN, SimpleCNNDeep, SimpleCNNStride,
                       SimpleResNet, SimpleResNetFiLM, ResUNet,
                       SimpleConvNeXt, ConvNeXtUNet, DilatedNet, InterleaveResNet)
from gm.infer import load_model_from_ckpt, resolve_ckpt_paths, generate_one_plane
from gm.train import get_eta
from dataset_focal import FocalDataset, DP_FOCAL, NUM_FOCAL_PLANES, calculate_psnr

# ──────────────────────────────────────────────────────────────
# In-Focus Mask & Metrics
# ──────────────────────────────────────────────────────────────

def compute_infocus_mask(coc_map, threshold=0.05):
    """
    CoC 맵에서 in-focus 마스크 생성.
    coc_map: (1, H, W) or (H, W), 값 범위 [0, 1] (0 = 완전 초점)
    threshold: CoC < threshold인 픽셀을 in-focus로 판정
    Returns: bool mask (H, W)
    """
    if coc_map.dim() == 3:
        coc_map = coc_map.squeeze(0)
    return (coc_map.abs() < threshold)


def compute_infocus_metrics(pred, gt, mask):
    """
    in-focus 마스크 영역의 픽셀만 추출하여 MSE, PSNR 계산.
    pred, gt: (3, H, W) tensor [0, 1]
    mask: (H, W) bool tensor
    Returns: dict with mse, psnr, num_pixels
    """
    # 마스크를 3채널로 확장
    mask_3ch = mask.unsqueeze(0).expand_as(pred)  # (3, H, W)

    pred_pixels = pred[mask_3ch]  # 1D tensor
    gt_pixels = gt[mask_3ch]

    num_pixels = pred_pixels.numel()
    if num_pixels == 0:
        return {'mse': float('nan'), 'psnr': float('nan'), 'num_pixels': 0}

    mse = torch.mean((pred_pixels - gt_pixels) ** 2).item()
    if mse < 1e-10:
        psnr = 50.0  # cap
    else:
        psnr = 10.0 * np.log10(1.0 / mse)

    return {
        'mse': round(mse, 8),
        'psnr': round(psnr, 4),
        'num_pixels': num_pixels,
        'coverage': round(num_pixels / (pred.shape[1] * pred.shape[2] * 3), 4)
    }


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

def visualize_infocus(pred, gt, mask, plane_idx, diopter_val, metrics, save_path):
    """
    In-focus 마스킹 비교 시각화.
    마스크 외 영역은 어둡게 처리하여 in-focus 영역만 강조.
    """
    pred_np = pred.permute(1, 2, 0).numpy()
    gt_np = gt.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    # 마스크 외 영역을 어둡게 (0.2 배)
    dim_factor = 0.2
    mask_3d = np.stack([mask_np] * 3, axis=-1)

    gt_masked = np.where(mask_3d, gt_np, gt_np * dim_factor)
    pred_masked = np.where(mask_3d, pred_np, pred_np * dim_factor)

    # 차이 맵 (in-focus 영역만)
    diff = np.abs(pred_np - gt_np)
    diff_masked = np.where(mask_3d, diff, 0)
    # 차이를 강조하기 위해 스케일링
    diff_vis = np.clip(diff_masked * 5.0, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.clip(gt_masked, 0, 1))
    axes[0].set_title('GT (in-focus)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(np.clip(pred_masked, 0, 1))
    axes[1].set_title('Output (in-focus)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(diff_vis)
    axes[2].set_title('|Diff| × 5 (in-focus)', fontsize=12)
    axes[2].axis('off')

    # 마스크 시각화
    axes[3].imshow(mask_np, cmap='gray')
    axes[3].set_title(f'Mask ({metrics["coverage"]*100:.1f}% pixels)', fontsize=12)
    axes[3].axis('off')

    plt.suptitle(
        f'Plane {plane_idx} ({diopter_val:.2f}D)  |  '
        f'In-focus MSE: {metrics["mse"]:.6f}  PSNR: {metrics["psnr"]:.2f} dB',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────
# Summary Plot
# ──────────────────────────────────────────────────────────────

def plot_summary(all_results, save_dir, tag):
    """전체 plane별 in-focus MSE/PSNR + 전체 PSNR 비교 그래프"""
    planes = sorted(all_results.keys())
    diopters = [float(DP_FOCAL[p]) for p in planes]
    mses = [all_results[p]['mse'] for p in planes]
    psnrs_infocus = [all_results[p]['psnr'] for p in planes]
    psnrs_full = [all_results[p].get('full_psnr', 0) for p in planes]
    coverages = [all_results[p]['coverage'] * 100 for p in planes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. In-focus PSNR vs Full PSNR
    ax = axes[0, 0]
    ax.plot(planes, psnrs_infocus, 'o-', color='#e74c3c', label='In-focus PSNR', linewidth=2)
    ax.plot(planes, psnrs_full, 's--', color='#3498db', label='Full PSNR', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Focal Plane Index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('In-Focus vs Full PSNR per Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. In-focus MSE
    ax = axes[0, 1]
    ax.bar(planes, mses, color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Focal Plane Index')
    ax.set_ylabel('MSE')
    ax.set_title('In-Focus MSE per Plane')
    ax.grid(True, alpha=0.3)

    # 3. Coverage (in-focus 영역 비율)
    ax = axes[1, 0]
    ax.bar(planes, coverages, color='#2ecc71', alpha=0.7)
    ax.set_xlabel('Focal Plane Index')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('In-Focus Region Coverage')
    ax.grid(True, alpha=0.3)

    # 4. Diopter vs In-focus PSNR
    ax = axes[1, 1]
    ax.plot(diopters, psnrs_infocus, 'o-', color='#9b59b6', linewidth=2)
    ax.set_xlabel('Diopter (D)')
    ax.set_ylabel('In-Focus PSNR (dB)')
    ax.set_title('In-Focus PSNR vs Diopter')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'In-Focus Evaluation Summary [{tag}]', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'infocus_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────

def run_eval_for_tag(tag, ckpt_path, saved_args, device, ds, gm_steps,
                     gm_step_size, eta_min, eta_schedule, langevin_noise,
                     output_base_dir, coc_threshold=0.05, vis_planes=[0, 20, 39]):
    """하나의 체크포인트 태그에 대해 in-focus 평가 실행"""

    diopter_mode = saved_args.get('diopter_mode', 'coc')
    energy_head = saved_args.get('energy_head', 'fc')
    arch = saved_args.get('arch', 'simple')
    channels = saved_args.get('channels', 256)
    use_film = saved_args.get('use_film', False)
    long_skip = saved_args.get('long_skip', False)
    interleave_rate = saved_args.get('interleave_rate', 2)

    model, ckpt_epoch = load_model_from_ckpt(
        ckpt_path, diopter_mode, energy_head, device, arch,
        channels=channels, use_film=use_film, long_skip=long_skip,
        interleave_rate=interleave_rate
    )

    print(f"\n{'='*60}")
    print(f"  [{tag}] epoch={ckpt_epoch}  |  In-Focus Evaluation")
    print(f"  CoC threshold: {coc_threshold}")
    print(f"{'='*60}")

    # 출력 폴더
    eval_dir = os.path.join(output_base_dir, 'eval_infocus', tag)
    os.makedirs(eval_dir, exist_ok=True)

    all_results = {}
    all_planes = list(range(NUM_FOCAL_PLANES))

    for p_idx in all_planes:
        diopter_val = float(DP_FOCAL[p_idx])

        # 데이터셋에서 해당 plane 찾기
        sample_idx = None
        for idx, (s, pp, qp) in enumerate(ds._match_samples):
            if qp == p_idx:
                sample_idx = idx
                break

        if sample_idx is None:
            print(f"  [SKIP] plane {p_idx} not in dataset")
            continue

        x, diopter, targets, gt = ds[sample_idx]
        x = x.unsqueeze(0).to(device)
        diopter_t = diopter.unsqueeze(0).to(device)
        gt_t = gt.unsqueeze(0).to(device)

        # 이미지 생성
        final_image, full_psnr, _, _ = generate_one_plane(
            model, x, diopter_t, gt_t, device, gm_steps, gm_step_size,
            eta_min, eta_schedule, langevin_noise
        )

        # CoC 맵 추출 (x[:, 7:8] — 데이터셋에서 이미 계산됨)
        if x.shape[1] > 7:
            coc_map = x[0, 7:8, :, :].cpu()  # (1, H, W)
        else:
            # spatial 모드면 CoC 없음 → 전체 이미지로 평가
            coc_map = torch.zeros(1, x.shape[2], x.shape[3])

        # in-focus 마스크
        mask = compute_infocus_mask(coc_map, threshold=coc_threshold)

        # 메트릭 계산
        gt_cpu = gt.squeeze()
        metrics = compute_infocus_metrics(final_image, gt_cpu, mask)
        metrics['full_psnr'] = round(full_psnr, 4)
        metrics['diopter'] = round(diopter_val, 4)
        metrics['plane_idx'] = p_idx

        all_results[p_idx] = metrics

        status = f"plane {p_idx:2d} ({diopter_val:.2f}D) | " \
                 f"infocus PSNR: {metrics['psnr']:6.2f} dB  MSE: {metrics['mse']:.6f}  " \
                 f"full PSNR: {full_psnr:6.2f} dB  coverage: {metrics['coverage']*100:.1f}%"
        print(f"  {status}")

        # 시각화 (대표 plane만)
        if p_idx in vis_planes:
            vis_path = os.path.join(eval_dir, f'infocus_plane{p_idx:02d}.png')
            visualize_infocus(final_image, gt_cpu, mask, p_idx, diopter_val, metrics, vis_path)
            print(f"    → Saved visualization: {vis_path}")

    # ── 결과 저장 ──
    # JSON
    json_result = {
        'tag': tag,
        'epoch': ckpt_epoch,
        'coc_threshold': coc_threshold,
        'gm_steps': gm_steps,
        'gm_step_size': gm_step_size,
        'eta_schedule': eta_schedule,
        'eta_min': eta_min,
        'diopter_mode': diopter_mode,
        'planes': {str(k): v for k, v in all_results.items()},  # JSON key = str
        'summary': {}
    }

    # 요약 통계
    valid = [v for v in all_results.values() if not np.isnan(v['psnr'])]
    if valid:
        json_result['summary'] = {
            'avg_infocus_psnr': round(np.mean([v['psnr'] for v in valid]), 4),
            'avg_infocus_mse': round(np.mean([v['mse'] for v in valid]), 8),
            'avg_full_psnr': round(np.mean([v['full_psnr'] for v in valid]), 4),
            'avg_coverage': round(np.mean([v['coverage'] for v in valid]), 4),
            'min_infocus_psnr': round(min(v['psnr'] for v in valid), 4),
            'max_infocus_psnr': round(max(v['psnr'] for v in valid), 4),
            'num_planes': len(valid)
        }

    json_path = os.path.join(eval_dir, 'infocus_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    # 요약 그래프
    if all_results:
        plot_summary(all_results, eval_dir, tag)
        print(f"  Summary plot: {eval_dir}/infocus_summary.png")

    # 콘솔 요약
    if json_result['summary']:
        s = json_result['summary']
        print(f"\n  {'─'*50}")
        print(f"  [{tag}] Summary ({s['num_planes']} planes)")
        print(f"  Avg In-Focus PSNR: {s['avg_infocus_psnr']:.2f} dB")
        print(f"  Avg Full PSNR:     {s['avg_full_psnr']:.2f} dB")
        print(f"  Avg In-Focus MSE:  {s['avg_infocus_mse']:.6f}")
        print(f"  Avg Coverage:      {s['avg_coverage']*100:.1f}%")
        print(f"  {'─'*50}")

    return json_result


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='In-Focus Region Quality Evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to run directory (contains args.json, checkpoints)')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Scene index (default: 0)')
    parser.add_argument('--coc_threshold', type=float, default=0.05,
                        help='CoC threshold for in-focus mask (default: 0.05)')
    parser.add_argument('--vis_planes', type=str, default='0,20,39',
                        help='Planes to visualize (comma-separated, default: 0,20,39)')
    parser.add_argument('--ckpt_tag', type=str, default='all',
                        choices=['best', 'best_psnr', 'latest', 'all'],
                        help='Checkpoint to evaluate (default: all)')
    parser.add_argument('--gm_steps', type=int, default=None,
                        help='Override GM steps (default: from args.json)')
    parser.add_argument('--gm_step_size', type=float, default=None,
                        help='Override step size (default: from args.json)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # args.json 로드
    args_path = os.path.join(args.run_dir, 'args.json')
    if not os.path.exists(args_path):
        print(f"ERROR: args.json not found in {args.run_dir}")
        return

    with open(args_path) as f:
        saved_args = json.load(f)

    # 추론 설정
    gm_steps = args.gm_steps or saved_args.get('gm_steps', 50)
    gm_step_size = args.gm_step_size or saved_args.get('gm_step_size', 0.2)
    eta_min = saved_args.get('eta_min', 0.002)
    eta_schedule = saved_args.get('eta_schedule', 'constant')
    langevin_noise = saved_args.get('langevin_noise', False)
    diopter_mode = saved_args.get('diopter_mode', 'coc')

    vis_planes = [int(p) for p in args.vis_planes.split(',')]

    print(f"\nRun dir:  {args.run_dir}")
    print(f"Scene:    {args.scene_idx}")
    print(f"CoC thr:  {args.coc_threshold}")
    print(f"GM steps: {gm_steps}, step_size: {gm_step_size}")
    print(f"Vis planes: {vis_planes}")

    # 데이터셋
    data_dir = saved_args.get('data_dir', '../data/pbrt-v4-scenes-lf')
    generated_data_dir = saved_args.get('generated_data_dir', None)
    if generated_data_dir is None:
        generated_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    single_scene_only = saved_args.get('single_scene_only', False)
    dataset_split = 'train' if single_scene_only else 'val'
    
    ds = FocalDataset(
        data_dir, generated_data_dir,
        split=dataset_split, unmatch_ratio=0,
        diopter_mode=diopter_mode, return_gt=True,
        single_scene_only=single_scene_only, num_scenes=0
    )

    if single_scene_only:
        print(f"Dataset split: train (Prototype mode detected)")
    else:
        print(f"Dataset split: val")

    # 체크포인트 해석
    ckpt_list = resolve_ckpt_paths(args.run_dir, args.ckpt_tag)
    if not ckpt_list:
        print(f"ERROR: No checkpoints found in {args.run_dir}")
        return

    print(f"Checkpoints: {[t for t, _ in ckpt_list]}")

    for tag, ckpt_path in ckpt_list:
        run_eval_for_tag(
            tag=tag,
            ckpt_path=ckpt_path,
            saved_args=saved_args,
            device=device,
            ds=ds,
            gm_steps=gm_steps,
            gm_step_size=gm_step_size,
            eta_min=eta_min,
            eta_schedule=eta_schedule,
            langevin_noise=langevin_noise,
            output_base_dir=args.run_dir,
            coc_threshold=args.coc_threshold,
            vis_planes=vis_planes
        )

    print(f"\n{'='*60}")
    print(f"  All evaluations complete!")
    print(f"  Results in: {args.run_dir}/eval_infocus/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
