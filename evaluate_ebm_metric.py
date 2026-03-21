"""
EBM → Metric Model 평가 스크립트

GM(EBM) 모델로 생성한 이미지를 Metric Prediction 모델(LossEstimationNet)에
입력하여, 모델이 예측한 metric(PSNR/SSIM/LPIPS)과 실제 metric을 비교합니다.

Usage:
    python evaluate_ebm_metric.py \
        --gm_run_dir runs_gm/<gm_run_name> \
        --metric_run_dir runs/<metric_run_name> \
        --plane_idx 0,20,39

    # 특정 GM 체크포인트만 사용
    python evaluate_ebm_metric.py \
        --gm_run_dir runs_gm/<gm_run_name> \
        --metric_run_dir runs/<metric_run_name> \
        --gm_ckpt_tag best_psnr
"""

import os
import sys
import json
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(__file__))

from model import LossEstimationNet, calCoC
from gm.model import (SimpleCNN, SimpleCNNDeep, SimpleCNNStride,
                       SimpleResNet, ResUNet, SimpleConvNeXt,
                       ConvNeXtUNet, DilatedNet)
from gm.train import get_eta, langevin_step
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr

import lpips as lpips_lib


# ──────────────────────────────────────────────────────────────
# GM 모델 로드
# ──────────────────────────────────────────────────────────────
def load_gm_model(ckpt_path, device, saved_args):
    """GM(EBM) 체크포인트 로드"""
    ckpt = torch.load(ckpt_path, map_location=device)

    arch = saved_args.get('arch', 'simple')
    diopter_mode = ckpt.get('diopter_mode', saved_args.get('diopter_mode', 'coc'))
    energy_head = ckpt.get('energy_head', saved_args.get('energy_head', 'fc'))
    channels = ckpt.get('channels', saved_args.get('channels', 256))
    use_film = ckpt.get('use_film', saved_args.get('use_film', False))
    long_skip = ckpt.get('long_skip', saved_args.get('long_skip', False))

    if arch == 'film_resnet':
        arch = 'resnet'
        use_film = True

    if arch == 'deep':
        model = SimpleCNNDeep(diopter_mode=diopter_mode, energy_head=energy_head)
    elif arch == 'stride':
        model = SimpleCNNStride(diopter_mode=diopter_mode, energy_head=energy_head)
    elif arch == 'resnet':
        model = SimpleResNet(diopter_mode=diopter_mode, energy_head=energy_head,
                             num_blocks=4, channels=channels, use_film=use_film,
                             long_skip=long_skip)
    elif arch == 'resunet':
        model = ResUNet(diopter_mode=diopter_mode, energy_head=energy_head,
                        base_channels=channels, num_bottleneck_blocks=3,
                        use_film=use_film)
    elif arch == 'convnext':
        model = SimpleConvNeXt(diopter_mode=diopter_mode, energy_head=energy_head,
                               num_blocks=4, channels=channels, use_film=use_film)
    elif arch == 'convnext_unet':
        model = ConvNeXtUNet(diopter_mode=diopter_mode, energy_head=energy_head,
                             num_blocks=9, channels=channels, use_film=use_film)
    elif arch == 'dilated':
        model = DilatedNet(diopter_mode=diopter_mode, energy_head=energy_head)
    else:
        model = SimpleCNN(diopter_mode=diopter_mode, energy_head=energy_head)

    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    print(f"  GM model loaded: arch={arch}, epoch={epoch}, ch={channels}, "
          f"film={use_film}, long_skip={long_skip}")
    return model, diopter_mode


# ──────────────────────────────────────────────────────────────
# Metric 모델 로드
# ──────────────────────────────────────────────────────────────
def load_metric_model(run_dir, device):
    """LossEstimationNet 체크포인트 로드"""
    candidates = ['best_model_match_only.pth', 'best_model.pth', 'latest.pth']
    ckpt_path = None
    for c in candidates:
        p = os.path.join(run_dir, c)
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No metric checkpoint found in {run_dir}")

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
    print(f"  Metric model loaded: {os.path.basename(ckpt_path)}, "
          f"epoch={epoch}, version={version}, mode={diopter_mode}")
    return model, diopter_mode, version


# ──────────────────────────────────────────────────────────────
# GM 이미지 생성
# ──────────────────────────────────────────────────────────────
def generate_image_with_gm(gm_model, gm_dataset, device, plane_idx,
                           gm_steps, gm_step_size, eta_min, eta_schedule,
                           langevin_noise):
    """GM 모델로 한 장의 focal plane 이미지 생성"""
    # dataset에서 해당 plane의 sample 찾기
    sample_idx = None
    for idx, (s, pp, qp) in enumerate(gm_dataset._match_samples):
        if qp == plane_idx:
            sample_idx = idx
            break

    if sample_idx is None:
        print(f"  WARNING: plane {plane_idx} not found in GM dataset")
        return None, None

    x, diopter, targets, gt = gm_dataset[sample_idx]
    x = x.unsqueeze(0).to(device)
    diopter = diopter.unsqueeze(0).to(device)
    gt = gt.unsqueeze(0).to(device)

    with torch.enable_grad():
        current_image = torch.randn_like(gt).to(device)
        N, C, H, W = x.shape

        for step in range(gm_steps):
            eta = get_eta(step, gm_steps, gm_step_size, eta_min, eta_schedule)
            current_image.requires_grad_(True)
            input_rgbd = x[:, :4, :, :]
            if C > 7:
                input_tail = x[:, 7:, :, :]
                model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
            else:
                model_input = torch.cat([input_rgbd, current_image], dim=1)

            energy = gm_model(model_input, diopter)
            grad = torch.autograd.grad(
                energy, current_image, torch.ones_like(energy),
                create_graph=False
            )[0]

            noise = langevin_noise and (step < gm_steps - 1)
            current_image = langevin_step(current_image, grad, eta, noise)

    generated = torch.clamp(current_image, 0, 1).detach()
    return generated, gt


# ──────────────────────────────────────────────────────────────
# 실제 메트릭 계산
# ──────────────────────────────────────────────────────────────
def compute_actual_metrics(generated, gt, lpips_fn):
    """생성 이미지와 GT 사이의 실제 PSNR/SSIM/LPIPS 계산"""
    gen_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # PSNR
    mse = float(np.mean((gen_np - gt_np) ** 2))
    psnr = float(np.clip(-10 * np.log10(mse + 1e-10), 0, 100))

    # SSIM
    ssim = float(ssim_fn(gen_np, gt_np, data_range=1.0, channel_axis=2))

    # LPIPS
    gen_lpips = generated * 2 - 1    # [0,1] → [-1,1]
    gt_lpips = gt * 2 - 1
    with torch.no_grad():
        lpips_val = float(lpips_fn(gen_lpips.cpu(), gt_lpips.cpu()).item())
    lpips_val = float(np.clip(lpips_val, 0.0, 1.0))

    return psnr, ssim, lpips_val


# ──────────────────────────────────────────────────────────────
# Metric 모델 예측
# ──────────────────────────────────────────────────────────────
def predict_metrics(metric_model, metric_diopter_mode,
                    rgb, depth, generated, diopter_val, device):
    """Metric 모델로 PSNR/SSIM/LPIPS 예측"""
    N = 1
    H, W = rgb.shape[1], rgb.shape[2]

    # rgb: (3, H, W), depth: (1, H, W), generated: (1, 3, H, W)
    gen_3ch = generated.squeeze(0)  # (3, H, W)

    if metric_diopter_mode == 'coc':
        # CoC map 계산
        depth_t = depth.unsqueeze(0).to(device)  # (1, 1, H, W)
        diopter_t = torch.tensor([diopter_val], dtype=torch.float32).to(device)
        coc = calCoC(depth_t, diopter_t)  # (1, 1, H, W)
        coc = coc.squeeze(0)  # (1, H, W)
        x = torch.cat([rgb.to(device), depth.to(device), gen_3ch.to(device), coc], dim=0)
    else:
        x = torch.cat([rgb.to(device), depth.to(device), gen_3ch.to(device)], dim=0)

    x = x.unsqueeze(0)  # (1, 7or8, H, W)
    diopter_t = torch.tensor([diopter_val], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = metric_model(x, diopter_t)

    p_psnr = pred['psnr'].item() * 100.0   # sigmoid → dB
    p_ssim = pred['ssim'].item()
    p_lpips = pred['lpips'].item()

    return p_psnr, p_ssim, p_lpips


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='EBM 생성 이미지 → Metric 모델 예측 비교')

    # 경로
    parser.add_argument('--gm_run_dir', type=str, required=True,
                        help='GM(EBM) run 디렉토리 경로')
    parser.add_argument('--metric_run_dir', type=str, required=True,
                        help='Metric prediction 모델 run 디렉토리 경로')
    parser.add_argument('--data_dir', type=str, default='../varifocal/data',
                        help='원본 EXR 데이터 경로')

    # 평가 설정
    parser.add_argument('--plane_idx', type=str, default='0,20,39',
                        help='평가할 focal plane indices (쉼표 구분)')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Scene index')
    parser.add_argument('--gm_ckpt_tag', type=str, default='best_psnr',
                        choices=['best', 'best_psnr', 'latest'],
                        help='사용할 GM 체크포인트 (default: best_psnr)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── GM args.json 로드 ──
    gm_args_path = os.path.join(args.gm_run_dir, 'args.json')
    if os.path.exists(gm_args_path):
        with open(gm_args_path) as f:
            gm_saved_args = json.load(f)
    else:
        gm_saved_args = {}

    gm_diopter_mode = gm_saved_args.get('diopter_mode', 'coc')
    gm_steps = gm_saved_args.get('gm_steps', 50)
    gm_step_size = gm_saved_args.get('gm_step_size', 0.2)
    gm_eta_schedule = gm_saved_args.get('eta_schedule', 'constant')
    gm_eta_min = gm_saved_args.get('eta_min', 0.002)
    gm_langevin_noise = gm_saved_args.get('langevin_noise', False)
    gm_single_scene = gm_saved_args.get('single_scene_only', False)

    # ── GM 체크포인트 경로 ──
    tag = args.gm_ckpt_tag
    ckpt_map = {
        'best': 'best_model.pth',
        'best_psnr': 'best_psnr_model.pth',
        'latest': 'latest.pth',
    }
    gm_ckpt_path = os.path.join(args.gm_run_dir, ckpt_map[tag])
    if not os.path.exists(gm_ckpt_path):
        print(f"ERROR: GM checkpoint not found: {gm_ckpt_path}")
        return

    # ── 모델 로드 ──
    print("\n[1] Loading GM (EBM) model...")
    gm_model, _ = load_gm_model(gm_ckpt_path, device, gm_saved_args)

    print("\n[2] Loading Metric prediction model...")
    metric_model, metric_diopter_mode, metric_version = load_metric_model(
        args.metric_run_dir, device)

    # ── 데이터셋 (GM용) ──
    generated_data_dir = gm_saved_args.get('generated_data_dir', None)
    if generated_data_dir is None:
        generated_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    gm_ds = FocalDataset(
        args.data_dir, generated_data_dir,
        split='train' if gm_single_scene else 'test',
        unmatch_ratio=0,
        diopter_mode=gm_diopter_mode,
        return_gt=True,
        single_scene_only=gm_single_scene,
    )

    # ── LPIPS 함수 초기화 ──
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_fn = lpips_lib.LPIPS(net='alex')
    lpips_fn.eval()

    # ── 평가할 plane 목록 ──
    plane_indices = [int(x.strip()) for x in args.plane_idx.split(',')]

    # ── RGBD 로드 (scene별 공유) ──
    scene_idx = args.scene_idx if not gm_single_scene else 0
    rgb_np, depth_np = gm_ds._load_rgb_depth(gm_ds.scenes[0])
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float()     # (3, H, W)
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).float()       # (1, H, W)

    # ── 평가 실행 ──
    print(f"\n{'='*70}")
    print(f"  GM: {os.path.basename(args.gm_run_dir)} [{tag}]")
    print(f"  Metric: {os.path.basename(args.metric_run_dir)}")
    print(f"  Planes: {plane_indices}")
    print(f"  GM steps: {gm_steps}, eta_schedule: {gm_eta_schedule}")
    print(f"{'='*70}\n")

    results = []

    for p_idx in plane_indices:
        diopter_val = float(DP_FOCAL[p_idx])
        print(f"── Plane {p_idx} ({diopter_val:.2f}D) ──")

        # 1. GM으로 이미지 생성
        print(f"  Generating with GM ({gm_steps} steps)...")
        generated, gt = generate_image_with_gm(
            gm_model, gm_ds, device, p_idx,
            gm_steps, gm_step_size, gm_eta_min, gm_eta_schedule, gm_langevin_noise
        )
        if generated is None:
            continue

        # 2. 실제 메트릭 계산
        actual_psnr, actual_ssim, actual_lpips = compute_actual_metrics(
            generated, gt, lpips_fn)

        # 3. Metric 모델 예측
        pred_psnr, pred_ssim, pred_lpips = predict_metrics(
            metric_model, metric_diopter_mode,
            rgb_t, depth_t, generated, diopter_val, device)

        # 4. 결과 출력
        print(f"  {'':15s} {'Predicted':>12s} {'Actual':>12s} {'Error':>12s}")
        print(f"  {'PSNR (dB)':15s} {pred_psnr:12.2f} {actual_psnr:12.2f} {abs(pred_psnr - actual_psnr):12.2f}")
        print(f"  {'SSIM':15s} {pred_ssim:12.4f} {actual_ssim:12.4f} {abs(pred_ssim - actual_ssim):12.4f}")
        print(f"  {'LPIPS':15s} {pred_lpips:12.4f} {actual_lpips:12.4f} {abs(pred_lpips - actual_lpips):12.4f}")
        print()

        results.append({
            'plane_idx': p_idx,
            'diopter': diopter_val,
            'predicted': {'psnr': pred_psnr, 'ssim': pred_ssim, 'lpips': pred_lpips},
            'actual':    {'psnr': actual_psnr, 'ssim': actual_ssim, 'lpips': actual_lpips},
            'error':     {
                'psnr': abs(pred_psnr - actual_psnr),
                'ssim': abs(pred_ssim - actual_ssim),
                'lpips': abs(pred_lpips - actual_lpips),
            },
        })

    # ── 요약 ──
    if results:
        print(f"{'='*70}")
        print(f"  Summary (Average Error)")
        print(f"{'='*70}")
        avg_err_psnr = np.mean([r['error']['psnr'] for r in results])
        avg_err_ssim = np.mean([r['error']['ssim'] for r in results])
        avg_err_lpips = np.mean([r['error']['lpips'] for r in results])
        print(f"  PSNR  MAE: {avg_err_psnr:.2f} dB")
        print(f"  SSIM  MAE: {avg_err_ssim:.4f}")
        print(f"  LPIPS MAE: {avg_err_lpips:.4f}")
        print(f"{'='*70}")

        # JSON 저장
        out_dir = os.path.join(args.gm_run_dir, 'ebm_metric_eval')
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f'eval_{tag}.json')
        summary = {
            'gm_run': args.gm_run_dir,
            'gm_ckpt': tag,
            'metric_run': args.metric_run_dir,
            'planes': plane_indices,
            'results': results,
            'avg_error': {
                'psnr': avg_err_psnr,
                'ssim': avg_err_ssim,
                'lpips': avg_err_lpips,
            }
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
