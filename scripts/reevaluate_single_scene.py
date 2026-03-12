import os
import sys
import json
import shutil
import torch
import numpy as np
import argparse
from glob import glob

# 이 스크립트는 rgbd_focalstack_loss/ 디렉토리에서 실행되어야 합니다.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gm.model import SimpleCNN, SimpleCNNDeep
from gm.train import validate, get_eta, langevin_step
from gm.infer import generate_one_plane, run_inference_for_tag, resolve_ckpt_paths
from dataset_focal import FocalDataset, DP_FOCAL, calculate_psnr


def find_single_scene_runs(base_dir):
    """지정된 base_dir 하위의 모든 run 폴더를 탐색하여 single_scene_only=True 인 run 목록을 반환합니다."""
    runs = []
    for root, dirs, files in os.walk(base_dir):
        if 'args.json' in files:
            args_path = os.path.join(root, 'args.json')
            with open(args_path, 'r') as f:
                try:
                    args_dict = json.load(f)
                    if args_dict.get('single_scene_only', False):
                        runs.append(root)
                except Exception as e:
                    print(f"Error parsing {args_path}: {e}")
    return runs


def reevaluate_run(run_dir, device='cuda', skip_inference=False):
    print(f"\n==================================================")
    print(f"Re-evaluating run: {run_dir}")
    print(f"==================================================")

    args_path = os.path.join(run_dir, 'args.json')
    with open(args_path, 'r') as f:
        saved_args = json.load(f)

    # dict를 Namespace 객체처럼 접근할 수 있게 변환
    class Args:
        pass
    args = Args()
    args.__dict__.update(saved_args)

    # 오래된 run의 args.json에 없을 수 있는 필드들에 기본값 설정
    defaults = {
        'run_dir': run_dir,
        'scene_idx': 0,
        'energy_head': 'fc',
        'eta_schedule': 'constant',
        'eta_min': 0.002,
        'langevin_noise': False,
        'num_scenes': 0,
        'arch': 'simple',
        'generated_data_dir': None,
    }
    for key, val in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, val)

    if args.generated_data_dir is None:
        args.generated_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # ── Dataset (Train set의 0번 씬으로 검증해야 함) ──
    use_coc = (args.diopter_mode == 'coc')

    val_ds = FocalDataset(
        args.data_dir, args.generated_data_dir,
        split='train', unmatch_ratio=0,
        use_coc=use_coc, return_gt=True,
        single_scene_only=True,
        num_scenes=args.num_scenes if hasattr(args, 'num_scenes') else 0
    )

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    # ── Model ──
    arch = getattr(args, 'arch', 'simple')
    if arch == 'deep':
        model = SimpleCNNDeep(diopter_mode=args.diopter_mode,
                              energy_head=args.energy_head).to(device)
    else:
        model = SimpleCNN(diopter_mode=args.diopter_mode,
                          energy_head=args.energy_head).to(device)

    # ── Checkpoint 리스트업 ──
    checkpoint_files = glob(os.path.join(run_dir, 'checkpoint_epoch_*.pth'))

    for extra_ckpt in ['best_model.pth', 'best_psnr_model.pth', 'latest.pth']:
        extra_path = os.path.join(run_dir, extra_ckpt)
        if os.path.exists(extra_path):
            checkpoint_files.append(extra_path)

    checkpoint_files = list(set(checkpoint_files))

    if not checkpoint_files:
        print("No .pth files found. Skipping.")
        return

    best_val_loss = float('inf')
    best_loss_ckpt = None

    best_mean_psnr = -float('inf')
    best_psnr_ckpt = None

    results = []
    infer_steps = max(args.gm_steps, 50)
    target_planes = [0, 20, 39]

    for ckpt_path in checkpoint_files:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"--- Evaluating {ckpt_name} ---")

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        # 1. Validation Loss 계산
        val_loss = validate(model, val_loader, device, args.gm_steps, args.gm_step_size,
                            args.eta_min, args.eta_schedule, args.langevin_noise)

        # 2. Validation PSNR 계산 (대표 3장: 0, 20, 39)
        model.eval()
        psnr_sum = 0.0

        for q_idx in target_planes:
            sample_idx = 0
            for idx, (s, p, q) in enumerate(val_ds._match_samples):
                if q == q_idx:
                    sample_idx = idx
                    break

            x, diopter, targets, gt = val_ds[sample_idx]
            x = x.unsqueeze(0).to(device)
            diopter = diopter.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)

            final_image, psnr_val, _, _ = generate_one_plane(
                model, x, diopter, gt, device, infer_steps, args.gm_step_size,
                args.eta_min, args.eta_schedule, args.langevin_noise
            )
            psnr_sum += psnr_val

        mean_psnr = psnr_sum / len(target_planes)
        print(f"  Val Loss: {val_loss:.6f} | Mean PSNR ({target_planes}): {mean_psnr:.2f} dB")

        results.append({
            'ckpt_name': ckpt_name,
            'val_loss': val_loss,
            'mean_psnr': mean_psnr,
            'ckpt_path': ckpt_path
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_ckpt = ckpt_path

        if mean_psnr > best_mean_psnr:
            best_mean_psnr = mean_psnr
            best_psnr_ckpt = ckpt_path

    print(f"\n✅ Finished evaluation: {run_dir}")
    print(f"🏆 Best Loss: {os.path.basename(best_loss_ckpt)} (Loss: {best_val_loss:.6f})")
    print(f"🏆 Best PSNR: {os.path.basename(best_psnr_ckpt)} (Mean PSNR: {best_mean_psnr:.2f} dB)")

    # ── 기존 파일 백업 후 저장 ──
    best_loss_dest = os.path.join(run_dir, 'best_model.pth')
    if best_loss_ckpt != best_loss_dest:
        if os.path.exists(best_loss_dest):
            shutil.copy2(best_loss_dest, os.path.join(run_dir, 'best_model_old.pth'))
        shutil.copy2(best_loss_ckpt, best_loss_dest)

    best_psnr_dest = os.path.join(run_dir, 'best_psnr_model.pth')
    if best_psnr_ckpt != best_psnr_dest:
        if os.path.exists(best_psnr_dest):
            shutil.copy2(best_psnr_dest, os.path.join(run_dir, 'best_psnr_model_old.pth'))
        shutil.copy2(best_psnr_ckpt, best_psnr_dest)

    # ── JSON 로그 저장 ──
    with open(os.path.join(run_dir, 'reevaluation_results.json'), 'w') as f:
        json.dump({
            'best_loss_ckpt': os.path.basename(best_loss_ckpt),
            'best_loss_val': best_val_loss,
            'best_psnr_ckpt': os.path.basename(best_psnr_ckpt),
            'best_psnr_val': best_mean_psnr,
            'all_evals': results
        }, f, indent=2)

    # ── 자동 Inference (best, best_psnr, latest 이미지 생성) ──
    if skip_inference:
        print("Skipping inference step.")
        return

    print(f"\n{'='*50}")
    print(f"Running auto-inference for {run_dir}...")
    print(f"{'='*50}")

    plane_indices = [0, 20, 39]
    gm_steps = infer_steps
    gm_step_size = args.gm_step_size
    eta_min = args.eta_min
    eta_schedule = args.eta_schedule
    langevin_noise = args.langevin_noise

    ckpt_list = resolve_ckpt_paths(run_dir, 'all')
    for tag, ckpt_path in ckpt_list:
        run_inference_for_tag(
            tag, ckpt_path, args, saved_args, device,
            val_ds, plane_indices, gm_steps, gm_step_size,
            eta_min, eta_schedule, langevin_noise
        )

    print(f"\n✅ All done for {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Re-evaluate single_scene_only runs on correct dataset and auto-infer'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='특정 run 폴더 또는 상위 output 디렉토리')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Inference 단계를 건너뛸지 여부')
    cli_args = parser.parse_args()

    target_dir = cli_args.run_dir

    if os.path.exists(os.path.join(target_dir, 'args.json')):
        runs_to_process = [target_dir]
    else:
        runs_to_process = find_single_scene_runs(target_dir)

    print(f"Found {len(runs_to_process)} single_scene_only run(s) to re-evaluate.")

    for run in runs_to_process:
        reevaluate_run(run, device=cli_args.device, skip_inference=cli_args.skip_inference)
