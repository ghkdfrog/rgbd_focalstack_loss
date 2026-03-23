import os
import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Interactive Inference Viewer\n",
    "이 노트북을 사용하면 특정 실행 폴더(`run dir`), `scene`, `plane`을 입력하여 추론 결과 이미지와 Ground Truth 이미지를 비교할 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import json\n",
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# 현재 디렉토리를 path에 추가\n",
    "sys.path.insert(0, os.path.abspath('.'))\n",
    "\n",
    "from gm.infer import load_model_from_ckpt, generate_one_plane\n",
    "from dataset_focal import FocalDataset, DP_FOCAL\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 설정 (Configuration)\n",
    "원하는 옵션으로 아래 값들을 변경하세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# === CONFIGURATION ===\n",
    "RUN_DIR = r'runs_gm/gm_scene0_coc_linear_20260319_150917'\n",
    "SCENE_IDX = 0      # 확인할 Scene 번호\n",
    "PLANE_IDX = 0      # 확인할 Focal Plane (0~39)\n",
    "CKPT_TAG = 'best'  # 'best', 'best_psnr', 'latest' 중 선택\n",
    "# =====================\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 모델 및 데이터셋 로드 & 추론 실행"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "# args.json 로드 (학습 시 설정 불러오기)\n",
    "args_path = os.path.join(RUN_DIR, 'args.json')\n",
    "with open(args_path) as f:\n",
    "    saved_args = json.load(f)\n",
    "\n",
    "diopter_mode = saved_args.get('diopter_mode', 'coc')\n",
    "energy_head = saved_args.get('energy_head', 'fc')\n",
    "arch = saved_args.get('arch', 'simple')\n",
    "channels = saved_args.get('channels', 256)\n",
    "use_film = saved_args.get('use_film', False)\n",
    "long_skip = saved_args.get('long_skip', False)\n",
    "interleave_rate = saved_args.get('interleave_rate', 2)\n",
    "\n",
    "# 체크포인트 경로 설정\n",
    "tag_to_filename = {\n",
    "    'best':      'best_model.pth',\n",
    "    'best_psnr': 'best_psnr_model.pth',\n",
    "    'latest':    'latest.pth',\n",
    "}\n",
    "ckpt_path = os.path.join(RUN_DIR, tag_to_filename[CKPT_TAG])\n",
    "\n",
    "# 모델 로드\n",
    "model, ckpt_epoch = load_model_from_ckpt(\n",
    "    ckpt_path, diopter_mode, energy_head, device, arch, \n",
    "    channels=channels, use_film=use_film, long_skip=long_skip, interleave_rate=interleave_rate\n",
    ")\n",
    "print(f\"Model loaded from epoch {ckpt_epoch}\")\n",
    "\n",
    "# 데이터셋 설정 (단일 scene이면 학습 데이터로 간주)\n",
    "single_scene_only = saved_args.get('single_scene_only', False)\n",
    "infer_split = 'train' if single_scene_only else 'test'\n",
    "\n",
    "data_dir = saved_args.get('data_dir') or 'dataset'\n",
    "generated_data_dir = saved_args.get('generated_data_dir') or 'data'\n",
    "\n",
    "ds = FocalDataset(\n",
    "    data_dir, generated_data_dir,\n",
    "    split=infer_split, unmatch_ratio=0,\n",
    "    diopter_mode=diopter_mode, return_gt=True,\n",
    "    single_scene_only=single_scene_only,\n",
    ")\n",
    "\n",
    "# 원하는 Scene과 Plane에 해당하는 인덱스 찾기\n",
    "sample_idx = None\n",
    "for idx, (s, pp, qp) in enumerate(ds._match_samples):\n",
    "    if qp == PLANE_IDX and s == SCENE_IDX:\n",
    "        sample_idx = idx\n",
    "        break\n",
    "\n",
    "if sample_idx is None:\n",
    "    print(f\"Dataset에서 Scene {SCENE_IDX}, Plane {PLANE_IDX}을 찾을 수 없습니다.\")\n",
    "else:\n",
    "    print(f\"Inference 시작 (Scene {SCENE_IDX}, Plane {PLANE_IDX})...\")\n",
    "    x, diopter, targets, gt = ds[sample_idx]\n",
    "    x = x.unsqueeze(0).to(device)\n",
    "    diopter = diopter.unsqueeze(0).to(device)\n",
    "    gt = gt.unsqueeze(0).to(device)\n",
    "    \n",
    "    gm_steps = saved_args.get('gm_steps', 50)\n",
    "    gm_step_size = saved_args.get('gm_step_size', 0.2)\n",
    "    eta_min = saved_args.get('eta_min', 0.002)\n",
    "    eta_schedule = saved_args.get('eta_schedule', 'constant')\n",
    "    use_langevin_noise = saved_args.get('langevin_noise', False)\n",
    "    \n",
    "    final_image, psnr, history, _ = generate_one_plane(\n",
    "        model, x, diopter, gt, device, gm_steps, gm_step_size,\n",
    "        eta_min, eta_schedule, use_langevin_noise\n",
    "    )\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n",
    "    im_inferred = final_image.squeeze().permute(1, 2, 0).numpy()\n",
    "    im_gt = gt.cpu().squeeze().permute(1, 2, 0).numpy()\n",
    "    \n",
    "    axes[0].imshow(np.clip(im_inferred, 0, 1))\n",
    "    axes[0].set_title(f\"Inferred (PSNR: {psnr:.2f} dB)\")\n",
    "    axes[0].axis('off')\n",
    "    \n",
    "    axes[1].imshow(np.clip(im_gt, 0, 1))\n",
    "    axes[1].set_title(\"Ground Truth\")\n",
    "    axes[1].axis('off')\n",
    "    \n",
    "    plt.suptitle(f\"Scene {SCENE_IDX}, Plane {PLANE_IDX} ({float(DP_FOCAL[PLANE_IDX]):.2f}D)\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Inference_Viewer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print("Notebook generated: Inference_Viewer.ipynb")
