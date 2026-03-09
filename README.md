# RGBD Focal Stack Loss Estimation Model

Reference-free quality metric for focal stack predictions from RGBD input.

## Quick Start

### 1. Generate Training Dataset
```bash
cd rgbd_focalstack_loss
python generate_dataset.py
```
This runs DeepFocus LFS on all 130 seed scenes and computes MSE/SSIM/LPIPS metrics (~6GB, 2-3 hours on GPU).

### 2. Train Model
```bash
python train.py --epochs 50 --batch_size 8
```

### 3. Monitor Training
```bash
tensorboard --logdir runs/run_TIMESTAMP/logs
```

## Model Architecture

**Input:**
- RGBD: 4 channels (RGB + depth)
- Predicted focal plane: 3 channels
- Diopter: scalar → spatially replicated

**Architecture:**
- ResNet-like backbone with residual blocks
- Global Average Pooling
- Multi-head output: MSE, SSIM, LPIPS

**Output:** 3 scalar predictions for quality metrics

## Dataset

- Train: seed0000-seed0089 (90 scenes × 40 planes = 3,600 samples)
- Val: seed0090-seed0109 (20 scenes × 40 planes = 800 samples)
- Test: seed0110-seed0129 (15 scenes × 40 planes = 600 samples)

## Files

- `generate_dataset.py`: Data generation script
- `model.py`: Network architecture
- `dataset.py`: PyTorch Dataset class
- `train.py`: Training script
