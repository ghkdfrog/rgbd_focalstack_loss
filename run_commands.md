# DeepFocus Loss Estimation Model - 실행 커맨드 정리

---

## 1. 데이터 생성

```bash
# Weak + Strong + AiF 한 번에
python generate_augmented_dataset.py --data_dir ../varifocal/data --mode all
```

---

## 2. 데이터 분포 확인

```bash
python check_distribution.py
```
→ `data_distribution.png` 저장됨

---

## 3. 학습

### 3-A. 기존 방식 (S+W+A + Adversarial)
```bash
python train_robust.py \
  --epochs 60 \
  --use_augmented \
  --use_weak \
  --use_aif \
  --adv_mode append \
  --adv_ratio 0.33 \
  --batch_size 8 \
  --num_workers 4
```

### 3-B. 새 방식 — Focal Distance-Aware Training ⭐
> DeepFocus 예측 이미지만 사용. Sinusoidal Encoding으로 diopter 조건을 게대로 학습.

```bash
# 추천 설정
python train_focal.py \
    --data_dir ../varifocal/data \
    --diopter_mode sinusoidal \
    --unmatch_ratio 3 \
    --epochs 20 \
    --batch_size 8

# Spatial 모드 (기존 방식 대조군)
python train_focal.py \
    --data_dir ../varifocal/data \
    --diopter_mode spatial \
    --unmatch_ratio 3 \
    --epochs 20
```

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--diopter_mode` | `sinusoidal` | `spatial` (기존 broadcast) / `sinusoidal` (새 인코딩) |
| `--unmatch_ratio` | `3` | Match 1개당 Unmatch 몇 개. 0이면 Match만 학습 |
| `--sin_freqs` | `8` | Sinusoidal 주파수 개수 (→ 16차원 인코딩) |
| `--epochs` | `20` | 학습 에폭 수 |
| `--save_every` | `5` | N 에폭마다 체크포인트 저장 |

#### Match / Unmatch 개념
| 종류 | pred_plane | query_diopter | 모델이 배워야 할 것 |
| :--- | :--- | :--- | :--- |
| **Match** | `frame_i` | `diopter[i]` | 높은 점수 (초점 일치) |
| **Unmatch** | `frame_i` | `diopter[j≠i]` | 낮은 점수 (초점 불일치) |

- Match: 3,600개 고정 (90 scenes × 40 planes)
- Unmatch: 매 에폭마다 새로 랜덤 샘플링 → K=3이면 10,800개/에폭

---

## 4. 검증

### 4-A. Pixel Optimization Test (Loss Hacking 방어 여부)
```bash
python verify_as_loss.py \
  --checkpoint runs/run_YYYYMMDD/best_model.pth \
  --num_steps 300
```
**성공 기준**: Real PSNR 상승, Pred PSNR 폭주 없음

### 4-B. Focal Perception Test (diopter 인식 능력 확인)
```bash
# 특정 focal plane 이미지 고정, 40개 diopter에 대한 점수 비교
python compare_models.py \
    --ckpt_a runs/[baseline]/best_model.pth \
    --ckpt_b runs/[new_focal]/best_model.pth \
    --label_a "Baseline" --label_b "Focal Sinusoidal" \
    --scene_idx 110 \
    --fixed_input 1.0   # 또는 aif / 0.5 / 2.0 등

# 결과 분석 (Peak가 정확히 맞는지 자동 체크)
python analyze_compare_results.py
```
**성공 기준**: PSNR Peak가 `fixed_input` diopter에서 발생 (현재 두 모델 모두 실패)

---

## 5. 일반 평가

```bash
python evaluate.py \
  --checkpoint runs/run_YYYYMMDD/best_model.pth
```