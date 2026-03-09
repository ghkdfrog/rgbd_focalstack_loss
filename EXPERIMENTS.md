# Focal Loss Estimation — Experiment Log

## 1. 목적

DeepFocus 예측 이미지가 주어진 query diopter에서 얼마나 좋은 품질인지(PSNR, SSIM, LPIPS)를 예측하는 `LossEstimationNet`을 학습하는 것.

- **Match sample**: pred_plane == query_plane → 품질이 높아야 함
- **Unmatch sample**: pred_plane ≠ query_plane → 품질이 낮아야 함

모델이 두 케이스를 구별하는 능력 = focal distance discrimination ability.

---

## 2. 모델 아키텍처

### `LossEstimationNet`
- **입력**: RGB(3) + Depth(1) + pred_focal_plane(3) = **7ch** (+ diopter conditioning)
- **백본**: ResNet-like (Conv1 → Layer1~3 → GAP → 512-dim feature)
- **출력**: 3개 헤드 — PSNR (sigmoid, [0,1]), SSIM (sigmoid, [0,1]), LPIPS (sigmoid, [0,1])
- **파라미터**: ~11M

### Diopter Conditioning 방식 3종

| Mode | 방식 | 특징 |
|:--|:--|:--|
| `spatial` | diopter scalar → (H,W) 채널로 broadcast → 8ch 입력 | 단순, 효과적 |
| `sinusoidal` | sinusoidal positional encoding → FC concat | NeRF-inspired, 학습 불안정 |
| `coc` | `depth + diopter → CoC map` → 8ch 입력 | 물리 기반, DeepFocus LVF 방식 |

#### CoC 계산 (물리 모델)
```
dp_fl = 1/film_len_dist + dp_focal
CoC = D * |( dp_fl - 1/depth ) / ( dp_fl - dp_focal ) - 1|
     → 픽셀 단위로 normalize (/ film_width / cocScale)
```
- `film_len_dist=0.017m`, `D=0.004m`, `fov=0.7854 rad`, `cocScale=30`
- CoC: 해당 focal distance에서 픽셀별 defocus blur 크기 → diopter conditioning의 물리적 근거

#### LFS vs LVF (원본 DeepFocus 논문)
- **LFS**: 40개 plane 동시 출력 → CoC 불필요 (depth만으로 모든 blur 학습 가능)
- **LVF**: 단일 plane 출력 → 특정 diopter conditioning 필요 → CoC 사용
- 우리 모델은 LVF와 동일한 "단일 diopter 조건부 평가" 구조 → CoC conditioning이 이론적으로 적합

---

## 3. 데이터셋 설계

### `FocalDataset`

| Split | Scene 범위 | Scene 수 |
|:--|:--|:--|
| train | 0 ~ 89 | 90 |
| val | 90 ~ 109 | 20 |
| test | 110 ~ 129 | 20 |

### 레이블
- **Match**: `labels.json`에 사전 계산 (MSE, SSIM, LPIPS)
- **Unmatch**: `labels_unmatch.json`에 사전 계산 (→ `generate_unmatch_labels.py`)
  - 없으면 on-the-fly fallback (느림: ~640ms/샘플)

### 샘플 구성 예
```
train (unmatch_ratio=39): 90 scene × 40 × (1+39) = 144,000 샘플
val   (val_unmatch_ratio=39): 20 scene × 40 × (1+39) = 32,000 샘플
```
> epochs 기준: ratio=39, epochs=5 ≒ ratio=3, epochs=20 (총 노출량 유사)

### 학습 설계 원칙
- Match 샘플: 항상 전수 포함 (고정)
- Unmatch 샘플: 에폭마다 랜덤 리샘플 (다양성 확보)
- Val: **match + full unmatch(39개)** = 진정한 discrimination 능력 평가

### PSNR 정규화
- 모델 출력: sigmoid → [0, 1]
- loss 계산: target_psnr / 100.0 (dB 스케일 맞춤)
- sweep 표시: 모델 출력 × 100 → dB 복원

---

## 4. 성능 병목 및 해결

### 병목 원인
- Unmatch 샘플 계산: GT EXR 로드 (~72ms) + SSIM (~94ms) + **LPIPS AlexNet CPU (~471ms)** = ~637ms/샘플
- DataLoader workers = fork → CUDA 불가 → LPIPS 무조건 CPU
- GPU Util ≈ 0% (GPU 대기, CPU가 병목)

### 해결: `generate_unmatch_labels.py`
- GPU batched LPIPS로 모든 (scene, pred_plane, query_plane) 쌍 사전 계산
- 출력: `data/seed{XXXX}/labels_unmatch.json`
- 완료 후 `__getitem__`에서 파일 읽기만 → 사실상 0ms

```bash
python generate_unmatch_labels.py --data_dir ../varifocal/data
# 예상 소요: ~20분 (GPU 기준)
# 전수: 130 scene × 40 × 39 = 202,800 entry, 총 ~12MB
```

---

## 5. 실행된 실험

### 실험 조건 비교

| Run | Mode | unmatch_ratio | val_unmatch | epochs | num_workers |
|:--|:--|:--|:--|:--|:--|
| `focal_spatial_20260223_025740` | spatial | 3 | 0 (match only) | 20 | 4 |
| `focal_sinusoidal_20260223_030324` | sinusoidal | 3 | 0 (match only) | 20 | 4 |
| `focal_coc_20260224_001919` | coc | **5** | 0 (match only) | 20 | 2 |

> ⚠️ CoC는 unmatch_ratio=5라 다른 두 모드와 **엄밀한 비교 불가**.

### 결과 요약 (seed0110, Fixed @ 2.00D)

| Mode | PSNR 추적 | SSIM 추적 | LPIPS 추적 | 종합 |
|:--|:--|:--|:--|:--|
| **spatial** | ✅ 거의 완벽 | ✅ 거의 완벽 | ✅ 우수 | **최고** |
| **coc** | 🔶 피크 ok, unmatch 스케일 틀림 | 🔶 방향 ok | 🔶 양호 | 중간 |
| **sinusoidal** | ❌ 거의 실패 (flat) | 🔶 방향만 학습 | 🔶 방향만 학습 | 최저 |

### 세부 분석

**Spatial**: 전체적으로 가장 우수한 결과. 3개 scene × 5개 fixed plane 모두 일관되게 real 곡선을 잘 추적함.
- **PSNR**: match 피크 높이, 위치, 좌우 asymmetry(diopter 차이에 따른 비대칭 curve)까지 real과 거의 일치. 특히 mid-range fixed plane(1.0D, 2.0D)에서 거의 완벽한 추적. 극단 fixed plane(0.1D, 4.0D)은 단조감소/단조증가 형태에서도 전 구간에 걸쳐 real 곡선을 잘 따름.
- **SSIM**: PSNR보다 더 안정적으로 real을 추적. 피크 높이(~1.0)와 양쪽 감소 곡선 모두 매우 근사. scene 간 편차도 거의 없음.
- **LPIPS**: 전반적으로 우수하나, match 근방 ~±0.5D 구간에서 real보다 약간 낮게(좋아보이게) 예측하는 경향이 일부 scene에서 관찰됨. 전체 형태와 최솟값 위치는 정확.
- **결론**: diopter를 (H, W) 공간 채널로 broadcast하는 단순 방식이 가장 효과적. 각 픽셀이 diopter 값을 직접 참조할 수 있어 spatial feature와 자연스럽게 통합됨. 정량적 스케일 calibration도 세 mode 중 유일하게 성공.

**CoC**: 지표별로 학습 결과가 크게 다름 (3개 scene, 5개 fixed plane 모두 동일한 패턴).
- **PSNR**: match 포인트에서 날카로운 spike는 있으나, unmatch 구간 전체에서 ~11~12dB의 noise floor로 완전 collapse. real은 22~30dB인데 모델은 "나쁜 건 알지만 얼마나 나쁜지는 모름"에 해당. unmatch 간 상대적 품질 차이를 전혀 구별 못함.
- **SSIM**: match 피크 위치는 정확하고 날카로움. unmatch 구간에서 방향은 맞지만 real 대비 전반적으로 과소평가. 특히 먼 unmatch 구간일수록 차이가 커짐.
- **LPIPS**: 세 지표 중 가장 잘 추적함. 피크 위치, 방향, 전체 형태 모두 reasonable하게 학습됨. 단 좌우 asymmetry 존재 (match 기준 좌측이 더 부정확).
- **결론**: CoC 모델은 "match vs unmatch 판별" 자체는 명확히 학습했으나(피크 spike), PSNR 예측의 정량적 정밀도가 매우 부족. spatial과 비교하면 SSIM/LPIPS는 비슷하거나 오히려 더 날카로운 피크를 보이지만 PSNR에서 크게 밀림. unmatch_ratio 차이(5 vs 3)보다는 CoC conditioning 자체가 PSNR 범위를 calibrate하는 데 부적합할 가능성 있음.

**Sinusoidal**: PSNR은 완전 실패. SSIM/LPIPS는 "정성적 방향"만 부분 학습. FC layer에서만 concat하는 방식이 spatial/visual feature와 통합이 안 된 것으로 분석.

---

## 6. 코드 구조

```
rgbd_focalstack_loss/
├── model.py                    # LossEstimationNet, calCoC(), SinusoidalEncoding
├── dataset_focal.py            # FocalDataset (match/unmatch, coc 지원)
├── train_focal.py              # 학습 루프, run_focal_sweep() 포함
├── run_focal_sweep.py          # 학습 후 standalone sweep 스크립트
├── generate_unmatch_labels.py  # unmatch 레이블 사전 계산 (1회 실행)
├── data/
│   └── seed{XXXX}/
│       ├── pred_frame{XXXX}.exr
│       ├── labels.json         # match 레이블
│       └── labels_unmatch.json # unmatch 레이블 (generate 후 생성)
└── runs/
    └── focal_{mode}_{timestamp}/
        ├── args.json
        ├── best_model.pth
        └── focal_sweep/
```

### 주요 실행 명령어

```bash
# 1. Unmatch 레이블 사전 계산 (최초 1회, ~20분)
python generate_unmatch_labels.py --data_dir ../varifocal/data

# 2. 학습 (spatial baseline, 전수 unmatch)
python train_focal.py \
    --data_dir ../varifocal/data \
    --diopter_mode spatial \
    --unmatch_ratio 39 \
    --val_unmatch_ratio 39 \
    --epochs 5 \
    --batch_size 8 \
    --num_workers 4

# 3. Test Evaluation (학습 완료 모델 정량 평가)
python test_focal.py \
    --run_dir runs/focal_spatial_XXXXXXXX \
    --data_dir ../varifocal/data \
    --unmatch_ratio 39

# 4. Sweep (결과물 시각화)
python run_focal_sweep.py \
    --run_dir runs/focal_spatial_XXXXXXXX \
    --scenes 110 115 120 \
    --data_dir ../varifocal/data
```

### 코드 변경 이력

| 날짜 | 파일 | 변경 내용 |
|:--|:--|:--|
| 2026-02-24 | `train_focal.py` | `--val_unmatch_ratio` argparse 추가 (기본값 39) |
| 2026-02-24 | `train_focal.py` | `run_focal_sweep()` 내부 CoC 8ch 처리 추가 |
| 2026-02-24 | `run_focal_sweep.py` | `sweep_scene()` 내 CoC 8ch 처리 추가 (`model.diopter_mode` 참조) |

---

## 7. 다음 실험 방향

### 즉시 실행
1. `generate_unmatch_labels.py` 실행 (학습 가속 + 올바른 val)
2. **동일 조건** (unmatch_ratio=3, val_unmatch_ratio=39)에서 spatial vs coc 재비교

### 중기
3. Spatial baseline에서 epochs 늘리기 (40~50 epoch)
4. CoC가 논문 contribution이 되려면 **완전 동일 조건**에서 CoC ≥ Spatial 증명 필요
5. Loss weight 조정 (`w_psnr`, `w_ssim`, `w_lpips`): PSNR 학습이 약하면 w_psnr 증가

### 베스트 모델 선택 기준
- `best_model.pth`: val_loss 기준 저장
- **val에 unmatch 포함 필수** → match만 보면 "항상 높은 점수" 예측하는 모델이 best 선택될 위험
- `--val_unmatch_ratio 39` (기본값)로 설정 권장

---

## 8. 알려진 이슈 / 주의사항

| 항목 | 내용 |
|:--|:--|
| RTX 5090 sm_120 경고 | PyTorch sm_120 미지원, PTX로 fallback. 메모리는 정상, 성능 다소 저하 가능 |
| CUDA in DataLoader workers | fork 방식으로 CUDA 재초기화 불가 → LPIPS는 CPU or 사전계산 필요 |
| PSNR 스케일 | 모델 출력: [0,1], 표시: ×100 (dB), loss target: /100 |
| Depth 단위 | `depth / 12.0` → [0,1]로 정규화. CoC 계산 시 ×12 복원 |

---

---

# Loss Estimation (Robust Training)

## 목적

`LossEstimationNet`이 pixel-level adversarial attack에 robust하도록 학습.
일반 학습(`train_focal.py`)은 "clean DeepFocus predictions"에만 노출되어
악의적으로 조작된 이미지에 점수를 높게 예측할 위험이 있음.

Robust training은 per-batch로 adversarial 예측 이미지를 생성하고,
그 adversarial 이미지에 대한 **실제 지표(PSNR/SSIM/LPIPS)**를 정확히 예측하도록 강제함.

---

## 코어 아이디어

```
일반 배치:  pred_clean  → model → predicted_metrics (MSE with real metrics)
Adv 배치:   pred_clean
              → pixel-optimize (maximize model output)
              → pred_adv
              → compute real metrics(pred_adv, GT)
              → model(pred_adv) → predicted_metrics (MSE with real adv metrics)
```

**Key**: adversarial 이미지에 대해 "네가 나쁘다고 생각하는 것처럼 보이지만, 실제로 얼마나 나쁜지를 정확히 예측해라"를 학습.

---

## 구현 (`train_robust.py`)

### `RobustDataset`
- `LossEstimationDataset` 확장
- `(x, diopter, targets, gt)` 반환 — `gt`: GT focal plane (adversarial metric 계산용)

### `generate_adversarial_batch()`
- `x[:, 4:7]` (pred 채널)만 perturbation
- PGD-like pixel optimization (gradient ascent on model output)
- `adv_steps=15`, `adv_lr=0.01`
- `adv_mode`: `'replace'` (pred 채널 교체) or `'augment'` (배치에 추가)

### `compute_real_metrics_batch()`
- adversarial pred vs GT로 실제 PSNR/SSIM/LPIPS 계산
- 이 값을 loss target으로 사용

### `train_epoch_robust()`
- `adv_ratio` 비율로 adversarial 배치 삽입
  - `adv_ratio=0.33` → 배치 3개 중 1개가 adversarial

---

## 실행 명령어

```bash
python train_robust.py \
    --data_dir ../varifocal/data \
    --diopter_mode spatial \
    --adv_ratio 0.33 \
    --adv_steps 15 \
    --adv_lr 0.01 \
    --epochs 20 \
    --batch_size 8
```

---

## 현황 & 다음 방향

| 항목 | 상태 |
|:--|:--|
| 구현 | ✅ 완료 (`train_robust.py`) |
| 학습 실험 | ⬜ 미실행 |
| Focal pipeline과 통합 | ⬜ 미완 (현재 기존 `LossEstimationDataset` 사용) |

### TODO
1. `train_robust.py`를 `FocalDataset` 기반으로 전환 (match/unmatch + diopter conditioning)
2. Focal training 완료 모델(spatial)을 resume해서 robust fine-tuning
3. Robust model과 clean model의 adversarial 공격 취약성 비교

---

## Gradient Alignment Training (신규 제안)

기존 adv training(15-step Adam)을 **1 forward + gradient alignment**로 대체.

### 원리
- `generate_adversarial_batch()`: 15 step Adam → 배치당 ~15× 비용
- **제안**: `initial_plane.grad`를 1 backward로 얻고 `gt - initial` 방향과 cosine alignment

```python
initial_plane.requires_grad_(True)
out = model(torch.cat([rgbd, initial_plane], dim=1), diopter)

# PSNR/SSIM은 높을수록 좋음(+), LPIPS는 낮을수록 좋음(-)
scalar = (out['psnr'] + out['ssim'] - out['lpips']).mean()
scalar.backward(retain_graph=True, create_graph=True)

direction_pred = initial_plane.grad                     # 모델의 "좋아지는 방향"
direction_gt   = (gt_plane - initial_plane).detach()   # 실제 gt 방향

alignment_loss = -F.cosine_similarity(
    direction_pred.flatten(1), direction_gt.flatten(1), dim=1
).mean()
alignment_loss.backward()  # metric_network weights로 흐름
optimizer.step()
```

### Initial Plane 종류
- **strong noise**, **weak noise**, **AIF** 등 — 각 initial의 다양성이 넓은 gradient landscape 교정

### 효과
- metric network가 올바른 gradient landscape를 갖도록 교정
- 이 metric으로 어떤 image를 최적화할 때 gradient descent가 gt 방향으로 수렴 보장
- loss function 사용을 위한 핵심 조건



## [2026-02-26] V2 구조 업그레이드 및 치명적 버그 수정

### 1. PyTorch 브로드캐스팅(Broadcasting) 버그 수정
- **증상:** 기존 코드에서 	rain_focal.py의 Loss 계산 시 예측값은 (Batch,), 타겟(정답)은 (Batch, 1) 형태여서, L1Loss 연산 중 PyTorch가 두 텐서의 크기를 억지로 맞추기 위해 (Batch, Batch) 형태(예: 8x8=64개)로 브로드캐스팅 연산을 강제로 분배해 버리는 심각한 버그가 있었습니다.
- **영향:** 정답 1개(Match)에 모델이 집중하지 못하고 배치 내의 다른 이미지 정답들과도 오차가 섞여 들어가서 Match Peak가 뾰족하게 학습되지 못하고 평평하게(Average) 뭉개졌습니다.
- **해결:** model.py 출력부를 .squeeze() 없이 (Batch, 1) 상태로 내보내도록 수정하고, 타겟 또한 무조건 .view(-1, 1)로 명시하여 1:1 요소별(Element-wise) 오차가 완벽히 계산되도록 규격화했습니다.

### 2. --version 아키텍처 도입 (v1 vs v2)
Match 거리에서의 예측값을 다른 평범한 구간과 뚜렷하게 구별해내는 뾰족한 Peak 특성을 살리기 위해 모델 구조와 Loss를 이원화(v1, v2)하여 실험할 수 있도록 기능을 추가했습니다.

**v1 버넷 (기존 레거시)**
- LinearLayer(1개) 기반의 얕은 헤드 사용
- L1Loss(MAE) 사용 → Outlier(정답 Peak) 오차에 대한 페널티가 상대적으로 둔감함.

**v2 버넷 (신규 기본값)**
- **MLP 헤드 딥러닝:** 예측 헤드 3개(PSNR, SSIM, LPIPS)를 모두 2-Layer MLP (Linear(128) -> LeakyReLU -> Linear(1))로 업그레이드하여, 비선형적인 뾰족한 Peak 함수를 자유자재로 모사할 수 있는 네트워크 수용력(Capacity)을 대폭 넓혔습니다.
- **MSELoss 도입:** L1Loss 대신 오차를 제곱하는 MSELoss로 변경하여, 다른 평평한 부분은 대충 맞추더라도 뾰족하게 솟은 정답 Peak 구간(Outlier)에서 추론을 살짝이라도 틀릴 경우 어마어마한 제곱 페널티 폭탄을 맞도록 유도하여 모델이 자발적으로 Peak를 정확하게 찍어내도록 훈련 방식을 극한으로 강제했습니다.

---

## [2026-03-05] Loss Weights 및 Diopter Mode 비교 결과

### 1. Spatial Mode Loss Weights 튜닝
- **테스트 가중치:** `1 : 1 : 1` vs `1 : 0.5 : 0.5` (PSNR : SSIM : LPIPS)
- **결과:** **1 : 1 : 1** 가중치로 학습했을 때, Real 지표 대비 예측 MSE(Mean Squared Error)가 가장 낮아 최적의 설정으로 확인됨.

### 2. Spatial vs CoC Mode 성능 평가 (최적 조건 `1:1:1` 적용 시)
Real 측정치와 예측치 간의 오차(MSE)를 분석한 결과.

| Mode | PSNR 오차 (MSE) | SSIM 오차 (MSE) | LPIPS 오차 (MSE) |
|:--|:--|:--|:--|
| **Spatial** | 0.51 | 0.012 | 0.0145 |
| **CoC** | **0.38** | **0.007** | **0.009** |

- **요약:** CoC 모드(물리적 디포커스 반영)가 모든 평가지표에서 Spatial 방식보다 오차가 적었으며, 실측 지표를 더욱 정확하게 학습해내는 것을 확인함.

---

## [2026-03-05] Score-based Generative Model 파이프라인 구축 (진행형)

### 1. 목적 (패러다임 전환)
기존의 **품질 예측 회귀 모델 (Regression)** 훈련 방식을 완전히 뒤엎고, 주어진 3D Scene의 공간 정보(RGBD)와 특정 Diopter 값을 바탕으로 해당 초점 평면(Focal Plane)의 최적 이미지를 무에서 유로 직접 그려내는 **제너러티브(생성형) 모델링 방식**으로 접근을 변경했습니다.

### 2. 학습 핵심 사항: Gradient Matching (Score Matching)
- **에너지 함수 (Energy Function)**: 뉴럴넷 모델(`model_score.py`) 구조를 단일 스칼라 점수(Energy)를 뿜어내도록 개조했습니다.
- **Score (방향성)**: 입력 이미지를 넣었을 때 모델이 도출해낸 에너지 값에 대해 `torch.autograd.grad(create_graph=True)`를 수행하여, "이 이미지가 지금 여기서 얼만큼, 어디로 수정되어야 하는가?"라는 픽셀 단위의 Score(Gradient 방향 모멘텀) 값을 획득합니다.
- **Training**: 노이즈 이미지(`current_image`)에서 시작해서 모델한테 길(Score)을 물어보고, 그 길이 **실제 정답(GT Focal Plane)으로 가는 직선 방향(gt - current_image)** 과 똑같아지도록 MSELoss를 걸고 훈련시킵니다.
- **Trajectory Learning**: 한 에폭당 단방향으로 끝나는 게 아니라, 내부적으로 `gm_steps` (예: 50번) 횟수만큼 스텝을 끊어가면서 **계속 길을 물어보고 -> 모델의 지시에 따라 이동하고 -> 다시 그 위치에서 길을 물어보는 궤적(Trajectory)** 을 통째로 학습(+기울기 누적) 시킵니다.

### 3. 주요 구현 모듈
- **[Modify] `dataset_focal.py`**: 정답을 향해 궤적을 수정해 나가야 하므로, 기존 Metrics 라벨 뿐만 아니라 `return_gt=True` 옵션을 통해 무조건 **Original GT Focal Plane 이미지 텐서**를 반환하도록 구조를 추가했습니다. 프로토타이핑을 위해 `single_scene_only=True` 옵션도 추가되었습니다.
- **[New] `train_score.py`**: 상기 Gradient Matching의 궤적 학습 알고리즘을 담당하는 최고 핵심 학습 스크립트입니다. 랜더 팜(서버) 환경을 고려하여 짜여졌습니다.
- **[New] `test_score.py`**: 학습이 무사히 종료되었다면, 완전한 랜덤 노이즈 텐서를 던져주면 모델이 스스로 알아서 방향(Score)을 지시하며 N번의 스텝(`gm_steps`)만에 깨끗하고 정확한 목적지(Focal Plane) 이미지를 렌더링/복원해내는 과정을 담은 Inference 데모 스크립트입니다. 생성된 궤적 히스토리는 이미지 형태로 저장됩니다.
