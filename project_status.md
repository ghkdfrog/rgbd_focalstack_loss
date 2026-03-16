# 🔬 Gradient Matching 프로젝트 현황 종합 정리

## 프로젝트 개요

**Energy-Based Model (EBM)** 을 사용한 **Gradient Matching** 방식으로 varifocal/depth-of-field focal stack 이미지를 생성하는 프로젝트. RGBD 입력(4ch) + 생성 중인 이미지(3ch) + CoC/diopter 조건(1ch)을 받아 에너지 스칼라를 출력하고, 그 gradient를 따라 Langevin dynamics로 이미지를 반복 생성.

---

## 코드 구조 (`gm/` 패키지)

| 파일 | 역할 |
|:---|:---|
| [config.py](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/config.py) | `argparse` 기반 학습/추론 하이퍼파라미터 정의 |
| [model.py](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/model.py) | [SimpleCNN](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/model.py#17-66) (5-layer), [SimpleCNNDeep](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/model.py#68-129) (10-layer) 에너지 모델 정의 |
| [train.py](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/train.py) | 학습 루프 + validation + PSNR 측정 + 학습 후 자동 추론 |
| [infer.py](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/infer.py) | 체크포인트에서 focal plane 생성 + PSNR 수렴 CSV/그래프 저장 |
| [reevaluate_single_scene.py](file:///d:/Deepfocus/rgbd_focalstack_loss/scripts/reevaluate_single_scene.py) | 기존 run을 재평가(best model 재선정) + 자동 추론 |

### 모델 아키텍처

- **[SimpleCNN](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/model.py#17-66)**: 5-layer Conv (16→32→64→128→256), stride=1, ReLU, 해상도 유지
- **[SimpleCNNDeep](file:///d:/Deepfocus/rgbd_focalstack_loss/gm/model.py#68-129)**: 10-layer Conv (위 + 256×5 추가 레이어)
- **Energy Head**: `fc` (Linear 67M params, 512×512 전용) vs `conv1x1` (해상도 무관, 파라미터 극소)
- **Diopter 조건부**: `spatial` (스칼라 map) vs `coc` (CoC map 입력에 포함)

### 학습 메커니즘

1. 랜덤 노이즈에서 시작 → `gm_steps`번 반복
2. 매 스텝: `pred_grad = ∂E/∂x`, `loss = MSE(pred_grad, GT-x)`, `x ← x + η·pred_grad`
3. η 스케줄: `constant` / `cosine` / `linear` 감쇠 지원
4. 선택적 Langevin noise: `√(2η)·z` 추가

---

## 실험 결과 요약 (23개 run)

### 🏆 Single-Scene Prototype Runs (scene 0 오버피팅)

> [!IMPORTANT]
> **Best PSNR: ~27.86 dB** — **ResNet**, `linear` eta 스케줄, 50 GM steps (32 epochs 진행 중)

| Run | 설정 | Best Epoch | Best PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0314_160511` | coc, **resnet**, **linear**, 100ep, 50steps | **29 ep** | **27.86 dB** (26.6/29.9/27.1) | 🥇 새로운 최고 성능 (기존 대비 약 +2.5dB 향상). 현재 32ep까지 진행됨 |
| `0313_133317` | coc, **stride**, fc, **linear**, 100ep, 50steps | **13 ep** | **19.51 dB** (19.5/19.7/19.4) | latest(100ep) 19.74 dB. stride 구조 변경으로 인한 성능 하락(25.4 → 19.5dB) |
| `0310_011629` | coc, fc, **linear**, 100ep, 50steps | **100 ep** | **25.40 dB** (24.2/26.1/25.8) | 🥈 이전 최고 성능. 100ep까지 하락 없이 계속 상승 (plateau) |
| `0310_011608` | coc, fc, **cosine**, 100ep, 50steps | **100 ep** | **25.27 dB** (24.4/26.1/25.2) | 🥉 cosine도 100ep까지 하락 없이 안정적 수렴 |
| `0309_084426` | coc, fc, constant, 100ep, 100steps | **latest** | **25.13 dB** (24.3/26.9/24.2) | p20 단독 최고(26.9). 특정 에폭에 피크 |
| `0309_175036` | coc, fc, constant, 100ep, 50steps | **39 ep** | **24.91 dB** (24.0/25.9/24.8) | 39ep에 피크 달성 후 100ep까지 **2dB 폭락** (오버피팅) |
| `0311_191411` | coc, fc, constant, **deep**(10L), 50ep | **40 ep** | **22.95 dB** (21.5/23.2/24.2) | 40ep에서 피크 후 50ep까지 유지됨 (plateau 도달) |
| `0312_120940` | coc, fc, constant, **deep**(10L), 100ep| **53 ep** | **23.04 dB** (22.5/23.0/23.6) | `191411`에서 resume 진행. 100ep 완료. 53ep 이후 정체/소폭 하락 (수렴) |
| `0313_041643` | coc, fc, **linear**, **deep**(10L), 50ep| **26 ep** | **23.75 dB** (23.0/23.8/24.4) | `120804` (중단됨)에서 resume. 50ep 완료. Deep 구조에서도 linear가 constant보다 성능이 높음 |
| `0308_212501` | coc, fc, constant, 50ep, 100steps | **latest** | **22.69 dB** (22.4/23.7/22.0) | 50ep에서 점진적 상승 중 종료 |
| `0307_232125` | **spatial**, fc, constant, 100ep, 50steps | **latest** | **18.59 dB** (18.7/19.4/17.6) | spatial 모드는 성능 불리 |
| `0309_174409` | coc, **conv1x1**, constant, 100ep | **best** | **16.39 dB** → latest **붕괴** | conv1x1은 불안정(latest -2dB) |

### 🌐 Multi-Scene Runs (전체 데이터셋)

| Run | 설정 | Best Epoch | PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0310_080631` | coc, conv1x1, cosine, 100ep, 5장 | **latest** | **20.80 dB** (20.2/21.5/20.7) | conv1x1이 multi에서 안정적 |
| `0310_075915` | coc, fc, cosine, 100ep, 5장 | **latest** | **19.88 dB** (19.2/20.7/19.7) | fc가 multi에서는 conv1x1보다 낮음 |
| `0311_142349` | coc, fc, constant, **5ep**, 전체 | **best_psnr** | **21.46 dB** (20.7/21.8/21.9) | 전체(90) scene 학습 (5 epoch). Best PSNR 기준 |
| `0312_134650` | coc, fc, **linear**, 100ep, 5장 | **14 ep** | **20.97 dB** (21.0/21.1/20.8) | 100ep 완료. 초반 피크(14ep) 후 성능 하락. Multi-scene에서 linear 감쇠가 불안정할 수 있음 |



---

## 핵심 인사이트 (CSV_ANALYSIS.md 포함)

### 1. η 스케줄의 효과
- `linear`/`cosine` 감쇠가 `constant`보다 **single-scene에서 2~3 dB 유리** (25.4 vs 22.7 dB)
- 단, 추론 스텝을 학습 스텝보다 과도하게 늘리면(200steps) **cosine/linear에서 1~1.5 dB 하락**

### 2. 최적 추론 스텝 수
- 학습 스텝의 **절반~동일(30~80스텝)** 부근에서 Peak PSNR 달성
- 200스텝으로 과도하게 늘리면 PSNR이 유지되거나 오히려 하락 (over-Langevin 현상)

### 3. 아키텍처 비교
- **ResNet**: **최고 성능 (27.86 dB) 달성**. 기존 SimpleCNN 대비 압도적인 성능 향상(+2.5dB).
- **SimpleCNNDeep (10L)**: single-scene에서 22.95 dB로 SimpleCNN(5L) 대비 낮음. 50ep 부족할 수 있음
- **conv1x1 head**: single-scene에서 불안정하지만, multi-scene에서는 fc보다 오히려 안정적

### 4. Diopter 조건부 모드
- **CoC 모드가 Spatial 모드보다 압도적 우위** (25 dB vs 18.6 dB)
- Spatial 모드는 plane별 PSNR 편차도 큼 (p0=16, p39=10)

### 5. 현재 미해결 과제
- Multi-scene 학습이 아직 **충분한 에폭으로 돌리지 않음** (5ep only)
- Deep 아키텍처도 50ep single-scene만 시도 → 더 긴 학습 필요
- Langevin noise 적용 후 스케일 문제: `noise=True`로 실험(`0312_120655`)을 진행하였으나 약 20 epoch 후 **취소됨**. `step_size`(`eta`)에 비해 노이즈 크기(`sqrt(2*eta)`)가 과도하게 커서 노이즈가 지배적이 되어 의미 없는 Loss/PSNR이 도출되는 문제 발견. 보정 필요.
