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

## 실험 결과 요약 (30개 run)

### 🏆 Single-Scene Prototype Runs (scene 0 오버피팅)

> [!IMPORTANT]
> **Best PSNR: ~29.71 dB** — **FiLMResNet** 128ch, `coc_signed`, `linear` eta 스케줄, 50 GM steps (100 epochs)

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

### 🧪 New Architecture Single-Scene Runs

| Run | 설정 | Best Epoch | Best PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0319_150917` | coc, **film_resnet** 128ch, fc, **linear**, 100ep | **학습중** | **30.09 dB** (27.5/30.7/32.1) | 🥇 **전체 최고 성능 갱신**. 중간 체크포인트임에도 기존 (29.71)을 크게 상회 |
| `0318_130552` | **coc_signed**, **film_resnet** 128ch, fc, **linear**, 100ep | **best_psnr** | **29.71 dB** (27.3/30.5/31.3) | 🥈 이전 최고 성능. FiLM+coc_signed 조합이 p39에서 31.3dB 달성 |
| `0318_112238` | coc, **resnet** 128ch, fc, **linear**, 50ep | **best_psnr(19ep)** | **27.59 dB** (26.1/29.6/27.1) | ResNet 128ch. 256ch(0314_160511, 27.86)과 유사한 성능 |
| `0317_034204` | coc, **convnext_unet**, **conv1x1**, **linear**, 100ep | **best** | **25.82 dB** (25.4/26.5/25.5) | ConvNeXtUNet+conv1x1. plane별 편차 작음 |
| `0317_033217` | coc, **convnext_unet**, fc, **linear**, 100ep | **best** | **25.12 dB** (24.7/25.9/24.8) | ConvNeXtUNet+fc. conv1x1보다 약간 낮음 |
| `0318_202001` | coc, **resnet** 128ch, **conv1x1**, **linear**, 50ep | **best_psnr(8ep)** | **24.96 dB** (24.8/25.4/24.7) | ResNet 128ch+conv1x1. fc(27.59)보다 낮지만 안정적 |
| `0318_132413` | **coc_abs**, **dilated**, **conv1x1**, **linear**, 50ep | **best_psnr(50ep)** | **22.40 dB** (21.3/23.7/22.2) | DilatedNet+coc_abs. 새 diopter 모드 실험 |
| `0319_150817` | **coc_signed**, **resnet** 128ch, fc, **linear**, 100ep | **학습중** | **27.47 dB** (25.9/29.4/27.1) | 중간 체크포인트 추론. 최고 기록(29.71)의 구조(FiLM) 변형 대조군으로 보임 |
| `0319_151904` | **coc_abs**, **convnext_unet** 128ch, **conv1x1**, **linear**, 100ep | **학습중** | **26.24 dB** (25.6/27.1/26.0) | 중간 체크포인트 추론. conv1x1+coc_abs 조합의 한계점 확인 |

### 🌐 Multi-Scene Runs (전체 데이터셋)

| Run | 설정 | Best Epoch | PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0310_080631` | coc, conv1x1, cosine, 100ep, 5장 | **latest** | **20.80 dB** (20.2/21.5/20.7) | conv1x1이 multi에서 안정적 |
| `0310_075915` | coc, fc, cosine, 100ep, 5장 | **latest** | **19.88 dB** (19.2/20.7/19.7) | fc가 multi에서는 conv1x1보다 낮음 |
| `0311_142349` | coc, fc, constant, **5ep**, 전체 | **best_psnr** | **21.46 dB** (20.7/21.8/21.9) | 전체(90) scene 학습 (5 epoch). Best PSNR 기준 |
| `0312_134650` | coc, fc, **linear**, 100ep, 5장 | **14 ep** | **20.97 dB** (21.0/21.1/20.8) | 100ep 완료. 초반 피크(14ep) 후 성능 하락. Multi-scene에서 linear 감쇠가 불안정할 수 있음 |
| `0316_083358` | coc, **resnet**, fc, **linear**, 5ep, 전체 | — | — | 전체 scene ResNet 학습 시도. 5ep만 진행, 추론 결과 없음 (중단/미완료) |

---

## 핵심 인사이트

### 1. η 스케줄의 효과
- `linear`/`cosine` 감쇠가 `constant`보다 **single-scene에서 2~3 dB 유리**

### 2. 아키텍처 비교
- **FiLMResNet(coc_signed) 128ch**: **전체 최고 성능 29.71 dB**. FiLM conditioning + 부호 있는 CoC의 시너지
- **ResNet 256ch**: 27.86 dB. 128ch(27.59)와 유사 → 채널 수보다 구조/conditioning이 중요
- **ConvNeXtUNet**: conv1x1(25.82) > fc(25.12). plane별 편차가 매우 작아 안정적
- **DilatedNet + coc_abs**: 22.40 dB. 절대값 CoC로는 방향 정보 부족

### 3. Diopter 조건부 모드
- **coc_signed > coc > coc_abs > spatial** 순서로 성능 우위
- coc_signed는 "초점 앞/뒤" 방향 정보를 명시적으로 제공

### 4. FiLM Conditioning
- FiLMResNet(coc_signed) 128ch: **29.71 dB** — ResNet 256ch(27.86) 대비 +1.85 dB
- 적은 파라미터(128ch)로도 FiLM + coc_signed 조합이 효과적

### 5. 현재 미해결 과제
- Multi-scene 학습이 아직 **충분한 에폭으로 돌리지 않음** (5ep only)
- ConvNeXtUNet + FiLM 조합은 아직 미실험
- Langevin noise 스케일 문제 미해결

### 🗑️ 미완료 / 로그 없음 (Unrecorded & Incomplete Runs)
폴더는 생성되었으나, `log.txt` 등 학습 결과가 기록되지 않아 중단되거나 바로 실패한 run 목록입니다.
- `0306_221916`, `0306_222752`, `0307_094134`, `0307_094147`, `0307_172330`, `0308_164208`, `0308_212402`, `0309_084328`, `0309_175428` (`coc`, constant)
- `0307_172316`, `0308_164220` (`spatial`, constant)
- `0312_120655`, `0312_120804` (`coc`, base linear 설정 시도, `120940` 등으로 재시작됨)

---

## 🚀 다음 단계 (Future Plans)

### 후보 1. ConvNeXt 블록 (우선 적용 대상)
* **논문**: *A ConvNet for the 2020s (CVPR 2022)*
* **특징**: `7x7 Depthwise Conv` 와 `Pointwise Conv` 구조를 채택하여, 해상도를 유지하면서도 기존 ResNet 블록보다 **훨씬 넓은 Receptive Field(수용 영역) 확보**.

### 후보 2. Dilated Residual 블록 (팽창 합성곱)
* **논문**: *Multi-Scale Context Aggregation by Dilated Convolutions (ICLR 2016)*
* **특징**: 커널 사이에 빈 공간을 두어 연산량과 파라미터 증가 없이 Receptive Field만 극대화.

### 후보 3. SE-ResNet 블록 (채널 어텐션)
* **논문**: *Squeeze-and-Excitation Networks (CVPR 2018)*
* **특징**: 피처 맵(채널) 별로 가중치를 부여하여 CoC 조건에 따라 블러/선명 채널 조절.
