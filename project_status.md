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

## 실험 결과 요약

### 🏆 Single-Scene Prototype Runs (scene 0 오버피팅)

> [!IMPORTANT]
> **Best PSNR: ~27.86 dB** — **ResNet** 256ch, `coc`, `linear` eta 스케줄, 50 GM steps (100 epochs)

| Run | 설정 | Best Epoch | Best PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0314_160511` | coc, **resnet**, **linear**, 100ep, 50steps | **best** | **27.86 dB** (26.6/29.9/27.1) | 최고 성능. 100ep까지 학습 완료됨. |
| `0313_133317` | coc, **stride**, fc, **linear**, 100ep, 50steps | **15 ep** | **19.89 dB** (19.8/19.8/20.0) | stride 구조 변경으로 인한 성능 하락(25.4 → 19.8dB) |
| `0310_011629` | coc, fc, **linear**, 100ep, 50steps | **100 ep** | **25.36 dB** (24.2/26.0/25.8) | 이전 최고 성능. 100ep까지 하락 없이 계속 상승 (plateau) |
| `0310_011608` | coc, fc, **cosine**, 100ep, 50steps | **100 ep** | **25.25 dB** (24.4/26.1/25.3) | cosine도 100ep까지 하락 없이 안정적 수렴 |
| `0309_084426` | coc, fc, constant, 100ep, 100steps | **-** | **평가 로그 없음** | p20 단독 최고(26.9). 특정 에폭에 피크 |
| `0309_175036` | coc, fc, constant, 100ep, 50steps | **39 ep** | **24.91 dB** (24.0/25.9/24.8) | 39ep에 피크 달성 후 100ep까지 **2dB 폭락** (오버피팅) |
| `0311_191411` | coc, fc, constant, **deep**(10L), 50ep | **50 ep** | **22.83 dB** (21.3/23.1/24.1) | 50ep에도 계속 상승 |
| `0312_120940` | coc, fc, constant, **deep**(10L), 100ep| **53 ep** | **23.04 dB** (22.5/23.0/23.6) | `191411`에서 resume 진행. 100ep 완료. 수렴 확인 |
| `0313_041643` | coc, fc, **linear**, **deep**(10L), 50ep| **26 ep** | **23.75 dB** (23.0/23.8/24.4) | `120804` (중단됨)에서 resume. Deep 구조에서도 linear가 우수 |
| `0308_212501` | coc, fc, constant, 50ep, 100steps | **50 ep** | **22.69 dB** (22.4/23.7/22.0) | 50ep에서 점진적 상승 중 종료 |
| `0307_232125` | **spatial**, fc, constant, 100ep, 50steps | **6 ep** | **20.07 dB** (19.7/20.8/19.7) | spatial 모드는 성능 불리 |
| `0309_174409` | coc, **conv1x1**, constant, 100ep | **4 ep** | **16.27 dB** (15.7/16.9/16.2) | conv1x1은 불안정 (초반 피크 후 붕괴) |

### 🧪 New Architecture Single-Scene Runs

> [!IMPORTANT]
> **Best PSNR: ~32.22 dB** — **ResNet(FiLM)** 128ch, `coc_abs`, `linear` eta 스케줄, 100 epochs (`0323_203943`)

| Run | 설정 | Best Epoch | Best PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0323_203943` | **coc_abs**, **film_resnet** 128ch, **SiLU**,  100ep | **best** | **32.22 dB** (28.9/33.0/34.8) | 🥇 **전체 최고 성능 갱신**. SiLU 활성화 |
| `0319_150917` | coc, **film_resnet** 128ch, fc, **linear**, 100ep | **100ep** | **31.20 dB** (27.7/30.6/32.1) | 🥈 100ep 완료 후 기존 예측치 향상, 안정적인 고성능 |
| `0318_130552` | **coc_signed**, **film_resnet** 128ch, fc, **linear**, 100ep | **100ep** | **29.79 dB** (27.4/30.5/31.5) | 🥉 `coc_signed` 조합이 p39에서 우수 |
| `0323_124739` | **coc_abs**, **resnet_film** 128ch, fc, **linear**, 100ep | **best** | **29.45 dB** (27.1/30.1/31.1) | `resnet_film` 특화 클래스 실험, 고성능 기록 |
| `0320_063400` | coc, **resunet** 64ch, fc, **linear**, 100ep | **best** | **27.72 dB** (26.3/29.2/27.7) | 100ep 완료. ResUNet 64ch + fc. 준수한 성능 |
| `0318_112238` | coc, **resnet** 128ch, fc, **linear**, 50ep | **19 ep** | **27.59 dB** (26.1/29.6/27.1) | 일반 ResNet 128ch. |
| `0319_151904` | **coc_abs**, **convnext_unet** 128ch, **conv1x1**, **linear**, 100ep | **best** | **27.53 dB** (26.4/28.5/27.7) | 100ep 완료. 1.3dB 상승 |
| `0319_150817` | **coc_signed**, **resnet** 128ch, fc, **linear**, 100ep | **best** | **27.47 dB** (25.9/29.4/27.1) | FiLM 뺀 대조군 실험 |
| `0317_034204` | coc, **convnext_unet**, **conv1x1**, **linear**, 100ep | **best** | **25.82 dB** (25.4/26.5/25.5) | ConvNeXtUNet+conv1x1. plane별 편차 작음 |
| `0317_033217` | coc, **convnext_unet**, fc, **linear**, 100ep | **best** | **25.11 dB** (24.7/25.9/24.8) | ConvNeXtUNet+fc. conv1x1보다 약간 낮음 |
| `0318_202001` | coc, **resnet** 128ch, **conv1x1**, **linear**, 50ep | **8 ep** | **24.96 dB** (24.8/25.4/24.7) | ResNet 128ch+conv1x1. 안정적 |
| `0321_125237` | **coc_abs**, **interleave_resnet**(FiLM) 256ch, fc, **linear**, 100ep | **best** | **24.76 dB** (23.6/24.5/26.1) | InterleaveResNet 256ch + FiLM. |
| `0323_042615` | **coc_abs**, **film_resnet** 128ch, **SiLU**, **linear**, 100ep | **best** | **24.44 dB** (24.3/24.9/24.0) | SiLU 적용. 준수한 성능 |
| `0320_055937` | coc, **resunet** 64ch, **conv1x1**, **linear**, 100ep | **best** | **23.71 dB** (23.4/24.1/23.7) | 100ep 완료. conv1x1 사용 |
| `0324_180506` | **coc_abs**, **film_resnet** 128ch, **Sharp(1.0)**, 50ep | **best** | **23.66 dB** (25.7/24.1/21.1) | Sharpness Prior(1.0) 적용 |
| `0318_132413` | **coc_abs**, **dilated**, **conv1x1**, **linear**, 50ep | **50 ep** | **22.40 dB** (21.3/23.7/22.2) | DilatedNet+coc_abs. |
| `0323_042653` | **coc_abs**, **film_resnet** 128ch, fc, **linear**, 100ep | **best** | **18.84 dB** (23.9/21.1/11.5) | 매우 낮은 수렴 성능 |
| `0323_203707` | **coc_abs**, **film_resnet** 128ch, fc, **linear**, 100ep | **best** | **18.44 dB** (18.6/21.9/14.9) | 매우 낮은 수렴 성능 |

### 🌐 Multi-Scene Runs (전체 데이터셋)

| Run | 설정 | Best Epoch | PSNR (avg p0/p20/p39) | 비고 |
|:---|:---|:---:|---:|:---|
| `0311_142349` | coc, fc, constant, **5ep**, 전체 | **best** | **21.46 dB** (20.7/21.8/21.9) | 전체 scene 학습. Best PSNR |
| `0312_134650` | coc, fc, **linear**, 100ep, 5장 | **14 ep** | **20.97 dB** (21.0/21.1/20.8) | 14ep 피크 후 하락. linear 감쇠 불안정 가능성 |
| `0310_080631` | coc, conv1x1, cosine, 100ep, 5장 | **-** | **-** | 평가 로그 없음 |
| `0310_075915` | coc, fc, cosine, 100ep, 5장 | **-** | **-** | 평가 로그 없음 |
| `0316_083358` | coc, **resnet**, fc, **linear**, 5ep, 전체 | **-** | **-** | 전체 scene ResNet 시도. 평가 로그 없음 |
| `0323_152249` | **coc_abs**, **film_resnet** 128ch, fc, **linear**, 5ep, 다중 | **-** | **-** | 평가 로그 없음 |
| `0324_100203` | **coc_abs**, **film_resnet** 128ch, fc, **linear**, 5ep, 다중 | **-** | **-** | 평가 로그 없음 |

---

## 핵심 인사이트

### 1. η 스케줄의 효과
- `linear`/`cosine` 감쇠가 `constant`보다 **single-scene에서 2~3 dB 유리**.


### 3. Diopter 조건부 모드
- 기존 `coc_signed > coc > coc_abs > spatial` 가설에 변화 발생. `coc_abs`도 FiLM 레이어를 통해 적절히 스케일링/시프트 되면 충분한 성능을 낼 수 있음.

### 4. 현재 미해결 과제
- 아키텍처 동일 설정 하에서도 수렴의 편차가 크므로 안정을 위한 방안 필요.
- Multi-scene 학습을 긴 에폭으로 돌려 객관적인 일반화 능력 검증 필요.

### 🗑️ 미완료 / 중단된 Runs
폴더는 생성되었으나 바로 실패하거나 결과 기록이 없는 목록들:
- `0306_221916`, `0306_222752`, `0307_094134`, `0307_094147`, `0307_172330`, `0308_164208`, `0308_212402`, `0309_084328`, `0309_175428` (`coc`, constant)
- `0307_172316`, `0308_164220` (`spatial`, constant)
- `0312_120655`, `0312_120804` (`coc`, linear)
- `0321_130849` (`coc_abs`, interleave_resnet(FiLM) 512ch, linear)

---

## 🚀 다음 단계 (Future Plans)

### 후보 1. ConvNeXt 블록 (진행사항 체크)
* 기존 ConvNeXtUNet 결과 비교 계속 (현재 `27.5dB` 수준으로 준수한 상태)

### 후보 2. Sharpness Prior 최적화 및 안정성 확보
* `0323_203943` (32.22 dB)에서 입증된 **Sharpness Prior (lambda=10.0)** 와 **SiLU** 의 효과를 `coc_signed` 등 다른 조건에서도 검증.

### 후보 3. SE-ResNet 블록 (채널 어텐션)
* 피처 맵(채널) 별로 가중치를 부여하여 CoC 조건에 따라 블러/선명 채널 조절.
