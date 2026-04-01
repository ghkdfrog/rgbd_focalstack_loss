# Sharp Prior 실험 계획서

이 문서는 새롭게 구현된 Sharp Prior 기능을 테스트하기 위한 핵심 실험 4가지를 정리한 것입니다. 기존 Best(FC head) 결과와 비교하여 `conv1x1` 헤드에서의 성능과 Sharp Prior의 유효성을 검증합니다.

---

## 실험 대원칙
- **공통 기반 설정**: `resnet_film`, `coc_abs`, `channels 128`, `silu`, `linear eta`, `single_scene_only`, `200 epochs`
- **변수**: `energy_head`, `sharp_prior_method`, `sharp_lambda_mode`/`sharp_gamma_mode`

---

## 핵심 실험 리스트

### 1단계: conv1x1 베이스라인 확보
> **목적**: 기존 FC 헤드 결과와 비교하여, 헤드 변경 자체가 성능(PSNR)에 미치는 영향을 먼저 확인합니다.

- **설명**: Sharp Prior 없이 `conv1x1` 헤드만 사용하여 학습합니다.
- **실행 명령어**:
```bash
python -m gm.train --arch resnet_film --diopter_mode coc_abs --energy_head conv1x1 --channels 128 --activation silu --eta_schedule linear --single_scene_only --epochs 200
```

### 2단계: Penalty 방식 (Option A) 검증
> **목적**: 전체 에너지에 패널티를 더하는 기존 방식(Option A)이 초점 영역 복원에 주는 효과를 측정합니다.

- **설명**: `gamma=100` 고정, `lambda` 학습 가능(learnable) 모드입니다. (기본 설정 활용)
- **실행 명령어**:
```bash
python -m gm.train --arch resnet_film --diopter_mode coc_abs --energy_head conv1x1 --channels 128 --activation silu --eta_schedule linear --single_scene_only --epochs 200 --sharp_prior
```

### 3단계: Energy Density 방식 (Option B') 검증
> **목적**: 픽셀별 에너지 밀도를 가중치로 억제하는 방식(Option B')이 Penalty 방식보다 우수한지 비교합니다.

- **설명**: 초점 영역의 NN 기여도를 직접 줄여 Prior-NN 간의 경쟁을 최소화합니다.
- **실행 명령어**:
```bash
python -m gm.train --arch resnet_film --diopter_mode coc_abs --energy_head conv1x1 --channels 128 --activation silu --eta_schedule linear --single_scene_only --epochs 200 --sharp_prior --sharp_prior_method energy_density
```

### 4단계: Lambda 고정(Fixed) 효과 확인
> **목적**: Lambda를 학습시키는 것보다 고정된 강도를 사용하는 것이 학습 안정성에 유리한지 확인합니다.

- **설명**: 2단계(Penalty) 설정에서 `lambda`까지 고정합니다. (3단계 기반으로 테스트해도 무방)
- **실행 명령어**:
```bash
python -m gm.train --arch resnet_film --diopter_mode coc_abs --energy_head conv1x1 --channels 128 --activation silu --eta_schedule linear --single_scene_only --epochs 200 --sharp_prior --sharp_lambda_mode fixed
```

---

## 권장 실험 순서 및 이유

1.  **1단계 (Baseline)**: 모든 비교의 기준점이 필요합니다.
2.  **2단계 (Penalty)**: 논리적으로 가장 단순한 Prior 추가 효과를 확인합니다.
3.  **3단계 (Energy Density)**: 2단계 결과가 나온 뒤, "경쟁 억제" 로직이 실제 품질 향상으로 이어지는지 확인합니다.
4.  **4단계 (Fixed Lambda)**: 앞선 실험들 중 가장 좋았던 Method를 골라 Lambda 고정 여부를 최종 결정합니다.

> [!TIP]
> 각 실험이 끝날 때마다 `generation_check` 이미지를 통해 **초점 영역(Plane 20 등)의 선명도**가 Baseline 대비 얼마나 살아나는지 시각적으로 먼저 비교하는 것이 중요합니다.
