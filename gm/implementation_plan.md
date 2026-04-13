# 물리적 렌더링을 모사하는 다중 헤드 조합형 에너지 모델 (Physically-aware Compositional EBM) 구현 계획

이 계획은 기존 단일 에너지 기반 그래디언트 매칭 모델을 개편하여 구조(Structure), 지각(Perception), 물리(Physics) 3가지의 독립된 에너지 지형을 학습하고, 추론 단계에서 이들의 그래디언트를 조합하여 최종 이미지를 생성하는 모델 구조 및 학습/추론 루프 수정 방안을 담고 있습니다.

## Memory Management & OOM Mitigation Strategy (Crucial)

> [!CAUTION]  
> 3개의 에너지를 각각 백워드(Backward)하고 LPIPS 모델까지 포함하면 기존 하드웨어(e.g., RTX 4090 24GB)에서 OOM(Out of Memory) 발생 확률이 매우 높습니다. 이를 방지하기 위한 설계입니다:
>
> **1. Sequential Detach 및 Target 그래디언트 VRAM 최적화:**  
> $T_{struct}$, $T_{percep}$, $T_{phys}$ 를 각각 연산할 때 텐서 그래프를 동시에 올리지 않습니다. 또한, **타겟 $T_k$의 그래디언트 $\nabla_x T_k$를 추출할 때는 반드시 `create_graph=False`를 적용**하여 불필요한 2차 미분 그래프 생성을 막고 즉시 `.detach()` 합니다. 반면 모델 에너지 $\nabla_x E_k$ 계산 시에는 파라미터 $\theta$ 학습을 위해 `create_graph=True`를 유지합니다.
>
> **2. LPIPS Backbone 최적화:**  
> 무거운 VGG 대신 파라미터가 적고 가벼운 `AlexNet` 기반의 LPIPS를 사용하여 VRAM 캐파를 확보합니다.
>
> **3. Strict AMP:**  
> 물리 연산(Sobel, 강제 형변환 등)과 LPIPS 추론 과정에서 `torch.cuda.amp.autocast()`를 철저히 감싸서 메모리 피크를 절반으로 줄입니다.

---

## Proposed Changes

### 1. `gm/config.py` 추가 및 변경
새로운 구조와 하이퍼파라미터를 지원하기 위해 `argparse` 옵션이 추가됩니다.

#### [MODIFY] `gm/config.py`
- **전역 모델 설정:**
  - `--compositional_ebm` 플래그 추가
- **Head On/Off 토글 및 가중치:**
  - `--enable_struct`, `--enable_percep`, `--enable_phys` (개별 Head 끄고 켜기 기능)
  - 헤드별 최종 Loss 결합 가중치: `--w_struct`, `--w_percep`, `--w_phys`
  - 각 헤드별 내부 Sobolev(Anchor) 고도 Loss 가중치: `--lambda_struct`, `--lambda_percep`, `--lambda_phys`
- **세부 물리(Phys) 로스 On/Off 토글 및 가중치:**
  - `--enable_phys_blur`, `--enable_phys_occ`, `--enable_phys_energy`, `--enable_phys_bokeh`
  - `--lambda_blur_edge`, `--lambda_occlusion`, `--lambda_energy`, `--lambda_bokeh`
- **구조(Struct) 가중치:**
  - `--alpha_struct` (L2), `--beta_struct` (SSIM)

---

### 2. `gm/model.py` 다중 헤드 확장
모델 백본(Backbone)은 공유하되, 최종 Output Layer에서 3개의 독립된 스칼라 에너지를 출력하도록 변경합니다.

#### [MODIFY] `gm/model.py`
- 기존 백본 모델(`SimpleResNetFiLM` 등)의 출력단을 3개의 독립된 Head(`fc_struct`, `fc_percep`, `fc_phys` 또는 `conv1x1` 3개)로 분기.
- `forward()` 함수의 반환값을 단일 `eng`에서 `(eng_struct, eng_percep, eng_phys)` 형태의 튜플로 변경.

---

### 3. 수학적 톱니바퀴의 정렬 및 Sobolev Training 구현
가장 핵심적인 변경 사항으로, 기존의 1가지 MSE/Target 기반 그래디언트 매칭을 3가지의 독립적 추적 로직으로 변경하고 `mean` 대신 `sum`으로 묶습니다.

#### [MODIFY] `gm/train.py`
- **Target 함수 모듈 준비:**
  - `SSIM` 모듈 초기화
  - `LPIPS` VGG/AlexNet 모델 로드 (frozen)
  - 4종류의 물리 로스 함수를 PyTorch 미분 가능 연산으로 직접 구현:
    1. **$L_{blur\_edge}$**: `kornia.filters.Sobel`을 사용해 공간 엣지를 추출하고 $W_{focal}(p) = \exp(-\gamma \cdot |CoC(p)|)$ 스칼라 가중치 맵을 곱하여 L2 Loss 계산 (sum 적용).
    2. **$L_{occlusion}$**: Depth Map의 엣지를 Sobel 필터로 추출해 $M_{occ}(p) = \tanh(\kappa \cdot \|\nabla_{spatial} D(p)\|)$ 마스크를 생성. 이때 그래디언트 안정성을 위해 $\kappa$의 초기값을 너무 크지 않게 설정하여 조절합니다.
    3. **$L_{energy}$**: `F.avg_pool2d(..., kernel_size=31, stride=1, padding=15)`를 통과시켜 광량의 로컬 평균 보존 강제.
    4. **$L_{bokeh}$**: 밝은 영역 마스킹 후 **`nn.MaxPool2d`를 통해 Dilation(팽창) 연산을 미분 가능하게 구현**하여 $M_{bokeh}(p)$ 생성. 이후 L1 Loss를 적용하여 어둡게 뭉개짐(Under-intensity) 철저히 차단.
- **Sobolev Training (기울기 및 고도 동시 학습):**
  - **오차 합산식 변경:** 모든 `F.mse_loss`의 `reduction='mean'`을 `reduction='sum'` 방식으로 일괄 교체하거나 명시적으로 `.sum()`을 사용. (수학적 적분 일치)
  - Step 마다 $x$ 를 입력으로 `(E_struct, E_percep, E_phys)` 출력.
  - 동시에 Target 값들 `(T_struct, T_percep, T_phys)` 계산.
  - `torch.autograd.grad`를 사용하여 3개의 E와 3개의 T에 대한 $\nabla_x$ 를 개별적으로 추출.
  - 각 헤드에 대해 Sobolev Loss 구성: 
    * `loss_k = \| \nabla_x E_k - \nabla_x T_k \|_2^2 + \lambda_k * (E_k - T_k)^2`  (단, $\|\cdot\|_2^2$ 등 연산 시 `sum` 기준 적용)
- **Loss 총합 및 TensorBoard 로깅:**
  - `L_total_train = w1 * loss_struct + w2 * loss_percep + w3 * loss_phys`
  - 물리 로스 내부 디테일(`L_blur_edge`, `L_occ`, `L_energy`, `L_bokeh`) 및 각 헤드의 Loss 흐름(궤적 오차, Anchor 오차)을 단계별로 `SummaryWriter` (TensorBoard)에 상세 추가.

---

### 4. Compositional Generation (추론 메커니즘 확립)
학습된 3대장 헤드를 단순히 합산하여 노이즈에서 이미지를 복원하는 추론 로직을 업데이트합니다.

#### [MODIFY] `gm/train.py` (Validation) 및 추론 코드
- **다중 그래디언트 합산:**
  - `current_image`에 대해 모델을 통과시켜 3개의 에너지를 얻음.
  - 각각에 대해 역전파하여 `grad_struct`, `grad_percep`, `grad_phys`를 추출.
  - 노이즈 역산 식: `pred_grad = grad_struct + grad_percep + grad_phys`
  - $x_{t+1} = x_t + \eta \cdot \text{pred\_grad}$ 으로 업데이트 진행.

---


## Verification Plan

### Automated Tests
1. **단일 스텝/단일 이미지 차원 체크 (Sanity Test):**
    - `python -m gm.train --single_scene_only --compositional_ebm --gm_steps 1`
    - Loss 값이 정상적인 Scale에서 출력되는지 확인.
    - OOM 발생 여부 확인.

### Manual Verification
1. **각 Head 그래디언트 시각화:**
    - Test 추론 스텝 중 $\nabla_x E_{struct}$, $\nabla_x E_{percep}$, $\nabla_x E_{phys}$ 가 각각 이미지에 어떠한 형태(구조선 피드백, 텍스처 피드백, 블러 렌더링 피드백)로 나타나는지 중간 텐서를 저장하여 눈으로 물리 모델의 조합(Composition)을 확인합니다.
