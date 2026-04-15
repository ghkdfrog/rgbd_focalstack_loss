# Mathematical Formulation of Physically-aware Compositional EBM

본 문서는 현재 코드(`gm/model.py`, `gm/train.py`, `gm/compositional.py`)에 구현된 Compositional EBM 모델의 핵심 수학적 수식을 정리한 문서입니다.

---

## 1. 다중 헤드 에너지 지형 (Compositional Energy Landscape)
단일 에너지 함수를 사용하는 기존 모델을 확장하여, 전체 에너지 지형 $E_{total}$을 3개의 독립적인 하위 에너지 지형의 합으로 정의합니다.

$$ E_{total}(x, c) = E_{struct}(x, c) + E_{percep}(x, c) + E_{phys}(x, c) $$
- **$c$**: 조건(Condition). Focus Blur 문제의 경우 Diopter 혹은 CoC(Circle of Confusion)
- **$x$**: 현재 렌더링 중(Langevin sampling 중)인 이미지

Langevin Dynamics 기반 추론 단계에서는 위 공간에서 에너지를 최소화하는 방향으로 업데이트를 진행합니다:
$$ x_{t+1} = x_t - \eta \cdot \nabla_x \big( E_{struct}(x) + E_{percep}(x) + E_{phys}(x) \big) + \epsilon $$

---

## 2. Sobolev Training과 타겟 그래디언트의 분리 (Sequential Detach)
에너지 지형이 우리가 원하는 에러(Error) 지표를 정확히 모사하도록(Surrogate) 학습해야 합니다.

### 기울기 매칭(Trajectory Matching)
각 에너지 헤드가 타겟 손실 함수($T_k$)가 가리키는 하강 방향을 그대로 학습합니다.

$$ \mathcal{L}_{traj} = \sum_{p} \left( w_{struct} \left\| \nabla_x E_{struct} - \nabla_x T_{struct} \right\|_c^2 + w_{percep} \left\| \nabla_x E_{percep} - \nabla_x T_{percep} \right\|_c^2 + w_{phys} \left\| \nabla_x E_{phys} - \nabla_x T_{phys} \right\|_c^2 \right) $$

> [!IMPORTANT]
> **OOM 방지를 위한 $\nabla_x T_k$ 차단 (Sequential Detach)**
> 타겟 그래디언트 계산 시 $\nabla_x T_k$의 연산 그래프가 역전파 과정에서 유지되면 엄청난 VRAM 메모리가 소비됩니다.
> 따라서 계산 직후 `.detach()`되며, 수식적으로 $\nabla_x T_k$는 변수가 아닌 **상수 벡터 필드(Constant Vector Field)**로 취급됩니다.

### 고도 매칭(Energy Anchor Loss)
정답 이미지($x_{gt}$)에서의 에너지가 가장 낮아야(항상 0이어야) 하므로 기준점(Anchor)을 고정합니다.

$$ \mathcal{L}_{anchor} = \sum_{k \in \{struct, percep, phys\}} \lambda_k \cdot \big\| E_k(x_{gt}, c) - 0 \big\|_2^2 $$

---

## 3. 타겟 에너지 (Physical Sub-Losses)
`gm/compositional.py` 에 구현된 구체적 타겟 물리 로스 함수($T$)의 정의입니다. 여기서 $\sum$은 미니배치, 채널, 공간 픽셀에 대한 합산 행위를 의미합니다 (`reduction='sum'`).

### 3.1. Structure Target ($T_{struct}$)
픽셀 레벨의 L2 스케일 오차와 마이크로 구조 오차(L1 Surrogate)의 합입니다.
$$ T_{struct}(x, x_{gt}) = \alpha_{struct} \|x - x_{gt}\|_2^2 + \beta_{struct} \|x - x_{gt}\|_1 $$

### 3.2. Perception Target ($T_{percep}$)
사전 학습된 딥러닝 피처(AlexNet 등) 기반의 인간 인지 오차입니다.
$$ T_{percep}(x, x_{gt}) = \text{LPIPS}_{alex}(x, x_{gt}) $$

---

### 3.3. Physics Target ($T_{phys}$)
**초점 심도 렌더링(DOF)**에서 발생하는 물리적 특성을 수학적으로 제어하는 종합 물리 제약 조건입니다.
$$ T_{phys} = \lambda_{blur} L_{blur\_edge} + \lambda_{occ} L_{occ} + \lambda_{enrgy} L_{energy} + \lambda_{bokeh} L_{bokeh} $$

각 항의 세부 수식은 다음과 같습니다:

#### (1) Blur Edge Loss (초점 엣지 오차)
아웃포커싱(Out-of-focus) 영역보다 인포커싱(In-focus) 영역에 있는 윤곽선 미스매치에 궤멸적인 페널티를 부여합니다.
- **공간 1차 미분(Gradient):** $\nabla_{xy} x = \sqrt{(F_{x} * x)^2 + (F_{y} * x)^2}$ (*: 2D Convolution)
- **가중치 맵:** $W_{focal}(p) = \exp(-\gamma \cdot |CoC(p)|)$
- **수식:** 
  $$ L_{blur\_edge} = \sum_p \Big[ W_{focal}(p) \cdot \big\| \nabla_{xy} x_{pred}(p) - \nabla_{xy} x_{gt}(p) \big\|_2^2 \Big] $$

#### (2) Occlusion Boundary Loss (깊이 단절 오차)
깊이(Depth) 차이가 크게 발생하는 경계면에서 픽셀 색상 오차가 번지는 것을(Color Bleeding) 방지합니다.
- **깊이 엣지 마스크:** $M_{occ}(p) = \tanh(\kappa \cdot \|\nabla_{xy} D(p)\|)$  (단, $D$는 Depth Map)
- **수식:** 
  $$ L_{occ} = \sum_p \Big[ M_{occ}(p) \cdot \| x_{pred}(p) - x_{gt}(p) \|_2^2 \Big] $$

#### (3) Energy Conservation Loss (에너지 보존 법칙)
초점이 날아가 영상이 블러(Blur) 처리 되더라도, 특정 반경 안의 전체 빛(광량)의 총합은 보존되어야 합니다.
- **수식:** 
  $$ L_{energy} = \sum_p \big\| \text{AvgPool}_{\Omega}(x_{pred})(p) \ - \ \text{AvgPool}_{\Omega}(x_{gt})(p) \big\|_2^2 $$
  (단, $\Omega$는 `energy_pool_k` 크기의 로컬 필터)

#### (4) High-Intensity Bokeh Loss (광원 보케 정합)
강한 광원($Threshold$ 이상) 주변의 보케(Bokeh) 팽창이 렌더링 도중 소실되거나 뭉개지는 현상을 방지합니다. Dilation 연산을 미분 가능한 `MaxPool` 층으로 대체하였습니다.
- **광원 마스크:** $I_{bright}(p) = \mathbb{1}(x_{gt}(p) > \tau)$
- **Dilation:** $M_{bokeh} = \text{MaxPool}_{\Omega}(I_{bright})$
- **수식:** (형태의 섬세한 정합을 위해 극단적 페널티인 L1 노름 채택)
  $$ L_{bokeh} = \sum_p \Big[ M_{bokeh}(p) \cdot \| x_{pred}(p) - x_{gt}(p) \|_1 \Big] $$
