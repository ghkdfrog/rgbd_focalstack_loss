# Focal-Conditioned Loss Estimation Network — Research Plan

## 최종 목표

**Focal-Aware Differentiable Perceptual Loss**: RGBD + predicted focal plane + diopter를 입력으로 받아 PSNR/SSIM/LPIPS를 예측하는 네트워크를 학습하고, 이를 differentiable loss function으로 사용 가능하게 만든다.

---

## 모델 구조 확정

**Spatial conditioning 채택** (실험 결과 기준)

| mode | PSNR | SSIM | LPIPS | 결론 |
|---|---|---|---|---|
| **Spatial** | real 거의 완벽 추적 | 매우 안정적 | 우수 | **채택** |
| CoC | noise floor collapse | 방향 맞지만 과소평가 | 가장 우수 | 비교 실험으로 남김 |
| Sinusoidal | 완전 실패 | 부분 성공 | 부분 성공 | 제외 |

> CoC는 물리적 해석 가능성이 있어 논문의 비교 대상(Section)으로 유지. Baseline은 Spatial.

---

## 학습 구성 (최종)

### 데이터 타입 (배치별)

| 배치 타입 | pred channel 내용 | 목적 |
|---|---|---|
| **match** | DeepFocus clean (올바른 focal plane) | 높은 점수 정확히 예측 |
| **unmatch ×39** | 나머지 39개 focal plane | 낮은 점수 정확히 예측 |
| **strong noise** | Gaussian noise (σ large) | 열화 이미지 평가 능력 |
| **weak noise** | Gaussian noise (σ small) | 미세 열화 구분 |
| **AIF** | All-in-focus image | out-of-context 처리 |
| **gradient alignment** | initial_plane (noise/AIF 등) | gradient landscape 교정 |

### 사전 준비
```bash
python generate_unmatch_labels.py --data_dir ../varifocal/data
# 약 20분, GPU 배치 처리 → labels_unmatch.json 생성
```

### 학습 명령어
```bash
python train_focal.py \
    --data_dir ../varifocal/data \
    --diopter_mode spatial \
    --unmatch_ratio 39 \
    --val_unmatch_ratio 39 \
    --epochs 5
```

### Validation 구성
- Scene 90~109 (20개 scene)
- match + unmatch 전체 → `val_unmatch_ratio=39`
- `labels_unmatch.json` 사전 생성 필요 (없으면 on-the-fly, 매우 느림)

---

## Gradient Alignment 학습 (신규)

기존 adv training (15-step Adam) 대체. 배치당 1.5배 비용으로 대폭 절감.

### 원리
```
direction_gt   = gt_plane - initial_plane   # 실제로 가야 할 방향
direction_pred = ∂(psnr + ssim - lpips) / ∂initial_plane  # 모델이 말하는 방향
alignment_loss = -cosine_similarity(direction_pred, direction_gt)
```

### 코드 스케치
```python
initial_plane.requires_grad_(True)
out = model(torch.cat([rgbd, initial_plane], dim=1), diopter)

# PSNR/SSIM은 높을수록 좋음(+), LPIPS는 낮을수록 좋음(-)
scalar = (out['psnr'] + out['ssim'] - out['lpips']).mean()
scalar.backward(retain_graph=True, create_graph=True)

direction_pred = initial_plane.grad
direction_gt   = (gt_plane - initial_plane).detach()

alignment_loss = -F.cosine_similarity(
    direction_pred.flatten(1), direction_gt.flatten(1), dim=1
).mean()
alignment_loss.backward()
optimizer.step()
```

### 효과
- metric network가 "올바른 gradient landscape"를 가지도록 교정
- 이 metric으로 image를 최적화할 때 gradient descent가 gt 방향으로 수렴
- **loss function으로 사용하기 위한 핵심 조건 충족**

---

## Selling Points

### ✅ 1. 명확한 Problem Gap
기존 LPIPS/SSIM/PSNR은 focal/diopter 정보 없음. "어느 plane에 focus됐는가"를 평가하는 differentiable metric이 없음.

### ✅ 2. Renderer 없이 빠른 focal quality 예측
```
기존: RGBD → Renderer(느리고 defocus 부정확) → GT focal plane → LPIPS
제안: RGBD → LossEstimationNet(<5ms) → quality prediction
```

**Renderer defocus 지원 현황 (2025 기준)**:

| Renderer | Defocus 지원 | 속도 |
|---|---|---|
| PyTorch3D | ⚠️ 근사 (BlendParams.sigma) | 중간 |
| nvdiffrast | ❌ pinhole only | 빠름 |
| 3DGS (DOF-GS, CoCoGaussian) | ✅ CoC 물리 모델 | 실시간 (CVPR 2025) |
| NeRF (DoF-NeRF, DP-NeRF) | ⚠️ per-scene 학습 필요 | 느림 |

→ 3DGS가 2025년부터 막 지원 시작. 기존 대부분 renderer는 여전히 미흡.

### ✅ 3. End-to-end Differentiable
기존 renderer의 defocus는 미분 불가하거나 부정확. 우리 모델은 full gradient 가능.

### ✅ 4. LossEstimationNet은 DeepFocus보다 ~10배 이상 경량
- LossEstimationNet: ~6~8M params, <5ms 추론
- DeepFocus (LVF U-Net): ~60M params, ~100ms 추론
- optimization loop에서 수십~수백 회 반복 시 실용적인 차이

### ✅ 5. Gradient Landscape 교정 (gradient alignment)
단순 metric 예측이 아니라 gradient 방향까지 학습 → loss function으로 실제 수렴 보장.

---

## 공격받을 포인트 & 방어 논리

### ❌ Attack 1: "LPIPS를 focal-aware하게 wrapping하면 되지 않나?"
**방어**: GT blur kernel을 알아야 하고, 실제로는 spatially varying DoF (깊이 맵 기반) 이므로 단순 wrapping 불가. 우리 모델은 RGBD에서 직접 예측.

### ❌ Attack 2: "Label이 DeepFocus 예측 기반 PSNR/SSIM — DeepFocus의 quality를 예측하는 것 아닌가?"
**방어**: 현재는 맞음. **Real optical benchmark (실측 focal stack)에서의 generalization 검증 필요** (future work). 또는 Blender physically-simulated DoF GT 사용.

### ❌ Attack 3: "실제 downstream에서 LPIPS보다 낫다는 걸 보여야 한다"
**방어**: 핵심 실험 — 이 loss로 renderer/pipeline 최적화 시 LPIPS 대비 focal accuracy gain. 이 실험 하나가 논문의 핵심.

### ❌ Attack 4: "3DGS (DOF-GS, CoCoGaussian)이 이미 CoC-based DoF를 렌더링한다"
**방어**: 3DGS는 rendering 품질을 높이는 것이고, 우리는 rendering 결과를 평가/최적화하는 differentiable criterion. 상호 보완적이며 3DGS의 training signal로도 사용 가능.

---

## 관련 연구 (공격/비교 대상)

| 논문 | 관계 | Google Scholar |
|---|---|---|
| LPIPS (Zhang et al., CVPR 2018) | 직접 baseline | [링크](https://scholar.google.com/scholar?q=The+Unreasonable+Effectiveness+of+Deep+Features+as+a+Perceptual+Metric+Zhang+2018) |
| DeepFocus (Chakravarthula et al., SIGGRAPH Asia 2018) | 직접 대상 시스템 | [링크](https://scholar.google.com/scholar?q=DeepFocus+rendering+accommodation+supporting+displays+Chakravarthula+2018) |
| DOF-GS / CoCoGaussian (CVPR 2025) | renderer 진영 최신 | [링크](https://scholar.google.com/scholar?q=CoCoGaussian+Circle+of+Confusion+Gaussian+Splatting+2024) |
| Metameric Varifocal Holography (UCL) | 비슷한 목적, 다른 접근 | [링크](https://scholar.google.com/scholar?q=metameric+varifocal+holography+perceptual+loss) |
| DISTS (Ding et al., TPAMI 2022) | texture-aware IQA 비교 대상 | [링크](https://scholar.google.com/scholar?q=DISTS+image+quality+assessment+texture+structure+invariance+Ding+2022) |

---

## 향후 계획 (Roadmap)

```
Phase 1 (현재): Conditioning 방식 확정
  ✅ spatial > CoC > sinusoidal 확인
  → unmatch_ratio=39로 spatial 재학습

Phase 2: 학습 구성 완성
  → strong/weak noise + AIF + gradient alignment 통합
  → train_robust.py adv 부분 대체

Phase 3: Downstream 검증 (핵심)
  → 이 loss로 rendering pipeline 최적화
  → LPIPS-only 대비 focal accuracy 비교
  → 이게 논문의 Table 1

Phase 4: Label 품질 개선 (optional)
  → Real focal stack (실측) generalization 검증
  → Blender physically-simulated DoF GT

Phase 5: 논문화
  → SIGGRAPH / CVPR / ICCV 타겟
```
