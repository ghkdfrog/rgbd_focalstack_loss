# Gradient Matching Runs Summary (3/25 이후, CoC clip(-1,1) 기준)

> 3/25 이전 runs는 CoC clip(0,1) 방식이므로 참조 불필요.

## 유효한 Runs (결과 보유)

| # | Run Name | Arch | Head | Diopter | Sharp | Langevin | Epochs | 비고 |
|---|----------|------|------|---------|-------|----------|--------|------|
| 1 | `200543` | resnet (use_film) | fc | coc_abs | ✗ | ✗ | 100 | resnet + use_film=True |
| 2 | `200924` | resnet_film | fc | coc_abs | ✗ | ✗ | 200 | FiLM baseline, resumed (3/31) |
| 3 | `125534` | resnet_film | fc | coc_signed | ✗ | ✗ | 100 | coc_signed 테스트 |
| 4 | `135234` | resnet_film | fc | coc | ✓ (λ=5,γ=50) | ✗ | 100 | Sharp Prior (learnable) |
| 5 | `131103` | resnet_film | fc | coc_abs | ✗ | ✓ (0.1) | 100 | **Langevin noise 테스트** |

> 모든 run 공통: ch=128, act=silu, lr=1e-4, wd=1e-5, gm_steps=50, eta_schedule=linear

### PSNR 결과 비교 (best_psnr checkpoint 기준)

| # | Run | Plane 0 (0.1D) | Plane 20 (2.1D) | Plane 39 (4.0D) | **평균** |
|---|-----|:-:|:-:|:-:|:-:|
| 2 | **FiLM coc_abs 200ep** | **38.30** | **40.81** | **42.41** | **40.51** |
| 1 | resnet+film coc_abs 100ep | 33.12 | 35.67 | 35.77 | 34.85 |
| 3 | FiLM coc_signed 100ep | 31.20 | 34.28 | 34.26 | 33.25 |
| 5 | FiLM coc_abs **Langevin** 100ep | 30.83 | 32.39 | 32.49 | 31.90 |
| 4 | FiLM coc sharp 100ep | 28.18 | 28.72 | 25.50 | 27.47 |

> **분석**:
> - FiLM + coc_abs + 200ep (#2)이 압도적으로 우수. Resume으로 100→200ep 연장 학습.
> - Langevin noise (#5) 사용 시 100ep 기준 #2의 100ep 대비 성능 하락. 노이즈가 수렴을 방해했을 가능성.
> - Sharp Prior (#4)는 PSNR 자체는 낮으나, 초점 영역 선명도 개선이 목적이므로 단순 PSNR 비교에 한계.
> - #1은 `arch=resnet`이지만 `use_film=True`로 FiLM 블록을 사용 (SimpleResNet 내부 FiLM 옵션).

### 세부 결과

<details>
<summary><b>#1 — gm_scene0_coc_abs_linear_20260325_200543</b></summary>

- **Arch**: `resnet` + `use_film=True` (SimpleResNet with FiLM blocks)
- **Diopter**: coc_abs | **Sharp**: ✗ | **Langevin**: ✗
- **Epochs**: 100 | **save_every**: 10
- **Best (loss, ep100)**: P0: 33.14 / P20: 35.60 / P39: 35.81
- **Best (PSNR, ep99)**: P0: 33.12 / P20: 35.67 / P39: 35.77

</details>

<details>
<summary><b>#2 — gm_scene0_coc_abs_linear_20260325_200924</b> ⭐ 최고 성능</summary>

- **Arch**: `resnet_film` (SimpleResNetFiLM)
- **Diopter**: coc_abs | **Sharp**: ✗ | **Langevin**: ✗
- **Epochs**: 200 (100ep 학습 후 resume으로 200ep 연장, timestamp: 20260331)
- **save_every**: 10
- **Best (loss, ep199)**: P0: 38.29 / P20: 40.76 / P39: 42.26
- **Best (PSNR, ep200)**: P0: 38.30 / P20: 40.81 / P39: 42.41
- **비고**: 마지막 epoch에서도 계속 개선 중 → 더 학습하면 더 좋아질 가능성

</details>

<details>
<summary><b>#3 — gm_scene0_coc_signed_linear_20260325_125534</b></summary>

- **Arch**: `resnet_film` | **Diopter**: coc_signed | **Sharp**: ✗ | **Langevin**: ✗
- **Epochs**: 100 | **save_every**: 10
- **Best (loss, ep94)**: P0: 31.20 / P20: 34.28 / P39: 34.26
- **Best (PSNR, ep94)**: P0: 31.20 / P20: 34.28 / P39: 34.26
- **Latest (ep100)**: P0: 29.60 / P20: 31.38 / P39: 30.07 — **ep94 이후 성능 급락**
- **비고**: 학습 후반 불안정. coc_signed 방식의 한계 가능성.

</details>

<details>
<summary><b>#4 — gm_scene0_coc_linear_20260326_135234</b></summary>

- **Arch**: `resnet_film` | **Diopter**: coc (raw, 비정규화)
- **Sharp Prior**: ✓ (λ=5.0, γ=50.0, learnable) | **Langevin**: ✗
- **Epochs**: 100 | **save_every**: 5
- **Best (loss, ep19)**: P0: 27.66 / P20: 28.18 / P39: 24.64
- **Best (PSNR, ep18)**: P0: 28.18 / P20: 28.72 / P39: 25.50
- **Latest (ep100)**: P0: 27.75 / P20: 27.67 / P39: 25.54
- **비고**: Sharp Prior 사용. best가 ep18-19로 매우 이른 시점에 발생. 이후 수렴 정체. 운 좋게 NaN 회피한 run.

</details>

<details>
<summary><b>#5 — gm_scene0_coc_abs_linear_20260330_131103</b> 🔊 Langevin Noise</summary>

- **Arch**: `resnet_film` | **Diopter**: coc_abs
- **Sharp**: ✗ | **Langevin**: ✓ (`constant_scale`, scale=0.1)
- **Epochs**: 100 | **save_every**: 10
- **Best (loss, ep99)**: P0: 30.83 / P20: 32.38 / P39: 32.52
- **Best (PSNR, ep99)**: P0: 30.83 / P20: 32.39 / P39: 32.49
- **Latest (ep100)**: P0: 30.75 / P20: 32.27 / P39: 32.49
- **비고**: Langevin noise 추가. #2(동일 설정, noise 없음, 200ep)와 비교하면 100ep 기준 약간 낮음. 노이즈가 수렴 속도를 늦추었을 가능성.

</details>

---

## 현재 학습 중 ⚡

| Run Name | Arch | Head | Diopter | Sharp | Ch | Act | Epochs | 비고 |
|----------|------|------|---------|-------|-----|-----|--------|------|
| `gm_scene0_coc_abs_conv1x1_linear_20260402_141307` | resnet_film | conv1x1 | coc_abs | ✓ | 128 | silu | 200 | scene0 |
| `gm_scene0_coc_abs_conv1x1_linear_20260401_164601` | resnet_film | conv1x1 | coc_abs | ✗ | 128 | silu | 200 | scene0 |
| `gm_coc_abs_conv1x1_linear_20260325_125239` | resnet | conv1x1 | coc_abs | ✗ | 128 | relu | 5 | multi-scene |
| `gm_coc_abs_conv1x1_linear_20260327_131349` | resnet_film | conv1x1 | coc_abs | ✗ | 128 | silu | 5 | multi-scene |
| `gm_coc_abs_linear_20260327_131355` | resnet_film | fc | coc_abs | ✗ | 128 | silu | 5 | multi-scene |

---

## 삭제 대상 (실패/NaN/테스트 잔해) — 12개

### NaN 발생 (nan.txt 존재)
| Run Name | 설명 |
|----------|------|
| `gm_scene0_coc_abs_conv1x1_linear_20260402_090127` | sharp=True, conv1x1, NaN |
| `gm_scene0_coc_abs_conv1x1_linear_20260402_090659` | sharp=True, conv1x1, NaN |
| `gm_scene0_coc_abs_conv1x1_linear_20260402_092627` | sharp=True, conv1x1, NaN |
| `gm_scene0_coc_abs_linear_20260402_093731` | sharp=True, fc, NaN |

### 학습 미진행 (NaN 디버깅 잔해)
| Run Name | 설명 |
|----------|------|
| `gm_scene0_coc_linear_20260402_120706` | 135234 재현 시도, 미학습 |
| `gm_scene0_coc_linear_20260402_121958` | NaN 디버깅 테스트 |
| `gm_scene0_coc_linear_20260402_124134` | NaN 디버깅 테스트 |
| `gm_scene0_coc_linear_20260402_125411` | NaN 디버깅 테스트 |
| `gm_scene0_coc_linear_20260402_131925` | NaN 디버깅 테스트 |
| `gm_scene0_coc_linear_20260402_134345` | NaN 디버깅 테스트 |
| `gm_scene0_coc_linear_20260402_135056` | 구버전 코드 NaN 테스트 |
| `gm_scene0_coc_linear_20260402_140057` | 구버전 코드 NaN 테스트 |
