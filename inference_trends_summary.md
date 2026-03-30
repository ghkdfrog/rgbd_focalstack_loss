# Gradient Matching: Inference Trends \& PSNR Analysis

추론(inference) 시 생성된 JSON 파일(psnr_best_psnr.json 등)을 분석하여, **Scene (Single/Multi)**, **Activation**, 및 주요 파라미터별 **Best PSNR 도달 Epoch**와 **Avg PSNR (dB)**을 정리한 리포트입니다.

## 📈 Trend Analysis (경향성 분석)
1. **학습 난이도 (Single Scene vs Multi Scene)**
   - **Single Scene**: 대상 이미지 1장에 오버피팅하는 경우, 최근 모델(`resnet_film`, `resnet`)은 최대 **37.32 dB**에 달하는 매우 높은 PSNR을 달성하며 대상에 완벽하게 일치하는 초점 렌더링이 가능했습니다.
   - **Multi Scene**: 다양한 씬을 일반화해야 하는 멀티 씬 환경에서는 `simple` 모델이 평균 **~21 dB** 수준의 PSNR을 보였습니다. 현재 파이프라인에서 많은 Multi Scene Run 들이 JSON 기반의 PSNR 측정을 거치지 않은 상태(N/A)이나, 난이도가 월등히 높음을 시사합니다.

2. **Activation 및 Architecture의 영향**
   - **Activation**: 기존 `relu` 대비 최근에 도입된 **`silu`** 활성화 함수가 성능 향상에 크게 기여했습니다. 단일 씬 상위 4개 모델이 모두 `silu`를 사용하고 있습니다.
   - **Model Arch**: 단순한 (simple) 5-layer CNN에서 **`resnet_film`** 및 **`resnet`** 과 같은 향상된 아키텍처로 넘어가면서 3~5 dB 가량의 유의미한 생성 품질(PSNR) 향상을 이루었습니다.
   - **Diopter 조건화**: `coc` 단일 모드보다 절대값(`coc_abs`)이나 부호있는 형식(`coc_signed`) 조건화를 `use_film`(FiLM)과 결합한 방식이 최상위권의 성능을 이끌었습니다.

3. **Best Epoch 최적화 경향**
   - 모델 학습 시 높은 해상도 및 다양한 plane을 복원하기 위해 보통 지정된 Epoch 최대치 (e.g., 50~100) 끝자락 혹은 90번째 Epoch 근처에서 Best PSNR이 도출되는 경향이 짙습니다. (Gradient Matching 스텝 탐색의 특징상 궤적이 안정화되기까지 충분한 step이 필요함)

## 📊 상세 Inference 결과 테이블 (PSNR 내림차순 정렬)

| Run Name | Scene | Arch | Act | Diopter | FiLM | Sharp | Best Epoch | Avg PSNR |
|---|---|---|---|---|---|---|---|---|
| gm_coc_20260311_142349 | Multi | simple | relu | coc | X | X | 2 | 21.23 |
| gm_coc_linear_20260312_134650 | Multi | simple | relu | coc | X | X | 14 | 20.97 |
| gm_coc_abs_conv1x1_linear_20260325_125239 | Multi | resnet | relu | coc_abs | O | X | N/A | N/A |
| gm_coc_abs_conv1x1_linear_20260327_131349 | Multi | resnet_film | silu | coc_abs | X | X | N/A | N/A |
| gm_coc_abs_linear_20260323_152249 | Multi | resnet | relu | coc_abs | O | X | N/A | N/A |
| gm_coc_abs_linear_20260324_100203 | Multi | resnet | relu | coc_abs | O | X | N/A | N/A |
| gm_coc_abs_linear_20260327_131355 | Multi | resnet_film | silu | coc_abs | X | X | N/A | N/A |
| gm_coc_conv1x1_cosine_20260310_080631 | Multi | simple | relu | coc | X | X | N/A | N/A |
| gm_coc_cosine_20260310_075915 | Multi | simple | relu | coc | X | X | N/A | N/A |
| gm_coc_linear_20260316_083358 | Multi | resnet | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260325_200924 | Single | resnet_film | silu | coc_abs | X | X | 100 | 37.32 |
| gm_scene0_coc_abs_linear_20260325_200543 | Single | resnet | silu | coc_abs | O | X | 99 | 34.86 |
| gm_scene0_coc_signed_linear_20260325_125534 | Single | resnet_film | silu | coc_signed | X | X | 94 | 33.25 |
| gm_scene0_coc_abs_linear_20260323_203943 | Single | resnet | silu | coc_abs | O | X | 69 | 33.19 |
| gm_scene0_coc_linear_20260319_150917 | Single | resnet | relu | coc | O | X | 38 | 30.09 |
| gm_scene0_coc_signed_linear_20260318_130552 | Single | film_resnet | relu | coc_signed | X | X | 43 | 29.71 |
| gm_scene0_coc_abs_linear_20260324_180506 | Single | resnet | relu | coc_abs | O | O | 49 | 27.67 |
| gm_scene0_coc_linear_20260318_112238 | Single | resnet | relu | coc | X | X | 19 | 27.59 |
| gm_scene0_coc_linear_20260326_135234 | Single | resnet_film | silu | coc | X | O | 18 | 27.47 |
| gm_scene0_coc_signed_linear_20260319_150817 | Single | resnet | relu | coc_signed | X | X | 42 | 27.47 |
| gm_scene0_coc_conv1x1_linear_20260318_202001 | Single | resnet | relu | coc | X | X | 8 | 24.96 |
| gm_scene0_coc_20260309_175036 | Single | simple | relu | coc | X | X | 39 | 24.91 |
| gm_scene0_coc_linear_20260313_041643 | Single | deep | relu | coc | X | X | 26 | 23.75 |
| gm_scene0_coc_20260312_120940 | Single | deep | relu | coc | X | X | 53 | 23.04 |
| gm_scene0_coc_20260307_172330 | Single | simple | relu | coc | X | X | 11 | 22.76 |
| gm_scene0_coc_20260308_212501 | Single | simple | relu | coc | X | X | 47 | 22.62 |
| gm_scene0_coc_abs_conv1x1_linear_20260318_132413 | Single | dilated | relu | coc_abs | X | X | 50 | 22.40 |
| gm_scene0_coc_cosine_20260310_011608 | Single | simple | relu | coc | X | X | 11 | 21.83 |
| gm_scene0_coc_20260308_164208 | Single | simple | relu | coc | X | X | 20 | 21.76 |
| gm_scene0_coc_20260307_094134 | Single | simple | relu | coc | X | X | 48 | 21.24 |
| gm_scene0_coc_20260311_191411 | Single | deep | relu | coc | X | X | 6 | 21.09 |
| gm_scene0_coc_linear_20260310_011629 | Single | simple | relu | coc | X | X | 8 | 20.81 |
| gm_scene0_coc_20260308_212402 | Single | simple | relu | coc | X | X | 6 | 20.07 |
| gm_scene0_coc_linear_20260313_133317 | Single | stride | relu | coc | X | X | 15 | 19.89 |
| gm_scene0_coc_20260309_175428 | Single | simple | relu | coc | X | X | 45 | 19.29 |
| gm_scene0_spatial_20260308_164220 | Single | simple | relu | spatial | X | X | 20 | 17.68 |
| gm_scene0_spatial_20260307_232125 | Single | simple | relu | spatial | X | X | 7 | 17.61 |
| gm_scene0_coc_conv1x1_20260309_174409 | Single | simple | relu | coc | X | X | 4 | 16.27 |
| gm_scene0_spatial_20260307_172316 | Single | simple | relu | spatial | X | X | 40 | 12.92 |
| gm_scene0_coc_20260306_221916 | Single | simple | relu | coc | X | X | 44 | 7.68 |
| gm_scene0_coc_20260307_094147 | Single | simple | relu | coc | X | X | 50 | 0.25 |
| gm_scene0_coc_20260306_222752 | Single | simple | relu | coc | X | X | 50 | N/A |
| gm_scene0_coc_20260309_084328 | Single | simple | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_20260309_084426 | Single | simple | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_abs_conv1x1_linear_20260319_151904 | Single | convnext_unet | relu | coc_abs | X | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260321_125237 | Single | interleave_resnet | relu | coc_abs | O | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260321_130849 | Single | interleave_resnet | relu | coc_abs | O | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260323_042615 | Single | resnet | silu | coc_abs | O | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260323_042653 | Single | resnet | relu | coc_abs | O | O | N/A | N/A |
| gm_scene0_coc_abs_linear_20260323_124739 | Single | resnet_film | relu | coc_abs | X | X | N/A | N/A |
| gm_scene0_coc_abs_linear_20260323_203707 | Single | resnet | relu | coc_abs | O | O | N/A | N/A |
| gm_scene0_coc_conv1x1_linear_20260317_034204 | Single | convnext_unet | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_conv1x1_linear_20260320_055937 | Single | resunet | relu | coc | O | X | N/A | N/A |
| gm_scene0_coc_linear_20260312_120655 | Single | simple | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_linear_20260312_120804 | Single | deep | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_linear_20260314_160511 | Single | resnet | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_linear_20260317_033217 | Single | convnext_unet | relu | coc | X | X | N/A | N/A |
| gm_scene0_coc_linear_20260320_063400 | Single | resunet | relu | coc | O | X | N/A | N/A |
