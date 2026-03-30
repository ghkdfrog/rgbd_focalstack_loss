# Gradient Matching Runs Argument Summary
model.py, config.py, 	rain.py에 정의된 옵션과 아키텍처에 기초하여 
uns_gm 내부의 각 run 별 argument 설정을 분석한 보고서입니다.

---

## 📌 주요 옵션 및 하이퍼파라미터 설명

### 1. Model Architecture (gm/model.py)
- **simple**: 5-layer CNN (stride=1로 공간 해상도 유지)
- **deep**: 10-layer CNN (simple 버전에 레이어 추가 방식)
- **stride**: 5-layer CNN, 일부 컨볼루션 레이어에서 stride=2 를 사용하여 글로벌 Receptive Field 확장
- **resnet**: SimpleResNet. stride=1 을 유지하면서 깊이를 쌓는 Residual Block 구조
- **resnet_film**: SimpleResNetFiLM. CoC 맵을 채널 합치지 않고 FiLM conditioning에만 사용
- **interleave_resnet**: InterleaveResNet. DeepFocus의 Pixel Shuffle (Interleave/Deinterleave) 기법 적용
- 기타 (
esunet, convnext, convnext_unet, dilated) 등 다양한 모델 구조 실험 가능.

### 2. Conditioning & Energy Head (gm/config.py, gm/model.py)
- **Diopter Mode (diopter_mode)**: 모델이 대상이 되는 Focus 영역 정보를 전달받는 형식
  - spatial: 공간에 상수값 부여 (diopter 맵 차원 추가)
  - coc / coc_abs / coc_signed: Circle of Confusion 연산을 통해 포커스 블러가 없는 In-focus 영역 위주로 물리적 조건 추가
- **Energy Head (energy_head)**: 최종 Energy 값 도출 방식
  - c: Linear Layer 사용하여 (해상도 512x512 고정) 작동
  - conv1x1: 1x1 Convolutional Layer와 Spatial 합계만 사용하여 해상도 무관성 획득

### 3. Gradient Matching 학습 방식 (gm/train.py, gm/config.py)
- **Eta Schedule (eta_schedule)**: Gradient descent 기반의 Langevin dynamics(SGLD) Trajectory 스텝 보폭($\eta$)
  - constant (고정 $\etamax$), cosine / linear (스텝 진행에 따른 점진적 보폭 감소 처리 방식 적용)
- **Langevin Noise (langevin_noise)**: 업데이트 시 Langevin 추가 노이즈 더할지 여부 강제 추가 (수렴성 확인 용도)
- **GM Steps (gm_steps)**: 각 이미지 배치를 계산하여 Gradient Matching Trajectory 궤적을 탐색할 스텝 횟수
- **FiLM (use_film)**: Spatial FiLM conditioning의 ResBlock 안으로의 투입 활성화
- **Sharp Prior (sharp_prior)**: CoC=0 근방(In-focus 영역)일수록 Generator가 원본 RGB에 충실하도록 하는 해석적 이차항 Penalty 추가 
- **Scene0 (single_scene_only)**: 알고리즘 성능의 오버피팅 가능성을 검증하기 위해 Scene 0번째 데이터 하나만을 사용하여 학습. 대부분 O로 표기되어 빠른 성능 측정을 위해 사용되었음.

---

## 📊 전체 Run Arguments 요약 트리 (
uns_gm/)

| Run Name | Arch | Diopter Mode | Energy Head | Eta Sched | Noise | FiLM | Sharp | GM Step | Scene0 | LR | Epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gm_scene0_spatial_20260308_164220 | simple | spatial | fc | constant | X | X | X | 100 | O | 0.0001 | 20 |
| gm_scene0_spatial_20260307_232125 | simple | spatial | fc | constant | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_spatial_20260307_172316 | simple | spatial | fc | constant | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_signed_linear_20260325_125534 | resnet_film | coc_signed | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_signed_linear_20260319_150817 | resnet | coc_signed | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_signed_linear_20260318_130552 | film_resnet | coc_signed | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260326_135234 | resnet_film | coc | fc | linear | X | X | O | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260320_063400 | resunet | coc | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260319_150917 | resnet | coc | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260318_112238 | resnet | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_linear_20260317_033217 | convnext_unet | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260314_160511 | resnet | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260313_133317 | stride | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260313_041643 | deep | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_linear_20260312_120804 | deep | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260312_120655 | simple | coc | fc | linear | O | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_linear_20260310_011629 | simple | coc | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_cosine_20260310_011608 | simple | coc | fc | cosine | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_conv1x1_linear_20260320_055937 | resunet | coc | conv1x1 | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_conv1x1_linear_20260318_202001 | resnet | coc | conv1x1 | linear | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_conv1x1_linear_20260317_034204 | convnext_unet | coc | conv1x1 | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_conv1x1_20260309_174409 | simple | coc | conv1x1 | constant | X | X | X | 100 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260325_200924 | resnet_film | coc_abs | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260325_200543 | resnet | coc_abs | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260324_180506 | resnet | coc_abs | fc | linear | X | O | O | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_abs_linear_20260323_203943 | resnet | coc_abs | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260323_203707 | resnet | coc_abs | fc | linear | X | O | O | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260323_124739 | resnet_film | coc_abs | fc | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260323_042653 | resnet | coc_abs | fc | linear | X | O | O | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260323_042615 | resnet | coc_abs | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260321_130849 | interleave_resnet | coc_abs | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_linear_20260321_125237 | interleave_resnet | coc_abs | fc | linear | X | O | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_conv1x1_linear_20260319_151904 | convnext_unet | coc_abs | conv1x1 | linear | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_abs_conv1x1_linear_20260318_132413 | dilated | coc_abs | conv1x1 | linear | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_20260312_120940 | deep | coc | fc | constant | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_20260311_191411 | deep | coc | fc | constant | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_20260309_175428 | simple | coc | fc | constant | X | X | X | 50 | O | 3e-05 | 100 |
| gm_scene0_coc_20260309_175036 | simple | coc | fc | constant | X | X | X | 50 | O | 0.0001 | 100 |
| gm_scene0_coc_20260309_084426 | simple | coc | fc | constant | X | X | X | 100 | O | 0.0001 | 100 |
| gm_scene0_coc_20260309_084328 | simple | coc | fc | constant | X | X | X | 100 | O | 3e-05 | 100 |
| gm_scene0_coc_20260308_212501 | simple | coc | fc | constant | X | X | X | 100 | O | 3e-05 | 50 |
| gm_scene0_coc_20260308_212402 | simple | coc | fc | constant | X | X | X | 100 | O | 0.001 | 50 |
| gm_scene0_coc_20260308_164208 | simple | coc | fc | constant | X | X | X | 100 | O | 0.0001 | 20 |
| gm_scene0_coc_20260307_172330 | simple | coc | fc | constant | X | X | X | 100 | O | 0.0001 | 50 |
| gm_scene0_coc_20260307_094147 | simple | coc | fc | constant | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_20260307_094134 | simple | coc | fc | constant | X | X | X | 50 | O | 0.0001 | 50 |
| gm_scene0_coc_20260306_222752 | simple | coc | fc | constant | X | X | X | 5 | O | 0.0001 | 50 |
| gm_scene0_coc_20260306_221916 | simple | coc | fc | constant | X | X | X | 5 | O | 0.0001 | 50 |
| gm_coc_linear_20260316_083358 | resnet | coc | fc | linear | X | X | X | 50 | X | 0.0001 | 5 |
| gm_coc_linear_20260312_134650 | simple | coc | fc | linear | X | X | X | 50 | X | 0.0001 | 100 |
| gm_coc_cosine_20260310_075915 | simple | coc | fc | cosine | X | X | X | 100 | X | 0.0001 | 100 |
| gm_coc_conv1x1_cosine_20260310_080631 | simple | coc | conv1x1 | cosine | X | X | X | 100 | X | 0.0001 | 100 |
| gm_coc_abs_linear_20260327_131355 | resnet_film | coc_abs | fc | linear | X | X | X | 50 | X | 0.0001 | 5 |
| gm_coc_abs_linear_20260324_100203 | resnet | coc_abs | fc | linear | X | O | X | 50 | X | 0.0001 | 5 |
| gm_coc_abs_linear_20260323_152249 | resnet | coc_abs | fc | linear | X | O | X | 50 | X | 0.0001 | 5 |
| gm_coc_abs_conv1x1_linear_20260327_131349 | resnet_film | coc_abs | conv1x1 | linear | X | X | X | 50 | X | 0.0001 | 5 |
| gm_coc_abs_conv1x1_linear_20260325_125239 | resnet | coc_abs | conv1x1 | linear | X | O | X | 50 | X | 0.0001 | 5 |
| gm_coc_20260311_142349 | simple | coc | fc | constant | X | X | X | 50 | X | 0.0001 | 5 |
