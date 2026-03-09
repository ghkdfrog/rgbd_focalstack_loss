# RGBD Focal Stack Loss Estimation Model

## Phase 1: Data Generation ✅
- [x] DeepFocus LFS로 전체 씬 추론 및 저장
- [x] GT와 비교하여 MSE, SSIM, LPIPS 계산
- [x] Labels 파일 생성 및 저장

## Phase 2: Model Implementation ✅
- [x] Loss Estimation 네트워크 정의
- [x] Dataset 클래스 구현
- [x] 학습 스크립트 작성
- [x] MSE → PSNR 메트릭 전환

## Phase 3: Training & Validation ✅
- [x] 학습 실행 (50 epochs)
- [x] 검증 및 테스트
- [x] 결과 분석 및 시각화

## Phase 4: Data Augmentation & Improvement 🔄
- [x] 데이터 증강 파이프라인 구현 (Strong/Weak/AiF)
- [x] 증강 데이터 생성 스크립트 작성 (`generate_augmented_dataset.py`)
- [x] Dataset 클래스 업데이트 (증강 데이터 로딩 지원)
- [ ] 증강 데이터 생성 실행
- [ ] LPIPS 가중치 조정하여 재학습
- [ ] 증강 데이터 포함하여 평가

## Phase 5: Verification & Robustness ✅
- [x] Loss Function 검증 스크립트 작성 (`verify_as_loss.py`)
- [x] Adversarial Training 파이프라인 구현
    - [x] Pixel Optimization으로 Adversarial Sample 생성
    - [x] 모델을 속이는 샘플에 대해 Real Metric으로 재학습
- [x] Adversarial Training 실행 및 효과 분석
    - [x] Epoch-based Adv (Old) 실행 -> 실패 (Severe Hacking)
    - [x] On-the-fly Adv (Append Mode, Ratio 0.33) 실행 -> 완화 (Mild Hacking)
    - [x] Spectral Normalization (SN) 적용 -> 실패 (Performance Drop)

## Phase 6: Analysis & Plotting ✅
- [x] 실험 결과 수집 스크립트 작성 (`collect_results.py`)
- [x] Loss Curve 비교 그래프 생성 (`plot_loss_curves.py`)
- [x] Convergence 비교 그래프 생성 (`plot_convergence_comparison.py`)
- [x] 분석 결과 정리 및 다운로드 패키징

## Phase 7: Ground Truth Anchor (Core Fix) 🔄
- [x] GT Anchor 전략 수립 (노이즈 없는 100점짜리 데이터의 필요성)
- [ ] Dataset 클래스 수정 (GT Anchor 모드 추가)
    - [ ] 입력=정답(GT), 라벨=완벽함(PSNR 100, LPIPS 0) 쌍을 강제로 주입
- [ ] 50% 비율로 GT Anchor 데이터 섞어서 재학습
- [ ] `verify_as_loss` 재검증 (Hacking 완전 해결 목표)

## Phase 8: Model Optimization (Optional) ⏳
- [ ] 모델 구조 개선 (Safety Features)
    - [ ] Bounded Output (Sigmoid * 100) 적용 (폭주 방지)
    - [ ] Dropout 추가 (Overfitting 방지)
- [ ] 경량화 모델 설계 (Channel 50% 축소)
- [ ] Knowledge Distillation (Teacher-Student) 고려
- [ ] 최종 모델 선정 및 배포 스크립트 작성
