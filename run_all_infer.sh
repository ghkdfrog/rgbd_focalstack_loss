#!/bin/bash
# ============================================================
# 모든 유효한 GM Run에 대해 inference 실행
# - infer.py가 각 run의 args.json에서 설정을 자동 복원함
#   (gm_steps, step_size, eta_schedule, eta_min, langevin_noise, diopter_mode 등)
# - 각 run 폴더 안 inference/ 아래에 결과 저장됨:
#   step_psnr_table.csv, psnr_convergence.png, results.json 등
# ============================================================

PLANES="0,20,39"

echo "===== CoC + FC 그룹 ====="

# #1: 50ep, steps=5, ss=0.2 (prototype)
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260306_221916 --plane_idx $PLANES --gm_steps 200

# #2: 50ep, steps=5, ss=0.002 (보폭 너무 작음 → 참고용)
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260306_222752 --plane_idx $PLANES --gm_steps 200

# #3: 50ep, steps=50, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260307_094134 --plane_idx $PLANES --gm_steps 200

# #4: 50ep, steps=50, ss=0.002 (보폭 너무 작음 → 참고용)
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260307_094147 --plane_idx $PLANES --gm_steps 200

# #5: 50ep, steps=100, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260307_172330 --plane_idx $PLANES --gm_steps 200

# #6: 20ep, steps=100, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260308_164208 --plane_idx $PLANES --gm_steps 200

# #7: 50ep, lr=3e-5, steps=100, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260308_212402 --plane_idx $PLANES --gm_steps 200

# #8: 50ep, lr=3e-5, steps=100, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260308_212501 --plane_idx $PLANES --gm_steps 200

# #9: 100ep, steps=50, ss=0.2 (👑 최고 성능)
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260309_175036 --plane_idx $PLANES --gm_steps 200

# #10: 100ep, lr=3e-5, steps=50, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_20260309_175428 --plane_idx $PLANES --gm_steps 200

# #11: 100ep, cosine schedule, steps=50, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_cosine_20260310_011608 --plane_idx $PLANES --gm_steps 200

# #12: 100ep, linear schedule, steps=50, ss=0.2
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_linear_20260310_011629 --plane_idx $PLANES --gm_steps 200


echo "===== CoC + conv1x1 그룹 ====="

# #13: 100ep, conv1x1 head, constant, steps=100
python -m gm.infer --run_dir runs_gm/gm_scene0_coc_conv1x1_20260309_174409 --plane_idx $PLANES --gm_steps 200


echo "===== Spatial 그룹 ====="

# #14: 50ep, spatial, steps=50
python -m gm.infer --run_dir runs_gm/gm_scene0_spatial_20260307_172316 --plane_idx $PLANES --gm_steps 200

# #15: 100ep, spatial, steps=50
python -m gm.infer --run_dir runs_gm/gm_scene0_spatial_20260307_232125 --plane_idx $PLANES --gm_steps 200

# #16: 20ep, spatial, steps=100
python -m gm.infer --run_dir runs_gm/gm_scene0_spatial_20260308_164220 --plane_idx $PLANES --gm_steps 200


echo "===== Multi-scene (5장) 그룹 ====="

# #19: coc + fc + cosine, 5 scenes, batch=4
python -m gm.infer --run_dir runs_gm/gm_coc_cosine_20260310_075915 --plane_idx $PLANES

# #20: coc + conv1x1 + cosine, 5 scenes, batch=4
python -m gm.infer --run_dir runs_gm/gm_coc_conv1x1_cosine_20260310_080631 --plane_idx $PLANES

echo "===== Done! ====="
