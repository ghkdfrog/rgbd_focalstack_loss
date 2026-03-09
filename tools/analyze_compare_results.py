"""
Analyze compare_result JSON files to check if model predictions
peak at the matching diopter (where fixed input matches GT).
"""
import os
import json
import numpy as np

result_dir = r'C:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\compare_result'

# Correct diopter values
DP_FOCAL = np.linspace(0.1, 4.0, 40)

json_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.json')])

print("=" * 100)
print("DEFOCUS PERCEPTION VERIFICATION: Does Pred PSNR peak at the matching diopter?")
print("=" * 100)

for jf in json_files:
    with open(os.path.join(result_dir, jf), 'r') as f:
        data = json.load(f)
    
    fixed_input = data.get('fixed_input', '?')
    fixed_label = data.get('fixed_label', '?')
    label_a = data.get('label_a', 'Model A')
    label_b = data.get('label_b', 'Model B')
    planes = data['planes']
    
    # Find matching plane index (closest diopter to fixed_input)
    if fixed_input == 'aif':
        match_idx = None
        match_diopter = None
    else:
        target_d = float(fixed_input.replace('p', '.'))
        # But the diopter in the JSON might be from the OLD wrong DP_FOCAL
        # Use the plane index to determine the correct match
        dists = [abs(DP_FOCAL[i] - target_d) for i in range(40)]
        match_idx = int(np.argmin(dists))
        match_diopter = DP_FOCAL[match_idx]
    
    # Extract predictions
    pred_a_psnr = [p['pred_a']['psnr'] for p in planes]
    pred_b_psnr = [p['pred_b']['psnr'] for p in planes]
    pred_a_ssim = [p['pred_a']['ssim'] for p in planes]
    pred_b_ssim = [p['pred_b']['ssim'] for p in planes]
    pred_a_lpips = [p['pred_a']['lpips'] for p in planes]
    pred_b_lpips = [p['pred_b']['lpips'] for p in planes]
    real_psnr = [p['real']['psnr'] for p in planes]
    
    # Find peaks
    a_psnr_max_idx = int(np.argmax(pred_a_psnr))
    b_psnr_max_idx = int(np.argmax(pred_b_psnr))
    a_ssim_max_idx = int(np.argmax(pred_a_ssim))
    b_ssim_max_idx = int(np.argmax(pred_b_ssim))
    a_lpips_min_idx = int(np.argmin(pred_a_lpips))
    b_lpips_min_idx = int(np.argmin(pred_b_lpips))
    real_psnr_max_idx = int(np.argmax(real_psnr))
    
    print(f"\n--- {jf} ---")
    print(f"  Fixed input: {fixed_label}")
    if match_idx is not None:
        print(f"  Expected match: plane {match_idx} (diopter={match_diopter:.3f}D)")
    else:
        print(f"  Expected match: AiF (no specific diopter)")
    
    print(f"  Real PSNR peak:      plane {real_psnr_max_idx:>2} (d={DP_FOCAL[real_psnr_max_idx]:.3f}D)  val={real_psnr[real_psnr_max_idx]:.2f}")
    print()
    
    def check(metric_name, peak_idx, expected_idx, val):
        if expected_idx is None:
            status = "N/A"
        elif peak_idx == expected_idx:
            status = "OK"
        else:
            off = abs(peak_idx - expected_idx)
            status = f"MISS (off by {off} planes, {abs(DP_FOCAL[peak_idx]-DP_FOCAL[expected_idx]):.2f}D)"
        print(f"    {metric_name:>20}: peak at plane {peak_idx:>2} (d={DP_FOCAL[peak_idx]:.3f}D)  val={val:.4f}  [{status}]")
    
    print(f"  {label_a}:")
    check("PSNR max", a_psnr_max_idx, match_idx, pred_a_psnr[a_psnr_max_idx])
    check("SSIM max", a_ssim_max_idx, match_idx, pred_a_ssim[a_ssim_max_idx])
    check("LPIPS min", a_lpips_min_idx, match_idx, pred_a_lpips[a_lpips_min_idx])
    
    print(f"  {label_b}:")
    check("PSNR max", b_psnr_max_idx, match_idx, pred_b_psnr[b_psnr_max_idx])
    check("SSIM max", b_ssim_max_idx, match_idx, pred_b_ssim[b_ssim_max_idx])
    check("LPIPS min", b_lpips_min_idx, match_idx, pred_b_lpips[b_lpips_min_idx])

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("If peaks consistently match the expected plane -> model learned defocus blur well.")
print("If peaks are always at plane 0 or 39 (edges) -> model ignores diopter conditioning.")
print("If peaks are random -> model is confused about diopter.")
