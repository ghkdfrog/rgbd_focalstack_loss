
import os
import json

root = r'C:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\collected_results'

# Key runs to compare
runs = [
    ('robust_20260217_010547', 'Robust (No SN)'),
    ('robust_20260217_011302', 'Robust (SN, 20ep)'),
    ('run_20260214_040522', 'Epoch-based'),
    ('run_20260214_040344', 'Baseline')
]

print(f"{'RUN ID':<25} | {'DESC':<12} | {'INIT PSNR':<10} | {'FINAL PSNR':<10} | {'PRED PSNR':<10} | {'GAIN (dB)':<10} | {'RESULT'}")
print("-" * 110)

for r_id, desc in runs:
    json_path = os.path.join(root, r_id, 'verification', 'optimization_history.json')
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                history = json.load(f)
            
            init = history['real_psnr'][0]
            final = history['real_psnr'][-1]
            pred = history['pred_psnr'][-1]
            gain = final - init
            
            res = "SUCCESS" if gain > 0 else "FAIL (Hacked)"
            if gain > 0.5: res = "STRONG SUCCESS"
            if pred > 50 and final < init: res = "FAIL (Severe Hacking)"

            print(f"{r_id:<25} | {desc:<12} | {init:<10.2f} | {final:<10.2f} | {pred:<10.2f} | {gain:<+10.2f} | {res}")
        except Exception as e:
            print(f"{r_id:<25} | {desc:<12} | Error reading JSON: {e}")
    else:
        print(f"{r_id:<25} | {desc:<12} | Verification not run")
