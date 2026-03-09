
import os
import json
import glob

root = r'C:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\collected_results'

print(f"{'RUN ID':<25} | {'MODE':<10} | {'DATA':<20} | {'ADV (Int/Ratio)':<20} | {'EPOCHS':<6} | {'TEST PSNR':<10}")
print("-" * 115)

# Get all directories starting with run_ or robust_
runs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

for r in runs:
    p = os.path.join(root, r)
    args_p = os.path.join(p, 'args.json')
    res_p = os.path.join(p, 'test_results.json')
    
    mode = "N/A"
    data_str = "N/A"
    adv_str = "None"
    epochs = "-"
    test_psnr = "-"
    
    if os.path.exists(args_p):
        try:
            with open(args_p, 'r') as f:
                args = json.load(f)
            
            # Determine Mode (Normal vs Robust/Adversarial)
            # Old train.py didn't have adv_mode, it used adv_interval > 0 for epoch-based
            # New train_robust.py has adv_mode
            
            adv_mode = args.get('adv_mode', 'epoch') # 'epoch' is implicit for old train.py
            if adv_mode == 'epoch' and args.get('adv_interval', 0) == 0:
                adv_mode = 'none'
            
            mode = adv_mode
            
            # Data composition
            data = []
            if args.get('use_augmented'): data.append('S')
            if args.get('use_weak'): data.append('W')
            if args.get('use_aif'): data.append('A')
            data_str = '+'.join(data) if data else 'Clean Only'
            
            # Adversarial Settings
            if adv_mode == 'none':
                adv_str = "OFF"
            elif adv_mode == 'epoch':
                 # train.py style
                 interval = args.get('adv_interval', 0)
                 epochs_adv = args.get('adv_train_epochs', 1) # added later
                 adv_str = f"Epoch(Int={interval}, Rep={epochs_adv})"
            else:
                 # train_robust.py style (append/replace)
                 ratio = args.get('adv_ratio', 0)
                 adv_str = f"{adv_mode}(Ratio={ratio:.2f})"
            
            epochs = args.get('epochs', '-')
            
        except Exception as e:
            mode = "Error"
    
    if os.path.exists(res_p):
        try:
            with open(res_p, 'r') as f:
                res = json.load(f)
            # Handle different result formats if any
            # evaluate.py saves 'psnr', 'ssim', 'lpips'
            psnr = res.get('psnr', 0)
            if isinstance(psnr, (int, float)):
                test_psnr = f"{psnr:.2f}"
            else:
                 test_psnr = str(psnr)
        except:
            test_psnr = "Err"
            
    print(f"{r:<25} | {mode:<10} | {data_str:<20} | {adv_str:<20} | {epochs:<6} | {test_psnr:<10}")
