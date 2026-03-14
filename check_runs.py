import os
import json

base_dir = r"d:\Deepfocus\rgbd_focalstack_loss\runs_gm_results\runs_gm"
runs = ["gm_coc_linear_20260312_134650", "gm_scene0_coc_20260312_120940", "gm_scene0_coc_linear_20260312_120804", "gm_scene0_coc_linear_20260313_041643"]

for r in runs:
    print(f"--- {r} ---")
    args_path = os.path.join(base_dir, r, "args.json")
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
            print(f"Target Epochs: {args.get('epochs', 'N/A')}")
            print(f"Resume: {args.get('resume', False)}")
    else:
        print("No args.json")
        
    metrics_path = os.path.join(base_dir, r, "logs", "metrics.csv")
    if os.path.exists(metrics_path):
        import csv
        try:
            with open(metrics_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                epochs = []
                for row in reader:
                    if 'epoch' in row:
                        epochs.append(int(row['epoch']))
                if epochs:
                    max_ep = max(epochs)
                    print(f"Current Epoch in metrics: {max_ep}")
                    
                    # Check for restarts: if epoch numbers go down or jump weirdly
                    # Wait, simpler: if the list length > unique elements
                    if len(epochs) > len(set(epochs)):
                        print(f"Note: Found duplicate epochs in CSV. Suggests the run was stopped and resumed/restarted.")
                        
                    # Also print the last few epochs to see if it stopped before max epoch
                    print(f"Total rows in metrics: {len(epochs)}, Max Epoch: {max_ep}")
        except Exception as e:
            print(f"Error reading metrics: {e}")
    else:
        print("No metrics.csv")
        
    for p in ["psnr_best.json", "psnr_latest.json", "psnr_best_psnr.json"]:
        p_path = os.path.join(base_dir, r, p)
        if os.path.exists(p_path):
            with open(p_path, 'r') as f:
                d = json.load(f)
                psnr = d.get('avg', 'N/A')
                ep = d.get('epoch', 'N/A')
                print(f"{p}: Epoch {ep}, PSNR {psnr}")
