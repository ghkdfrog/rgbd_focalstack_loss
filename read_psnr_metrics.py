import os
import json

def print_psnr_results():
    runs_dir = r"c:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\runs_gm"
    
    print(f"{'Run Name':<40} | {'0.1D PSNR':<9} | {'2.0D PSNR':<9} | {'4.0D PSNR':<9} | {'Avg PSNR':<9}")
    print("-" * 85)
    
    for run_name in sorted(os.listdir(runs_dir)):
        if "20260309" not in run_name and "20260310" not in run_name:
            continue
            
        run_path = os.path.join(runs_dir, run_name)
        psnr_file = os.path.join(run_path, "psnr_best.json")
        
        if not os.path.exists(psnr_file):
            continue
            
        try:
            with open(psnr_file, 'r') as f:
                data = json.load(f)
                
            results = data.get('results', [])
            if not results:
                continue
                
            # Assume order is 0.1D, 2.0D, 4.0D based on the training script
            psnr_01 = results[0]['psnr'] if len(results) > 0 else 0
            psnr_20 = results[1]['psnr'] if len(results) > 1 else 0
            psnr_40 = results[2]['psnr'] if len(results) > 2 else 0
            
            avg_psnr = sum(r['psnr'] for r in results) / len(results)
            
            print(f"{run_name[:40]:<40} | {psnr_01:>9.2f} | {psnr_20:>9.2f} | {psnr_40:>9.2f} | {avg_psnr:>9.2f}")
        except Exception as e:
            pass

if __name__ == "__main__":
    print_psnr_results()
