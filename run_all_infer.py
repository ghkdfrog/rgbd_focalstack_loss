import os
import subprocess

# Set the base directory containing the runs
RUNS_DIR = "runs_gm"

def main():
    # Check if the directory exists
    if not os.path.exists(RUNS_DIR):
        print(f"Error: Directory '{RUNS_DIR}' not found.")
        return

    # List all entries in the runs directory
    all_entries = sorted(os.listdir(RUNS_DIR))
    
    # Filter for directories that start with 'gm_'
    run_names = [d for d in all_entries if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("gm_")]
    
    print(f"Found {len(run_names)} potential run directories in '{RUNS_DIR}'.")
    print("=" * 60)
    
    executed_count = 0
    skipped_count = 0
    
    for run_name in run_names:
        run_dir_path = os.path.join(RUNS_DIR, run_name)
        best_psnr_path = os.path.join(run_dir_path, "best_psnr_model.pth")
        
        # Check if best_psnr_model.pth exists in this run
        if os.path.exists(best_psnr_path):
            print(f"\n[RUN] Starting inference for: {run_name}")
            
            # Construct the inference command
            cmd = ["python", "-m", "gm.infer", "--run_dir", run_dir_path, "--ckpt_tag", "best_psnr"]
            
            try:
                # Execute the command
                subprocess.run(cmd, check=True)
                print(f"[SUCCESS] Finished inference for: {run_name}")
                executed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Inference failed for {run_name} with exit code {e.returncode}")
            except KeyboardInterrupt:
                print("\n[ABORT] Process interrupted by user.")
                break
        else:
            print(f"[SKIP] No best_psnr_model.pth found in: {run_name}")
            skipped_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs checked: {len(run_names)}")
    print(f"Inferences executed: {executed_count}")
    print(f"Runs skipped: {skipped_count}")

if __name__ == "__main__":
    main()
