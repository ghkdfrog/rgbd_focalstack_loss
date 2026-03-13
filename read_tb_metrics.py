import os
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_tb_metrics():
    runs_dir = r"c:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\runs_gm"
    
    print("Run Name | Best Val Loss | Best Epoch | Last Epoch")
    print("-" * 75)
    
    for run_name in sorted(os.listdir(runs_dir)):
        # Only look at recent runs from 20260309 and 20260310
        if "20260309" not in run_name and "20260310" not in run_name:
            continue
            
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
            
        logs_dir = os.path.join(run_path, "logs")
        if not os.path.isdir(logs_dir):
            continue
            
        # Initialize EventAccumulator
        event_acc = EventAccumulator(logs_dir)
        try:
            event_acc.Reload()
        except:
            continue
            
        # Check if val/loss exists
        tags = event_acc.Tags()
        if 'scalars' not in tags or 'val/loss' not in tags['scalars']:
            continue
            
        val_losses = event_acc.Scalars('val/loss')
        
        if not val_losses:
            continue
            
        losses = [v.value for v in val_losses]
        steps = [v.step for v in val_losses]
        
        best_loss = min(losses)
        best_epoch = steps[losses.index(best_loss)]
        last_epoch = steps[-1]
        
        print(f"{run_name[:40]:<40} | {best_loss:.6f} | {best_epoch:>3} | {last_epoch:>3}")

if __name__ == "__main__":
    print_tb_metrics()
