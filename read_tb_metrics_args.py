import os
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_tb_metrics_with_args():
    runs_dir = r"c:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\runs_gm"
    
    print(f"{'Run Name':<40} | {'Head':<7} | {'LR':<7} | {'Sched':<8} | {'Best Loss':<9} | {'Best Ep'}")
    print("-" * 90)
    
    for run_name in sorted(os.listdir(runs_dir)):
        if "20260309" not in run_name and "20260310" not in run_name:
            continue
            
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path): continue
            
        args_path = os.path.join(run_path, "args.json")
        logs_dir = os.path.join(run_path, "logs")
        
        if not os.path.exists(args_path) or not os.path.isdir(logs_dir):
            continue
            
        with open(args_path, 'r') as f:
            args = json.load(f)
            
        head = args.get('energy_head', 'fc')
        lr = f"{args.get('lr', 0):.1e}"
        sched = args.get('eta_schedule', 'constant')
        
        event_acc = EventAccumulator(logs_dir)
        try: event_acc.Reload()
        except: continue
        
        tags = event_acc.Tags()
        if 'scalars' not in tags or 'val/loss' not in tags['scalars']: continue
        
        val_losses = event_acc.Scalars('val/loss')
        if not val_losses: continue
            
        losses = [v.value for v in val_losses]
        steps = [v.step for v in val_losses]
        
        best_loss = min(losses)
        best_epoch = steps[losses.index(best_loss)]
        
        print(f"{run_name[:40]:<40} | {head:<7} | {lr:<7} | {sched:<8} | {best_loss:.6f} | {best_epoch:>3}")

if __name__ == "__main__":
    print_tb_metrics_with_args()
