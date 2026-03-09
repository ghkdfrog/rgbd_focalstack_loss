
import os
import json
import struct
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Try to use tensorboardX protos
try:
    from tensorboardX.proto import event_pb2
except ImportError:
    try:
        from tensorboard.compat.proto import event_pb2
    except ImportError:
        print("Error: Could not import event_pb2 from tensorboardX or tensorboard.")
        print("Please install tensorboardX: pip install tensorboardX")
        exit(1)

def read_events_file(file_path):
    """
    Manually read TF events file since tensorboard is not installed.
    Format:
    uint64 len
    uint32 crc
    bytes(len) payload
    uint32 crc
    """
    events = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                # Read length (8 bytes)
                header = f.read(8)
                if not header or len(header) < 8:
                    break
                
                event_len = struct.unpack('Q', header)[0]
                
                # Read CRC (4 bytes)
                crc1 = f.read(4)
                
                # Read payload
                payload = f.read(event_len)
                
                # Read CRC (4 bytes)
                crc2 = f.read(4)
                
                if len(payload) != event_len:
                    break
                
                # Parse protobuf
                event = event_pb2.Event()
                event.ParseFromString(payload)
                events.append(event)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return events

def get_config_label(args):
    """Generate a descriptive label from args.json"""
    mode = args.get('adv_mode', 'epoch')
    if mode == 'epoch' and args.get('adv_interval', 0) == 0:
        mode = 'none'
    
    if mode == 'none':
        data = []
        if args.get('use_augmented'): data.append('S')
        if args.get('use_weak'): data.append('W')
        if args.get('use_aif'): data.append('A')
        return f"Baseline ({'+'.join(data) if data else 'Clean'})"
    
    elif mode == 'epoch':
        interval = args.get('adv_interval', 1)
        epochs_adv = args.get('adv_train_epochs', 1)
        return f"Epoch-based (Int={interval}, Rep={epochs_adv})"
    
    else: # append/replace
        ratio = args.get('adv_ratio', 0)
        return f"Robust {mode.capitalize()} (Ratio={ratio:.2f})"

def main():
    root_dir = r'C:\Users\dhlab\Desktop\DeepFocus\rgbd_focalstack_loss\collected_results'
    output_png = 'loss_curves_comparison.png'
    
    # Store data: {label: {'train': [], 'val': [], 'epochs': []}}
    data_store = defaultdict(lambda: {'train_loss': {}, 'val_loss': {}, 'epochs': 0})
    
    target_runs = {
        'robust_20260217_010547': 'Best Robust (Append, Ratio=0.33)',
        'run_20260214_040522': 'Best Epoch-based (Rep=20)',
        'run_20260214_040344': 'Baseline (No Adv)'
    }
    
    print(f"Plotting comparison for selected runs: {list(target_runs.keys())}...")
    
    runs = os.listdir(root_dir)
    
    for run_id in target_runs.keys():
        if run_id not in runs:
            print(f"Warning: {run_id} not found in collected_results.")
            continue
            
        run_path = os.path.join(root_dir, run_id)
        args_path = os.path.join(run_path, 'args.json')
        log_dir = os.path.join(run_path, 'logs')
        
        if not os.path.exists(args_path) or not os.path.exists(log_dir):
            continue
            
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        label = target_runs[run_id]
        full_label = label  # Use clean label for legend 
        
        # Find event file
        event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue
        
        # Parse events
        # Prioritize newest file if multiple? usually usually appended or one file.
        # Just process all
        train_steps = {}
        val_steps = {}
        max_step = 0
        
        for ef in event_files:
            events = read_events_file(ef)
            for e in events:
                for v in e.summary.value:
                    if v.tag == 'train/loss' or v.tag == 'loss':
                        train_steps[e.step] = v.simple_value
                        max_step = max(max_step, e.step)
                    elif v.tag == 'val/loss':
                        val_steps[e.step] = v.simple_value
                        max_step = max(max_step, e.step)
        
        # Store for plotting
        # Sort by step
        sorted_train = sorted(train_steps.items())
        sorted_val = sorted(val_steps.items())
        
        data_store[full_label]['train'] = sorted_train
        data_store[full_label]['val'] = sorted_val
        data_store[full_label]['epochs'] = args.get('epochs', 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_store)))
    
    for idx, (label, data) in enumerate(data_store.items()):
        color = colors[idx]
        
        # Train Loss
        train_steps = [s for s,v in data['train']]
        train_vals = [v for s,v in data['train']]
        axes[0].plot(train_steps, train_vals, label=label, color=color, alpha=0.8)
        
        # Val Loss
        val_steps = [s for s,v in data['val']]
        val_vals = [v for s,v in data['val']]
        axes[1].plot(val_steps, val_vals, label=label, color=color, linestyle='--', marker='o', markersize=3, alpha=0.8)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (L1)')
    axes[0].grid(True, alpha=0.3)
    # axes[0].set_yscale('log')
    axes[0].legend(fontsize='small')

    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (L1)')
    axes[1].grid(True, alpha=0.3)
    # axes[1].set_yscale('log')
    axes[1].legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")

if __name__ == "__main__":
    main()
