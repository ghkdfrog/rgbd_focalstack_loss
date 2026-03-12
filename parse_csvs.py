import os, json, glob, csv

base = 'd:/Deepfocus/rgbd_focalstack_loss/runs_gm_results/runs_gm'
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

for d in dirs:
    dpath = os.path.join(base, d)
    args_file = os.path.join(dpath, 'args.json')
    if not os.path.exists(args_file): continue
    
    with open(args_file) as f:
        args = json.load(f)
    gm_steps = args.get('gm_steps', 50)
    sched = args.get('eta_schedule', 'constant')
    
    csv_files = glob.glob(os.path.join(dpath, 'inference', '**', 'step_psnr_table.csv'), recursive=True)
    if not csv_files: continue
    csv_file = csv_files[0]
    
    with open(csv_file) as f:
        reader = list(csv.DictReader(f))
        
    if not reader: continue
    row0 = reader[0]
    
    p20 = next((h for h in row0.keys() if 'plane20' in h), None)
    if not p20: continue
    
    train_row = next((r for r in reader if int(r['step']) == gm_steps), reader[-1])
    final_row = reader[-1]
    best20_row = max(reader, key=lambda r: float(r[p20]))
        
    print(f'===== {d} =====')
    print(f'Train Steps: {gm_steps} | Sched: {sched}')
    print(f'PSNR @ Train ({gm_steps}): {float(train_row[p20]):.2f} dB')
    s = best20_row["step"]
    print(f'PSNR @ Peak ({s}): {float(best20_row[p20]):.2f} dB')
    fs = final_row["step"]
    print(f'PSNR @ Final ({fs}): {float(final_row[p20]):.2f} dB\n')
