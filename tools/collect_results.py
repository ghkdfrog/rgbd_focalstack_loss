
import os
import shutil
import glob
from tqdm import tqdm

def collect_results(src_root="runs", dest_root="collected_results"):
    # 타겟 파일 및 폴더 이름 (Run 내부)
    target_files = ["args.json", "test_results.json", "prediction_scatter.png"]
    target_dirs = ["verification", "logs"]
    
    # 추가로 수집할 파일 (루트 경로)
    extra_files = [
        "analyze_collected_results.py", 
        "analyze_verification.py", 
        "plot_loss_curves.py",
        "loss_curves_comparison.png",
        "plot_convergence_comparison.py",
        "optimization_convergence_comparison.png"
    ]

    print(f"Collecting results from '{src_root}' to '{dest_root}'...")

    if os.path.exists(dest_root):
        # 기존 폴더 유지하고 내용물만 덮어쓰기 위해 rmtree 대신 makedirs(exist_ok=True) 사용
        # 사용자 요청에 따라 rmtree 할 수도 있지만 안전하게 유지.
        # 그러나 깔끔하게 하려면 지우는게 나음.
        # 사용자가 "다시 실행" 하는 경우를 고려해 일단 둔다.
        pass
    os.makedirs(dest_root, exist_ok=True)
    
    # 0. 분석 스크립트 및 결과 이미지 복사
    print("Collecting analysis scripts & plots...")
    for f in extra_files:
        if os.path.exists(f):
            print(f"  Copying {f}...")
            shutil.copy2(f, os.path.join(dest_root, f))
        else:
            print(f"  Warning: {f} not found.")

    # 1. runs 아래의 모든 run_ 폴더 찾기
    if os.path.exists(src_root):
        run_dirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
        
        print(f"Found {len(run_dirs)} run directories in '{src_root}'.")

        for run_dir in tqdm(run_dirs, desc="Collecting Runs"):
            src_run_path = os.path.join(src_root, run_dir)
            dest_run_path = os.path.join(dest_root, run_dir)
            
            # 해당 run 폴더 생성
            os.makedirs(dest_run_path, exist_ok=True)

            # 2. 파일 복사
            for fname in target_files:
                src = os.path.join(src_run_path, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest_run_path, fname))

            # 3. 폴더 전체 복사 (verification, logs)
            for dname in target_dirs:
                src = os.path.join(src_run_path, dname)
                dest = os.path.join(dest_run_path, dname)
                
                if os.path.exists(src):
                    if os.path.exists(dest):
                        shutil.rmtree(dest) # 덮어쓰기 위해 삭제
                    shutil.copytree(src, dest)
    else:
        print(f"Warning: Source directory '{src_root}' does not exist.")

    print(f"\n[OK] All results collected into '{dest_root}'!")
    print(f"You can now download the '{dest_root}' folder.")

if __name__ == "__main__":
    collect_results()
