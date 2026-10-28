import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

for window in range(10, 31, 5):
    print(f"\n{'='*50}\nTreinando Regressão Linear para window_sec = {window}s\n{'='*50}\n")
    
    input_dir = ROOT_DIR / "data" / f"saida_{window}s"
    output_dir = ROOT_DIR / "output" / f"linear_{window}s"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "train_lr.py"),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
    ]

    subprocess.run(cmd, check=True)
