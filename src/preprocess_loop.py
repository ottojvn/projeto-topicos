import sys
import subprocess
from pathlib import Path

# Garante caminhos corretos independentemente de onde o script for executado
ROOT_DIR = Path(__file__).resolve().parent.parent

for window in range(10, 31, 5):
    print(f"\nwindow_sec = {window}s\n")
    out_dir = ROOT_DIR / "data" / f"saida_{window}s"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "preprocess.py"),
        "--dataset_path", str(ROOT_DIR / "data" / "Data_DS2_raspberry_npz" / "Data_DS2_raspberry_npz"),
        "--gt_dir", str(ROOT_DIR / "data" / "Data_DS2_polar-main" / "Data_Heart"),
        "--out_dir", str(out_dir),
        "--window_sec", f"{window}.0",
        "--step_sec", "0.5",
    ]

    subprocess.run(cmd, check=True)
