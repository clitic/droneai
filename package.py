import zipfile
import os
import re
from pathlib import Path

def get_version():
    with open("pyproject.toml") as f:
        match = re.search(r'version\s*=\s*"([^"]+)"', f.read())
        return match.group(1) if match else "unknown"

def create_package():
    version = get_version()
    release_name = f"droneai-{version}.zip"
    
    targets = {
        "src/app.py": "src/app.py",
        "src/train_gru.py": "src/train_gru.py",
        "runs/detect/visdrone/weights/best.pt": "runs/detect/visdrone/weights/best.pt",
        "runs/gru_best.pt": "runs/gru_best.pt",
        "pyproject.toml": "pyproject.toml",
        "README.md": "README.md",
    }
    
    with zipfile.ZipFile(release_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, dest in targets.items():
            src_path = Path(src)
            if src_path.exists():
                zf.write(src_path, f"droneai-{version}/{dest}")

        run_bat = "@echo off\nuv sync\nuv run python src/app.py\n"
        zf.writestr(f"droneai-{version}/run.bat", run_bat)

        run_sh = "#!/bin/bash\nuv sync\nuv run python src/app.py\n"
        zf.writestr(f"droneai-{version}/run.sh", run_sh)

if __name__ == "__main__":
    create_package()
