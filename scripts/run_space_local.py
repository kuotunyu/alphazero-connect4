"""Run the Space app locally against a training checkpoint.

Usage: python scripts/run_space_local.py [path/to/best.pt]
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
weights = sys.argv[1] if len(sys.argv) > 1 else ROOT / "checkpoints/smoke/best.pt"
os.environ.setdefault("AZ_LOCAL_WEIGHTS", str(weights))
runpy.run_path(str(ROOT / "space" / "app.py"), run_name="__main__")
