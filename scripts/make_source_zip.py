"""Zip just the source needed to train on Colab (no .venv/checkpoints/.git).

Upload the result to the Drive root (MyDrive/alphazero-connect4-src.zip) as
the fallback source for alphazero_connect4_colab_train.ipynb when the GitHub repo isn't
published yet. Re-run this after any local code change and re-upload.

Usage: python scripts/make_source_zip.py [out.zip]
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE_FILES = ["pyproject.toml"]
INCLUDE_DIRS = ["src", "tests", "scripts"]
EXCLUDE_NAMES = {"__pycache__", "alphazero_connect4.egg-info"}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_NAMES for part in path.parts)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "alphazero-connect4-src.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE_FILES:
            zf.write(ROOT / name, name)
        for d in INCLUDE_DIRS:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and not should_skip(p.relative_to(ROOT)):
                    zf.write(p, p.relative_to(ROOT).as_posix())

    names = zipfile.ZipFile(out).namelist()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(names)} files)")
    print("-> upload to Google Drive: MyDrive/alphazero-connect4-src.zip")


if __name__ == "__main__":
    main()
