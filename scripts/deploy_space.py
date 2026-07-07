"""Deploy the Gradio arena to HF Spaces.

Bundles a copy of the az package into space/az (the Space is self-contained,
no git dependency), then uploads the space/ folder.

Usage:
    python scripts/deploy_space.py [--repo-id steven0226/connect4-arena] [--dry-run]

Requires a write-scoped HF token (HF_TOKEN env var or `hf auth login`).
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def bundle_az():
    dst = SPACE / "az"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "src" / "az", dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"bundled az package -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="steven0226/connect4-arena")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_az()
    if args.dry_run:
        print("dry run — space/ ready, not uploading")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    api.upload_folder(
        folder_path=SPACE, repo_id=args.repo_id, repo_type="space",
        ignore_patterns=["__pycache__/*", "flagged/*", ".gradio/*"],
    )
    print(f"pushed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
