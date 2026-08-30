"""Deploy the Gradio arena to HF Spaces.

Rebuilds the gitignored space/az deployment bundle from the canonical src/az
package (the Space is self-contained, with no git dependency), then uploads
the space/ folder. The model revision is pinned in space/model_assets.py.

Usage:
    python scripts/deploy_space.py [--repo-id steven0226/connect4-arena] [--dry-run]

Requires a write-scoped HF token (HF_TOKEN env var or `hf auth login`).
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def bundle_az(
    source: Path = ROOT / "src" / "az",
    destination: Path = SPACE / "az",
) -> Path:
    """Replace the generated Space bundle with the canonical ``src/az``."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"bundled canonical {source} -> generated {destination}")
    return destination


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
