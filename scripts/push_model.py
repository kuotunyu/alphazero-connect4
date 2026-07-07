"""Publish the best network to the HF model repo.

Converts checkpoints/<preset>/best.pt to model.safetensors + config.json,
generates a model card with the real numbers from elo.csv, bundles the Elo
curve (and demo.gif if present), and uploads everything.

Usage:
    python scripts/push_model.py --ckpt-dir checkpoints/full \
        [--repo-id steven0226/alphazero-connect4] [--assets assets]

Requires a write-scoped HF token (HF_TOKEN env var or `hf auth login`).
"""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import torch


def load_history(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def make_card(config: dict, history: list[dict], repo_id: str,
              anchor_elo: float, has_gif: bool) -> str:
    last = history[-1]
    total_games = sum(int(float(r["games"])) for r in history)
    hours = float(last["wallclock_s"]) / 3600
    gif = "![self-play demo](demo.gif)\n\n" if has_gif else ""
    return f"""---
license: mit
tags:
- reinforcement-learning
- alphazero
- connect-four
- self-play
- mcts
- pytorch
---

# AlphaZero-style Connect Four agent

{gif}Policy/value ResNet trained **from zero human knowledge** by self-play +
MCTS (AlphaZero-style) on the standard 6x7 Connect Four board.

Play against it here: [connect4-arena Space](https://huggingface.co/spaces/steven0226/connect4-arena)

## Model

| | |
|---|---|
| Architecture | Conv stem -> {config["blocks"]} residual blocks x {config["filters"]} filters -> policy head (7 logits, illegal columns masked) + value head (tanh) |
| Input | 3x6x7 planes: own stones / opponent stones / side-to-move |
| Parameters | ~{sum_params(config):,} |
| Format | `model.safetensors` + `config.json` (rebuild with `az.model.create_model`) |

## Training

- Self-play with the latest network: batched MCTS (lockstep games, one GPU
  forward per tick), Dirichlet root noise (alpha=1.0, eps=0.25), tau=1
  sampling for the first 10 plies.
- Replay buffer with horizontal-flip augmentation at sample time;
  loss = soft-target cross-entropy + value MSE, AdamW (lr 1e-3, wd 1e-4).
- Gating: a candidate replaces the published best only on >55% score over
  arena games vs. the current best.
- **{len(history)} iterations, {total_games:,} self-play games,
  {hours:.1f} h wallclock.**

## Evaluation (final iteration, real numbers)

Elo anchored at random agent = 0; pure-MCTS-200 anchor calibrated by ladder
matches at {anchor_elo:.0f} Elo.

| metric | value |
|---|---|
| Elo (MLE vs anchors) | **{float(last["elo"]):.0f}** |
| vs random agent (40 games) | {float(last["wr_random"]):.0%} |
| raw policy (no search) vs random | {float(last["wr_policy_random"]):.0%} |
| vs pure MCTS, 200 sims | {float(last["wr_mcts200"]):.0%} |

![elo curve](elo_curve.png)

## Usage

```python
from az.model import create_model  # az package: see the Space's bundled copy
import json, torch
from safetensors.torch import load_file

model = create_model(json.load(open("config.json")))
model.load_state_dict(load_file("model.safetensors"))
```

## Limitations

- Connect Four is solved (first player wins with perfect play); this agent
  is strong but **not perfect** — deep tactics beyond its search budget can
  beat it.
- Trained only for the standard 6x7 board.
- The 40-game gating/eval matches carry ~8% sampling noise per point; the
  Elo curve's fine structure is noisy even though the trend is real.
"""


def sum_params(config: dict) -> int:
    from az.model import create_model
    return sum(p.numel() for p in create_model(config).parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="steven0226/alphazero-connect4")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--dry-run", action="store_true",
                        help="stage files locally without uploading")
    args = parser.parse_args()

    from az.config import ANCHOR_MCTS200_ELO
    from safetensors.torch import save_file

    payload = torch.load(args.ckpt_dir / "best.pt", map_location="cpu",
                         weights_only=True)
    history = load_history(args.ckpt_dir / "elo.csv")

    staging = Path(tempfile.mkdtemp(prefix="hf-model-"))
    save_file(payload["model"], staging / "model.safetensors")
    (staging / "config.json").write_text(json.dumps(payload["config"]))
    shutil.copy(args.ckpt_dir / "elo_curve.png", staging / "elo_curve.png")
    shutil.copy(args.ckpt_dir / "elo.csv", staging / "elo.csv")
    gif = args.assets / "demo.gif"
    has_gif = gif.exists()
    if has_gif:
        shutil.copy(gif, staging / "demo.gif")
    (staging / "README.md").write_text(
        make_card(payload["config"], history, args.repo_id,
                  ANCHOR_MCTS200_ELO, has_gif), encoding="utf-8")

    print(f"staged: {[p.name for p in staging.iterdir()]}")
    if args.dry_run:
        print(f"dry run — files left in {staging}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=staging, repo_id=args.repo_id,
                      repo_type="model")
    print(f"pushed to https://huggingface.co/{args.repo_id}")
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
