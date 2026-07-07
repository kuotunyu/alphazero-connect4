"""Render one self-play game of the trained agent to an animated GIF.

Usage:
    python scripts/make_gif.py --ckpt checkpoints/smoke/best.pt \
        [--out assets/demo.gif] [--sims 200] [--seed 1]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from az import game
from az.mcts import MCTS, sample_move
from az.model import NetEvaluator, create_model
from az.viz import render_board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("assets/demo.gif"))
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = create_model(payload["config"])
    model.load_state_dict(payload["model"])
    evaluator = NetEvaluator(model, device="cpu")
    rng = np.random.default_rng(args.seed)
    mcts = MCTS(evaluator, args.sims, noise_frac=0.0, rng=rng)

    state = game.INITIAL
    frames = [render_board(state, title="AlphaZero self-play")]
    move_number = 0
    while not game.is_terminal(state):
        pi = mcts.policy(state)
        col = sample_move(pi, move_number, temp_moves=6, rng=rng)
        state = game.play(state, col)
        move_number += 1
        mover = "🔴" if move_number % 2 == 1 else "🟡"
        frames.append(render_board(state, last_move=col,
                                   title=f"move {move_number}: column {col + 1}"))
    result = ("red wins" if game.last_move_won(state) and move_number % 2 == 1
              else "yellow wins" if game.last_move_won(state) else "draw")
    frames[-1] = render_board(state, title=f"{result} in {move_number} moves")
    frames += [frames[-1]] * 4  # hold the final position

    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(args.out, save_all=True, append_images=images[1:],
                   duration=650, loop=0, optimize=True)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size_kb:.0f} KB, {result})")


if __name__ == "__main__":
    main()
