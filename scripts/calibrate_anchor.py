"""One-off Elo calibration of the pure-MCTS anchor via a ladder.

random (0 Elo) <-> MCTS-16 <-> MCTS-64 <-> MCTS-200, adjacent rungs play a
match each; the rating differences add up to the anchor's absolute Elo.
Playing MCTS-200 directly against random would hit the ~100% win-rate clamp
and measure nothing.

The resulting value is frozen in az/config.py (ANCHOR_MCTS200_ELO) so Elo
stays comparable across training runs.

Usage: python scripts/calibrate_anchor.py [--games 200] [--seed 0]
"""

import argparse
import time

import numpy as np

from az.arena import MCTSAgent, RandomAgent, play_match
from az.elo import elo_diff_from_score
from az.mcts import RolloutEvaluator

LADDER = [16, 64, 200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    elo = 0.0
    prev, prev_name = RandomAgent(), "random(0)"
    for sims in LADDER:
        agent = MCTSAgent(RolloutEvaluator(rng), sims, temp_moves=6)
        t0 = time.time()
        res = play_match(agent, prev, args.games, rng)
        delta = elo_diff_from_score(res.score, args.games)
        elo += delta
        print(f"MCTS-{sims:<4} vs {prev_name:<10} "
              f"+{res.wins}={res.draws}-{res.losses}  score={res.score:.3f}  "
              f"delta={delta:+7.1f}  cumulative={elo:7.1f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        prev, prev_name = agent, f"MCTS-{sims}"

    print(f"\nANCHOR_MCTS200_ELO = {elo:.1f}")
    print("-> update az/config.py with this value")


if __name__ == "__main__":
    main()
