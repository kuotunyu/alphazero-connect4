"""Training configuration with SMOKE (local dev) and FULL (Colab A100) presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Elo of the pure-MCTS anchor (RolloutEvaluator, 200 sims) relative to the
# random agent at 0 Elo. Calibrated once with scripts/calibrate_anchor.py
# (200 games per rung, seed 0):
#   MCTS-16  vs random   score 0.930  -> +449.4
#   MCTS-64  vs MCTS-16  score 0.880  -> +346.1
#   MCTS-200 vs MCTS-64  score 0.752  -> +193.2
# Frozen so Elo stays comparable across runs.
ANCHOR_MCTS200_ELO = 988.6


@dataclass
class Config:
    # model
    blocks: int = 6
    filters: int = 96
    # self-play
    games_per_iter: int = 128
    parallel: int = 128
    sims: int = 160
    c_puct: float = 1.5
    noise_frac: float = 0.25
    noise_alpha: float = 1.0
    temp_moves: int = 10
    autocast: bool = False  # fp16 inference during self-play (A100/4090)
    # replay buffer
    buffer_size: int = 150_000
    # optimisation
    batch_size: int = 256
    steps_per_iter: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # arena / evaluation
    arena_games: int = 40       # per anchor, colors alternate
    arena_sims: int = 160       # agent search budget in arena play
    arena_temp_moves: int = 6   # tau=1 opening plies to de-duplicate games
    anchor_sims: int = 200      # pure-MCTS anchor budget (spec)
    anchor_elo: float = ANCHOR_MCTS200_ELO
    raw_policy_games: int = 40  # no-search policy vs random (learning signal)
    gate_threshold: float = 0.55
    # run length
    iterations: int = 150

    def to_dict(self) -> dict:
        return asdict(self)


SMOKE = Config(
    blocks=3,
    filters=64,
    games_per_iter=24,
    parallel=24,
    sims=64,
    buffer_size=20_000,
    batch_size=128,
    steps_per_iter=16,
    arena_games=20,
    arena_sims=64,
    iterations=40,
)

FULL = Config()

PRESETS = {"smoke": SMOKE, "full": FULL}
