"""Batched self-play: many games in lockstep, one leaf per game per tick,
all leaves evaluated in a single evaluator call (one GPU forward per tick).

Each tree contributes at most one in-flight leaf per tick, so no virtual
loss is needed. Finished games are immediately replaced to keep the
evaluation batch full until the target game count is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from az import game
from az.mcts import SearchTree, sample_move


@dataclass
class _Live:
    state: tuple
    tree: SearchTree
    move_number: int = 0
    records: list = field(default_factory=list)  # [(state, pi), ...]
    moves: list = field(default_factory=list)


@dataclass
class SelfPlayResult:
    games: list  # per game: list of (state, pi, z) samples
    moves_per_game: list  # per game: list of played columns

    @property
    def samples(self):
        return [s for g in self.games for s in g]


def _finalize(live: _Live) -> list:
    """Turn a finished game's records into (state, pi, z) samples.

    z is from the perspective of the player to move at each recorded state.
    """
    final = live.state
    n = len(live.records)
    if game.last_move_won(final):
        # winner made the last move, i.e. was to move at record n-1
        return [
            (s, pi, 1.0 if (n - 1 - j) % 2 == 0 else -1.0)
            for j, (s, pi) in enumerate(live.records)
        ]
    return [(s, pi, 0.0) for (s, pi) in live.records]


def play_games(
    evaluator,
    n_games: int,
    sims: int,
    *,
    parallel: int = 32,
    c_puct: float = 1.5,
    noise_frac: float = 0.25,
    noise_alpha: float = 1.0,
    temp_moves: int = 10,
    rng=None,
) -> SelfPlayResult:
    """Self-play ``n_games`` games, up to ``parallel`` concurrently."""
    rng = rng if rng is not None else np.random.default_rng()

    def new_game():
        return _Live(
            state=game.INITIAL,
            tree=SearchTree(game.INITIAL, c_puct, noise_frac, noise_alpha, rng),
        )

    started = min(n_games, parallel)
    live = [new_game() for _ in range(started)]
    finished_games, finished_moves = [], []

    while live:
        # --- one tick: one selection per game, batch all pending leaves ---
        pending = []  # (live game, path, node)
        for g in live:
            path, node = g.tree.select()
            if node.terminal:
                g.tree.backup(path, node.tvalue)
            else:
                pending.append((g, path, node))
        if pending:
            priors, values = evaluator.evaluate_batch(
                [node.state for _, _, node in pending]
            )
            for i, (g, path, node) in enumerate(pending):
                g.tree.expand(node, priors[i])
                g.tree.backup(path, float(values[i]))

        # --- games whose root reached the budget play one move ------------
        still_live = []
        for g in live:
            if g.tree.root_visit_count() < sims:
                still_live.append(g)
                continue
            pi = g.tree.root_policy()
            g.records.append((g.state, pi))
            move = sample_move(pi, g.move_number, temp_moves, rng)
            g.state = game.play(g.state, move)
            g.moves.append(move)
            g.move_number += 1
            if game.is_terminal(g.state):
                finished_games.append(_finalize(g))
                finished_moves.append(g.moves)
                if started < n_games:
                    started += 1
                    still_live.append(new_game())
            else:
                g.tree = SearchTree(g.state, c_puct, noise_frac,
                                    noise_alpha, rng)
                still_live.append(g)
        live = still_live

    return SelfPlayResult(games=finished_games, moves_per_game=finished_moves)
