"""Evaluation matches against anchors, batched across games.

All agents expose a batched ``choose(states, move_numbers, rng)`` so that a
whole match advances in lockstep waves: each round, the games are grouped by
which agent is to move and that agent picks moves for its whole group at
once (one batched MCTS / one net forward / one loop of rollouts).

Anchor settings are frozen across iterations so Elo stays comparable. Both
sides sample tau=1 for the first few plies (``temp_moves``) — two
deterministic agents would otherwise replay one identical game 20 times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from az import game
from az.mcts import SearchTree, run_batched, sample_move


class RandomAgent:
    def choose(self, states, move_numbers, rng):
        return [
            moves[int(rng.integers(len(moves)))]
            for moves in (game.legal_moves(s) for s in states)
        ]


class PolicyAgent:
    """Raw policy argmax, no search — the cleanest 'is the net learning' probe."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def choose(self, states, move_numbers, rng):
        priors, _ = self.evaluator.evaluate_batch(states)
        return [int(np.argmax(p)) for p in priors]


class MCTSAgent:
    def __init__(self, evaluator, sims, c_puct=1.5, temp_moves=0):
        self.evaluator = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.temp_moves = temp_moves

    def choose(self, states, move_numbers, rng):
        trees = [SearchTree(s, self.c_puct, rng=rng) for s in states]
        run_batched(trees, self.evaluator, self.sims)
        return [
            sample_move(t.root_policy(), mn, self.temp_moves, rng)
            for t, mn in zip(trees, move_numbers)
        ]


@dataclass
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Score for agent A in [0, 1], draws worth half."""
        return (self.wins + 0.5 * self.draws) / max(1, self.n)


def play_match(agent_a, agent_b, n_games: int, rng) -> MatchResult:
    """A vs B, colors alternating: even game indices have A move first."""
    live = [
        {"state": game.INITIAL, "first": i % 2, "moves": 0}
        for i in range(n_games)
    ]
    agents = (agent_a, agent_b)
    wins = draws = losses = 0

    while live:
        # group by the agent to move, determined at round start: each game
        # advances exactly one ply per round
        groups = {0: [], 1: []}
        for g in live:
            groups[(g["first"] + g["moves"]) % 2].append(g)
        for side in (0, 1):
            if not groups[side]:
                continue
            cols = agents[side].choose(
                [g["state"] for g in groups[side]],
                [g["moves"] for g in groups[side]],
                rng,
            )
            for g, col in zip(groups[side], cols):
                g["state"] = game.play(g["state"], col)
                g["moves"] += 1
                if game.is_terminal(g["state"]):
                    if game.last_move_won(g["state"]):
                        if side == 0:
                            wins += 1
                        else:
                            losses += 1
                    else:
                        draws += 1
                    g["done"] = True
        live = [g for g in live if not g.get("done")]

    return MatchResult(wins, draws, losses)
