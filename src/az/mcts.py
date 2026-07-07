"""PUCT Monte-Carlo tree search with a pluggable, batch-friendly evaluator.

torch-free by design: the neural evaluator lives in :mod:`az.model`; this
module only needs numpy and the bitboard engine, so search logic is testable
without a GPU stack.

Perspective conventions (the #1 source of AlphaZero bugs — pinned here):

- ``Evaluator.evaluate_batch(states) -> (priors, values)`` where ``values``
  are from the perspective of **the player to move** at each state.
- A terminal node's value is likewise from the perspective of the player to
  move there: ``-1`` if the previous move won (they just lost), ``0`` draw.
- ``Node.N/W`` live on the edges of the parent: ``W`` accumulates values
  from the **parent player's** perspective, so backup flips the sign once
  per level going up.
"""

from __future__ import annotations

import math

import numpy as np

from az import game

COLS = game.COLS


class Evaluator:
    """Interface: returns (priors [B,7] masked+normalized, values [B])."""

    def evaluate_batch(self, states):  # pragma: no cover - interface
        raise NotImplementedError


class UniformEvaluator(Evaluator):
    """Uniform priors over legal moves, value 0. Deterministic; for tests."""

    def evaluate_batch(self, states):
        priors = np.zeros((len(states), COLS), dtype=np.float32)
        for i, s in enumerate(states):
            legal = game.legal_mask(s)
            priors[i, legal] = 1.0 / legal.sum()
        return priors, np.zeros(len(states), dtype=np.float32)


class RolloutEvaluator(Evaluator):
    """Uniform priors + one random playout per state (pure-MCTS anchor)."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def evaluate_batch(self, states):
        priors = np.zeros((len(states), COLS), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)
        for i, s in enumerate(states):
            legal = game.legal_mask(s)
            priors[i, legal] = 1.0 / legal.sum()
            values[i] = self._rollout(s)
        return priors, values

    def _rollout(self, state):
        sign = 1.0  # +1 while the simulated player to move == original one
        while not game.is_terminal(state):
            moves = game.legal_moves(state)
            state = game.play(state, moves[int(self.rng.integers(len(moves)))])
            sign = -sign
        if game.last_move_won(state):
            return -sign  # player to move at the end has just lost
        return 0.0


class Node:
    __slots__ = ("state", "children", "expanded", "terminal", "tvalue",
                 "legal", "P", "N", "W")

    def __init__(self, state):
        self.state = state
        self.children = [None] * COLS
        self.expanded = False
        if game.last_move_won(state):
            self.terminal, self.tvalue = True, -1.0
        elif state[1] == game.FULL_MASK:
            self.terminal, self.tvalue = True, 0.0
        else:
            self.terminal, self.tvalue = False, 0.0
        self.legal = None
        self.P = None
        self.N = None
        self.W = None


class SearchTree:
    """One game's tree. Drives select / expand / backup; the caller supplies
    evaluations, which is what makes cross-game batching possible."""

    def __init__(self, state, c_puct=1.5, noise_frac=0.0, noise_alpha=1.0,
                 rng=None):
        assert not game.is_terminal(state), "cannot search a terminal state"
        self.root = Node(state)
        self.c_puct = c_puct
        self.noise_frac = noise_frac
        self.noise_alpha = noise_alpha
        self.rng = rng if rng is not None else np.random.default_rng()

    def root_visit_count(self) -> int:
        if not self.root.expanded:
            return 0
        return int(self.root.N.sum())

    def select(self):
        """Descend by PUCT until an unexpanded or terminal node.

        Returns (path, node) where path is [(parent, action), ...].
        """
        node = self.root
        path = []
        while node.expanded and not node.terminal:
            scores = self._puct(node)
            action = int(np.argmax(scores))
            child = node.children[action]
            if child is None:
                child = Node(game.play(node.state, action))
                node.children[action] = child
            path.append((node, action))
            node = child
        return path, node

    def _puct(self, node):
        N, W = node.N, node.W
        q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)
        u = self.c_puct * node.P * math.sqrt(max(1.0, N.sum())) / (1.0 + N)
        scores = q + u
        scores[~node.legal] = -np.inf
        return scores

    def expand(self, node, priors):
        node.legal = game.legal_mask(node.state)
        p = np.asarray(priors, dtype=np.float32).copy()
        p[~node.legal] = 0.0
        total = p.sum()
        if total <= 0:  # degenerate evaluator output: fall back to uniform
            p[node.legal] = 1.0
            total = p.sum()
        p /= total
        if node is self.root and self.noise_frac > 0:
            noise = np.zeros(COLS, dtype=np.float32)
            k = int(node.legal.sum())
            noise[node.legal] = self.rng.dirichlet([self.noise_alpha] * k)
            p = (1 - self.noise_frac) * p + self.noise_frac * noise
        node.P = p
        node.N = np.zeros(COLS, dtype=np.float32)
        node.W = np.zeros(COLS, dtype=np.float32)
        node.expanded = True

    def backup(self, path, leaf_value):
        """Propagate a leaf value (leaf player-to-move perspective) upward."""
        v = leaf_value
        for parent, action in reversed(path):
            v = -v  # now from the perspective of the player at `parent`
            parent.N[action] += 1
            parent.W[action] += v

    def root_policy(self) -> np.ndarray:
        """Normalized root visit counts (the training target pi)."""
        visits = self.root.N
        total = visits.sum()
        assert total > 0, "no simulations were run"
        return (visits / total).astype(np.float32)


class MCTS:
    """Single-game convenience runner (tests, arena anchors, the Space app).

    Self-play uses the same SearchTree through az.selfplay's batched pool.
    """

    def __init__(self, evaluator, sims, c_puct=1.5, noise_frac=0.0,
                 noise_alpha=1.0, rng=None):
        self.evaluator = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.noise_frac = noise_frac
        self.noise_alpha = noise_alpha
        self.rng = rng if rng is not None else np.random.default_rng()

    def policy(self, state) -> np.ndarray:
        """Search and return the normalized root visit distribution."""
        return self.search_tree(state).root_policy()

    def search_tree(self, state) -> SearchTree:
        tree = SearchTree(state, self.c_puct, self.noise_frac,
                          self.noise_alpha, self.rng)
        while tree.root_visit_count() < self.sims:
            path, node = tree.select()
            if node.terminal:
                tree.backup(path, node.tvalue)
                continue
            priors, values = self.evaluator.evaluate_batch([node.state])
            tree.expand(node, priors[0])
            tree.backup(path, float(values[0]))
        return tree

    def best_move(self, state) -> int:
        return int(np.argmax(self.policy(state)))


def run_batched(trees, evaluator, sims):
    """Run many SearchTrees in lockstep until every root has ``sims`` visits.

    One leaf per tree per tick; all pending leaves share one evaluator call.
    """
    live = [t for t in trees if t.root_visit_count() < sims]
    while live:
        pending = []
        for tree in live:
            path, node = tree.select()
            if node.terminal:
                tree.backup(path, node.tvalue)
            else:
                pending.append((tree, path, node))
        if pending:
            priors, values = evaluator.evaluate_batch(
                [node.state for _, _, node in pending]
            )
            for i, (tree, path, node) in enumerate(pending):
                tree.expand(node, priors[i])
                tree.backup(path, float(values[i]))
        live = [t for t in live if t.root_visit_count() < sims]


def sample_move(pi: np.ndarray, move_number: int, temp_moves: int, rng) -> int:
    """Temperature schedule: sample tau=1 for the opening, then argmax."""
    if move_number < temp_moves:
        return int(rng.choice(COLS, p=pi / pi.sum()))
    return int(np.argmax(pi))
