"""Elo estimation against fixed-rating anchors (random agent = 0 Elo).

Win rates are clamped to [1/(2n), 1 - 1/(2n)] before use: n games can only
resolve a finite rating gap, so a 100% score against an anchor yields a
bounded estimate instead of infinity.
"""

from __future__ import annotations

import math


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def clamp_score(score: float, n_games: int) -> float:
    lo = 1.0 / (2 * n_games)
    return min(max(score, lo), 1.0 - lo)


def elo_diff_from_score(score: float, n_games: int) -> float:
    """Single-opponent rating difference implied by a match score."""
    p = clamp_score(score, n_games)
    return 400.0 * math.log10(p / (1.0 - p))


def mle_rating(results: list[tuple[float, float, int]],
               lo: float = -1500.0, hi: float = 4000.0) -> float:
    """Maximum-likelihood rating given [(opponent_elo, score_0to1, n_games)].

    The log-likelihood in R is concave; its derivative
    sum_i n_i * (p_i - E(R, elo_i)) is decreasing in R, so bisection works.
    """
    results = [(e, clamp_score(p, n), n) for e, p, n in results if n > 0]
    if not results:
        return 0.0

    def grad(r: float) -> float:
        return sum(n * (p - expected_score(r, e)) for e, p, n in results)

    for _ in range(80):
        mid = (lo + hi) / 2
        if grad(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
