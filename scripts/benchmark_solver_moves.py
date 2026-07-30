"""Benchmark the published agent against an exact Connect Four solver.

The benchmark samples deterministic, non-terminal positions that do not have
an immediate winning move.  For each position, ``connect-four-ai`` supplies
the exact score of every legal move; a model move counts as correct when it is
one of the maximum-scoring moves.

This is an optional, CPU-friendly evaluation.  It intentionally lives outside
the core project dependencies because the solver currently requires Python
3.13 or newer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from az import game  # noqa: E402
from az.mcts import SearchTree, run_batched  # noqa: E402


DEFAULT_REPO = "steven0226/alphazero-connect4"
DEFAULT_PLIES = (10, 14, 18, 22, 26)
DEFAULT_BUDGETS = (50, 200, 800)


@dataclass(frozen=True)
class PositionCase:
    moves: str
    state: tuple[int, int]
    target_ply: int


def _safe_continuations(state: tuple[int, int]) -> list[int]:
    """Legal moves that do not finish the game immediately."""
    return [
        move
        for move in game.legal_moves(state)
        if not game.last_move_won(game.play(state, move))
    ]


def sample_positions(
    total: int,
    seed: int,
    plies: tuple[int, ...] = DEFAULT_PLIES,
) -> list[PositionCase]:
    """Create reproducible, tactically non-trivial positions.

    An equal number of positions is drawn from each requested ply when
    possible.  Random trajectories are restricted to non-winning moves so
    every target depth is reachable, and target states with an immediate win
    are excluded.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not plies or any(ply < 1 or ply >= 42 for ply in plies):
        raise ValueError("plies must be between 1 and 41")

    rng = random.Random(seed)
    target_counts = {
        ply: total // len(plies) + (index < total % len(plies))
        for index, ply in enumerate(plies)
    }
    cases: list[PositionCase] = []
    seen: set[tuple[int, int]] = set()

    for target_ply, wanted in target_counts.items():
        accepted = 0
        attempts = 0
        while accepted < wanted:
            attempts += 1
            if attempts > max(10_000, wanted * 2_000):
                raise RuntimeError(
                    f"could not sample {wanted} positions at ply {target_ply}"
                )

            state = game.INITIAL
            moves: list[str] = []
            for _ in range(target_ply):
                choices = _safe_continuations(state)
                if not choices:
                    break
                move = rng.choice(choices)
                moves.append(str(move))
                state = game.play(state, move)

            if game.ply(state) != target_ply:
                continue
            if state in seen or game.is_terminal(state):
                continue
            if len(game.legal_moves(state)) < 2 or game.winning_moves(state):
                continue

            seen.add(state)
            cases.append(PositionCase("".join(moves), state, target_ply))
            accepted += 1

    return cases


def to_solver_moves(zero_indexed_moves: str) -> str:
    """Convert this repo's 0-indexed notation to the solver's 1-indexed form."""
    return "".join(str(int(move) + 1) for move in zero_indexed_moves)


def optimal_moves(scores: list[int | None]) -> set[int]:
    """Return all legal columns tied for the exact best score."""
    legal_scores = [(move, score) for move, score in enumerate(scores) if score is not None]
    if not legal_scores:
        raise ValueError("solver returned no legal move scores")
    best = max(score for _, score in legal_scores)
    return {move for move, score in legal_scores if score == best}


def outcome_label(scores: list[int | None]) -> str:
    """Classify the solved position by the best achievable outcome."""
    best = max(score for score in scores if score is not None)
    if best > 0:
        return "win"
    if best < 0:
        return "loss"
    return "draw"


def _load_published_evaluator(repo_id: str):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from az.model import NetEvaluator, create_model

    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = create_model(config)
    model.load_state_dict(load_file(weights_path))
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    return NetEvaluator(model, device="cpu"), config


def _policy_moves(evaluator, states: list[tuple[int, int]]) -> tuple[list[int], float]:
    started = time.perf_counter()
    priors, _ = evaluator.evaluate_batch(states)
    elapsed = time.perf_counter() - started
    return np.argmax(priors, axis=1).astype(int).tolist(), elapsed


def _mcts_moves(
    evaluator,
    states: list[tuple[int, int]],
    simulations: int,
    seed: int,
) -> tuple[list[int], float]:
    trees = [
        SearchTree(state, c_puct=1.5, rng=np.random.default_rng(seed + index))
        for index, state in enumerate(states)
    ]
    started = time.perf_counter()
    run_batched(trees, evaluator, simulations)
    elapsed = time.perf_counter() - started
    moves = [int(np.argmax(tree.root_policy())) for tree in trees]
    return moves, elapsed


def score_method(
    choices: list[int],
    oracle_moves: list[set[int]],
    elapsed_seconds: float,
) -> dict:
    hits = [choice in best for choice, best in zip(choices, oracle_moves)]
    count = len(hits)
    accuracy = sum(hits) / count
    ci_low, ci_high = wilson_interval(sum(hits), count)
    return {
        "correct": sum(hits),
        "positions": count,
        "accuracy": accuracy,
        "accuracy_ci_95": [ci_low, ci_high],
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_position_batched": 1000 * elapsed_seconds / count,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("require 0 <= successes <= total and total > 0")
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def exact_mcnemar(
    first_choices: list[int],
    second_choices: list[int],
    oracle_moves: list[set[int]],
) -> dict:
    """Two-sided exact McNemar test for paired move-correctness outcomes."""
    first_only = second_only = 0
    for first, second, best in zip(first_choices, second_choices, oracle_moves):
        first_hit = first in best
        second_hit = second in best
        first_only += first_hit and not second_hit
        second_only += second_hit and not first_hit
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Exact-solver move benchmark",
        "",
        (
            f"- Published model: `{result['model_repo']}` "
            f"({result['model_config']['blocks']} blocks × "
            f"{result['model_config']['filters']} filters)"
        ),
        (
            f"- Dataset: {result['positions']} deterministic positions, "
            f"seed `{result['seed']}`, plies "
            f"`{', '.join(map(str, result['plies']))}`"
        ),
        "- Filters: non-terminal, at least two legal moves, no immediate winning move",
        "- Oracle: `connect-four-ai==1.0.0`, exact score for every legal move",
        (
            "- Metric: chosen column is any move tied for the oracle's maximum "
            "score; latency is batched CPU throughput on the evaluation machine"
        ),
        "",
        "| Method | Exact-best move | Accuracy [Wilson 95% CI] | Batched CPU ms/position |",
        "|---|---:|---:|---:|",
    ]
    for name, values in result["methods"].items():
        label = "Policy only" if name == "policy" else f"MCTS-{name.removeprefix('mcts_')}"
        ci_low, ci_high = values["accuracy_ci_95"]
        lines.append(
            f"| {label} | {values['correct']}/{values['positions']} | "
            f"**{values['accuracy']:.1%}** [{ci_low:.1%}, {ci_high:.1%}] | "
            f"{values['milliseconds_per_position_batched']:.1f} |"
        )

    outcomes = result["oracle_outcomes"]
    paired = result["paired_comparisons"]["policy_vs_largest_mcts"]
    lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            (
                f"Policy-only was uniquely correct on {paired['first_only_correct']} "
                f"positions; the largest MCTS budget was uniquely correct on "
                f"{paired['second_only_correct']}. Two-sided exact McNemar "
                f"`p={paired['two_sided_exact_p']:.4f}`."
            ),
            (
                "The point estimate improves with search, but this 100-position "
                "sample is not sufficient to claim a statistically significant "
                "policy-to-largest-budget difference at α=0.05."
            ),
            "",
            "## Oracle outcome mix",
            "",
            (
                f"- Win: {outcomes.get('win', 0)}; draw: "
                f"{outcomes.get('draw', 0)}; loss: {outcomes.get('loss', 0)}"
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m pip install -r requirements-oracle.txt",
            (
                "python scripts/benchmark_solver_moves.py "
                "--positions 100 --budgets 50 200 800"
            ),
            "```",
            "",
            (
                "Position move strings, all oracle scores, model choices, and "
                "timings are preserved in `solver_benchmark.json`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--plies", type=int, nargs="+", default=DEFAULT_PLIES)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--model-repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("all MCTS budgets must be positive")

    try:
        from connect_four_ai import Position, Solver
    except ImportError as exc:
        raise SystemExit(
            "Install the optional oracle dependencies with "
            "`python -m pip install -r requirements-oracle.txt` using Python 3.13+."
        ) from exc

    cases = sample_positions(args.positions, args.seed, tuple(args.plies))
    solver = Solver()
    score_rows: list[list[int | None]] = []
    oracle: list[set[int]] = []
    for case in cases:
        scores = list(
            solver.get_all_move_scores(
                Position.from_moves(to_solver_moves(case.moves))
            )
        )
        score_rows.append(scores)
        oracle.append(optimal_moves(scores))

    evaluator, model_config = _load_published_evaluator(args.model_repo)
    states = [case.state for case in cases]
    policy_choices, policy_elapsed = _policy_moves(evaluator, states)
    methods = {
        "policy": score_method(policy_choices, oracle, policy_elapsed),
    }
    choices_by_method = {"policy": policy_choices}

    for budget in args.budgets:
        choices, elapsed = _mcts_moves(
            evaluator, states, budget, args.seed + budget * 10_000
        )
        key = f"mcts_{budget}"
        methods[key] = score_method(choices, oracle, elapsed)
        choices_by_method[key] = choices

    largest_budget = max(args.budgets)
    paired_comparison = exact_mcnemar(
        choices_by_method["policy"],
        choices_by_method[f"mcts_{largest_budget}"],
        oracle,
    )
    paired_comparison["first"] = "policy"
    paired_comparison["second"] = f"mcts_{largest_budget}"

    rows = []
    for index, (case, scores, best) in enumerate(zip(cases, score_rows, oracle)):
        rows.append(
            {
                "moves_zero_indexed": case.moves,
                "moves_solver_one_indexed": to_solver_moves(case.moves),
                "ply": case.target_ply,
                "oracle_scores": scores,
                "oracle_optimal_moves_zero_indexed": sorted(best),
                "oracle_outcome": outcome_label(scores),
                "choices_zero_indexed": {
                    name: choices[index] for name, choices in choices_by_method.items()
                },
            }
        )

    result = {
        "schema_version": 1,
        "model_repo": args.model_repo,
        "model_config": model_config,
        "positions": len(cases),
        "seed": args.seed,
        "plies": list(args.plies),
        "budgets": list(args.budgets),
        "position_filters": [
            "non_terminal",
            "at_least_two_legal_moves",
            "no_immediate_winning_move",
        ],
        "oracle": {
            "package": "connect-four-ai",
            "version": "1.0.0",
            "metric": "choice_is_any_maximum_exact_score_move",
        },
        "oracle_outcomes": dict(
            Counter(outcome_label(scores) for scores in score_rows)
        ),
        "methods": methods,
        "paired_comparisons": {
            "policy_vs_largest_mcts": paired_comparison,
        },
        "cases": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "solver_benchmark.json"
    md_path = args.output_dir / "solver_benchmark.md"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")

    print(render_markdown(result))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
