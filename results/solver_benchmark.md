# Exact-solver move benchmark

- Published model: `steven0226/alphazero-connect4` (6 blocks × 96 filters)
- Dataset: 100 deterministic positions, seed `20260726`, plies `10, 14, 18, 22, 26`
- Filters: non-terminal, at least two legal moves, no immediate winning move
- Oracle: `connect-four-ai==1.0.0`, exact score for every legal move
- Metric: chosen column is any move tied for the oracle's maximum score; latency is batched CPU throughput on the evaluation machine

| Method | Exact-best move | Accuracy [Wilson 95% CI] | Batched CPU ms/position |
|---|---:|---:|---:|
| Policy only | 74/100 | **74.0%** [64.6%, 81.6%] | 0.5 |
| MCTS-50 | 77/100 | **77.0%** [67.8%, 84.2%] | 18.4 |
| MCTS-200 | 80/100 | **80.0%** [71.1%, 86.7%] | 65.0 |
| MCTS-800 | 82/100 | **82.0%** [73.3%, 88.3%] | 249.1 |

## Paired interpretation

Policy-only was uniquely correct on 3 positions; the largest MCTS budget was uniquely correct on 11. Two-sided exact McNemar `p=0.0574`.
The point estimate improves with search, but this 100-position sample is not sufficient to claim a statistically significant policy-to-largest-budget difference at α=0.05.

## Oracle outcome mix

- Win: 33; draw: 3; loss: 64

## Reproduce

```bash
python -m pip install -r requirements-oracle.txt
python scripts/benchmark_solver_moves.py --positions 100 --budgets 50 200 800
```

Position move strings, all oracle scores, model choices, and timings are preserved in `solver_benchmark.json`.
