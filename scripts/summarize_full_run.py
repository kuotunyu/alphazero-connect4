"""Summarize the committed 150-iteration FULL training CSV.

The raw CSV is mirrored from the public Hugging Face model repository so the
headline README numbers can be independently recomputed without downloading
the model or rerunning self-play.

Usage:
    python scripts/summarize_full_run.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("results/elo_full.csv")
DEFAULT_MD = Path("results/full_run_summary.md")
DEFAULT_JSON = Path("results/full_run_summary.json")


def load_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int] = {
                key: float(value) for key, value in raw.items()
            }
            row["iteration"] = int(row["iteration"])
            row["games"] = int(row["games"])
            row["buffer_size"] = int(row["buffer_size"])
            row["gated"] = int(row["gated"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    expected = list(range(len(rows)))
    actual = [int(row["iteration"]) for row in rows]
    if actual != expected:
        raise ValueError("Iterations must be contiguous and zero-based")
    return rows


def summarize(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    peak_elo = max(float(row["elo"]) for row in rows)
    first_peak = next(
        int(row["iteration"]) for row in rows if float(row["elo"]) == peak_elo
    )
    tail = rows[-50:]
    sustained_90 = next(
        (
            int(row["iteration"])
            for index, row in enumerate(rows)
            if all(float(later["wr_mcts200"]) >= 0.9 for later in rows[index:])
        ),
        None,
    )
    return {
        "iterations": len(rows),
        "self_play_games": sum(int(row["games"]) for row in rows),
        "wallclock_hours": float(last["wallclock_s"]) / 3600,
        "gated_updates": sum(int(row["gated"]) for row in rows),
        "elo": {
            "first": float(first["elo"]),
            "last": float(last["elo"]),
            "peak": peak_elo,
            "first_peak_iteration": first_peak,
            "last_50_mean": statistics.mean(float(row["elo"]) for row in tail),
        },
        "vs_mcts200": {
            "first": float(first["wr_mcts200"]),
            "last": float(last["wr_mcts200"]),
            "last_50_mean": statistics.mean(
                float(row["wr_mcts200"]) for row in tail
            ),
            "all_iterations_at_least_90pct_from": sustained_90,
        },
        "policy_vs_random": {
            "first": float(first["wr_policy_random"]),
            "last": float(last["wr_policy_random"]),
        },
        "loss": {
            "policy_first": float(first["loss_policy"]),
            "policy_last": float(last["loss_policy"]),
            "value_first": float(first["loss_value"]),
            "value_last": float(last["loss_value"]),
        },
    }


def build_markdown(summary: dict[str, Any]) -> str:
    elo = summary["elo"]
    mcts = summary["vs_mcts200"]
    policy = summary["policy_vs_random"]
    loss = summary["loss"]
    return "\n".join(
        [
            "# AlphaZero Connect Four — FULL run summary",
            "",
            "- 來源：`results/elo_full.csv`（HF model repo 公開 `elo.csv` 的本地鏡像）",
            f"- {summary['iterations']} iterations，"
            f"{summary['self_play_games']:,} 局 self-play，"
            f"{summary['wallclock_hours']:.2f} 小時",
            f"- {summary['gated_updates']}/{summary['iterations']} iterations "
            "通過 gating",
            "",
            "| 指標 | 起點 | 終點 | 補充 |",
            "|---|---:|---:|---|",
            f"| Elo | {elo['first']:.1f} | **{elo['last']:.1f}** | "
            f"峰值 {elo['peak']:.1f} 首次出現在 iter {elo['first_peak_iteration']}；"
            f"末 50 iter 平均 {elo['last_50_mean']:.1f} |",
            f"| vs MCTS-200 | {mcts['first']:.1%} | **{mcts['last']:.1%}** | "
            f"末 50 iter 平均 {mcts['last_50_mean']:.1%}；"
            f"iter {mcts['all_iterations_at_least_90pct_from']} 起每輪 ≥90% |",
            f"| 裸 policy vs random | {policy['first']:.1%} | "
            f"**{policy['last']:.1%}** | 不依賴搜尋的學習訊號 |",
            f"| Policy loss | {loss['policy_first']:.4f} | "
            f"{loss['policy_last']:.4f} | |",
            f"| Value loss | {loss['value_first']:.4f} | "
            f"{loss['value_last']:.4f} | |",
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/summarize_full_run.py",
            "python scripts/plot_elo.py results/elo_full.csv assets/elo_curve_full.png",
            "```",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(load_rows(args.csv))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_markdown(summary), encoding="utf-8")
    args.out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
