"""Render the Elo curve PNG from a training run's elo.csv.

Usage: python scripts/plot_elo.py checkpoints/smoke/elo.csv [out.png]
"""

import csv
import sys
from pathlib import Path

from az.viz import plot_elo


def load_history(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {key: float(value) for key, value in row.items()}
            parsed["iteration"] = int(parsed["iteration"])
            parsed["gated"] = int(parsed["gated"])
            rows.append(parsed)
    return rows


def main():
    csv_path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_name(
        "elo_curve.png")
    plot_elo(load_history(csv_path), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
