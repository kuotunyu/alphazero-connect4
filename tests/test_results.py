from pathlib import Path

from scripts.summarize_full_run import load_rows, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_full_run_csv_is_complete_and_contiguous():
    rows = load_rows(ROOT / "results" / "elo_full.csv")
    assert len(rows) == 150
    assert rows[0]["iteration"] == 0
    assert rows[-1]["iteration"] == 149


def test_full_run_headline_numbers_are_reproducible():
    summary = summarize(load_rows(ROOT / "results" / "elo_full.csv"))
    assert summary["self_play_games"] == 19_200
    assert summary["gated_updates"] == 59
    assert round(summary["wallclock_hours"], 2) == 7.92
    assert summary["elo"]["last"] == 1625.6
    assert summary["elo"]["peak"] == 1625.6
    assert summary["vs_mcts200"]["all_iterations_at_least_90pct_from"] == 69
