from scripts.benchmark_solver_moves import (
    exact_mcnemar,
    optimal_moves,
    outcome_label,
    sample_positions,
    score_method,
    to_solver_moves,
    wilson_interval,
)

from az import game


def test_sample_positions_are_reproducible_and_nontrivial():
    first = sample_positions(10, seed=7, plies=(10, 14))
    second = sample_positions(10, seed=7, plies=(10, 14))

    assert [case.moves for case in first] == [case.moves for case in second]
    assert {case.target_ply for case in first} == {10, 14}
    assert all(not game.is_terminal(case.state) for case in first)
    assert all(not game.winning_moves(case.state) for case in first)
    assert all(len(game.legal_moves(case.state)) >= 2 for case in first)


def test_solver_notation_conversion_and_best_move_ties():
    assert to_solver_moves("036") == "147"
    scores = [None, -2, 3, 3, 0, None, 1]
    assert optimal_moves(scores) == {2, 3}
    assert outcome_label(scores) == "win"


def test_score_method_accepts_any_tied_optimal_move():
    result = score_method(
        choices=[0, 2, 6],
        oracle_moves=[{0, 1}, {2}, {4, 5}],
        elapsed_seconds=0.03,
    )
    assert result["correct"] == 2
    assert result["accuracy"] == 2 / 3
    assert result["milliseconds_per_position_batched"] == 10
    assert result["accuracy_ci_95"][0] < result["accuracy"] < result["accuracy_ci_95"][1]


def test_paired_exact_test_counts_discordant_cases():
    result = exact_mcnemar(
        first_choices=[0, 0, 0, 1],
        second_choices=[0, 1, 1, 1],
        oracle_moves=[{0}, {0}, {1}, {1}],
    )
    assert result["first_only_correct"] == 1
    assert result["second_only_correct"] == 1
    assert result["discordant"] == 2
    assert result["two_sided_exact_p"] == 1.0


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(82, 100)
    assert low < 0.82 < high
    assert (round(low, 3), round(high, 3)) == (0.733, 0.883)
