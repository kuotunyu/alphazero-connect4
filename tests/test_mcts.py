"""MCTS behavior tests: with 200 simulations and no noise, the search must
find one-step wins and forced blocks, with both a value-free uniform
evaluator and the random-rollout evaluator (the pure-MCTS anchor)."""

import numpy as np

from az import game
from az.game import from_moves
from az.mcts import MCTS, RolloutEvaluator, UniformEvaluator
from az.selfplay import play_games

SIMS = 200


def search(state, evaluator=None, sims=SIMS):
    mcts = MCTS(evaluator or UniformEvaluator(), sims, noise_frac=0.0,
                rng=np.random.default_rng(0))
    return mcts.policy(state)


# --- one-step wins -----------------------------------------------------------

def test_takes_horizontal_win():
    # P1 to move, three on the bottom row: must play column 3
    pi = search(from_moves("001122"))
    assert int(np.argmax(pi)) == 3
    assert pi[3] >= 0.7


def test_takes_vertical_win():
    pi = search(from_moves("343434"))
    assert int(np.argmax(pi)) == 3
    assert pi[3] >= 0.7


def test_takes_win_with_rollout_evaluator():
    pi = search(from_moves("001122"), RolloutEvaluator(np.random.default_rng(0)))
    assert int(np.argmax(pi)) == 3
    assert pi[3] >= 0.7


# --- forced blocks -----------------------------------------------------------

def test_blocks_horizontal_threat():
    # P2 to move; P1 threatens to complete the bottom row at column 3
    state = from_moves("00112")
    assert game.opponent_winning_moves(state) == [3]
    pi = search(state)
    assert int(np.argmax(pi)) == 3
    assert pi[3] >= 0.7


def test_blocks_with_rollout_evaluator():
    pi = search(from_moves("00112"),
                RolloutEvaluator(np.random.default_rng(1)))
    assert int(np.argmax(pi)) == 3
    assert pi[3] >= 0.7


def test_prefers_win_over_block():
    # P1 can win at column 0; P2 threatens at column 4. Winning comes first.
    state = from_moves("010203")
    assert game.winning_moves(state) == [0]
    assert game.opponent_winning_moves(state) == [4]
    pi = search(state)
    assert int(np.argmax(pi)) == 0
    assert pi[0] >= 0.7


# --- mechanics ---------------------------------------------------------------

def test_policy_zero_on_illegal_columns():
    pi = search(from_moves("333333"), sims=64)
    assert pi[3] == 0.0
    assert abs(pi.sum() - 1.0) < 1e-5


def test_search_is_deterministic_without_noise():
    a = search(from_moves("34"), sims=100)
    b = search(from_moves("34"), sims=100)
    np.testing.assert_array_equal(a, b)


# --- batched self-play smoke -------------------------------------------------

def test_selfplay_pool_smoke():
    result = play_games(UniformEvaluator(), n_games=4, sims=32, parallel=4,
                        rng=np.random.default_rng(0))
    assert len(result.games) == 4
    assert len(result.moves_per_game) == 4
    for samples, moves in zip(result.games, result.moves_per_game):
        assert len(samples) == len(moves)
        # replay the recorded moves: must be legal and end exactly terminal
        state = game.INITIAL
        for col in moves:
            assert col in game.legal_moves(state)
            state = game.play(state, col)
        assert game.is_terminal(state)
        for j, (s, pi, z) in enumerate(samples):
            assert pi.shape == (7,)
            assert abs(pi.sum() - 1.0) < 1e-4
            assert (pi >= 0).all()
            assert z in (-1.0, 0.0, 1.0)
        zs = [z for _, _, z in samples]
        if game.last_move_won(state):
            # z alternates ply by ply and the final mover's z is +1
            assert zs[-1] == 1.0
            assert all(zs[k] == -zs[k + 1] for k in range(len(zs) - 1))
        else:
            assert all(z == 0.0 for z in zs)
