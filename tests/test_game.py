"""Bitboard engine tests: wins in all directions, draw, illegal moves,
immediate-win / must-block detection, flips, encoding. 20 cases."""

import numpy as np
import pytest

from az import game
from az.game import (
    INITIAL,
    encode,
    flip_horizontal,
    from_moves,
    has_won,
    is_draw,
    is_terminal,
    last_move_won,
    legal_moves,
    opponent_winning_moves,
    play,
    ply,
    winning_moves,
)


def seq(*moves):
    return from_moves("".join(map(str, moves)))


def replay_no_premature_end(moves):
    """Replay, asserting the game only ends on the very last move."""
    state = INITIAL
    for i, col in enumerate(moves):
        assert not is_terminal(state), f"terminal before move {i}"
        state = play(state, col)
    return state


def grid_to_state(rows, to_move):
    """Build a state from 6 strings of 7 chars, row 0 = TOP.

    'X' = first player, 'O' = second player, '.' = empty. Independent
    construction path used to cross-check play()/encode().
    """
    first = second = 0
    for r_top, line in enumerate(rows):
        r = game.ROWS - 1 - r_top
        for c, ch in enumerate(line):
            bit = 1 << (c * game.H1 + r)
            if ch == "X":
                first |= bit
            elif ch == "O":
                second |= bit
    mask = first | second
    position = first if to_move == "X" else second
    return position, mask


# --- 1-2: legal moves ------------------------------------------------------

def test_01_empty_board_all_columns_legal():
    assert legal_moves(INITIAL) == list(range(7))
    assert ply(INITIAL) == 0


def test_02_full_column_removed_and_raises():
    state = from_moves("333333")  # alternate colors fill column 3
    assert not is_terminal(state)
    assert legal_moves(state) == [0, 1, 2, 4, 5, 6]
    with pytest.raises(ValueError):
        play(state, 3)


# --- 3-8: wins in every direction ------------------------------------------

def test_03_horizontal_win_bottom_row():
    state = replay_no_premature_end([0, 0, 1, 1, 2, 2, 3])
    assert last_move_won(state) and is_terminal(state) and not is_draw(state)


def test_04_vertical_win():
    state = replay_no_premature_end([3, 2, 3, 2, 3, 2, 3])
    assert last_move_won(state) and is_terminal(state)


def test_05_diagonal_up_right_win():
    # P1 stones at (c0,r0),(c1,r1),(c2,r2),(c3,r3)
    state = replay_no_premature_end([0, 1, 1, 2, 2, 3, 2, 3, 3, 5, 3])
    assert last_move_won(state) and is_terminal(state)


def test_06_diagonal_down_right_win():
    # P1 stones at (c3,r0),(c2,r1),(c1,r2),(c0,r3)
    state = replay_no_premature_end([3, 2, 2, 1, 1, 0, 1, 0, 0, 5, 0])
    assert last_move_won(state) and is_terminal(state)


def test_07_horizontal_win_row_one():
    # catches row-offset bugs: the four sits one row above the bottom
    state = replay_no_premature_end([1, 2, 3, 4, 1, 6, 2, 6, 3, 5, 4])
    assert last_move_won(state) and is_terminal(state)


def test_08_vertical_win_ending_at_top_row():
    # exercises the sentinel boundary: P1 completes rows 2-5 of column 0
    state = replay_no_premature_end([6, 0, 5, 0, 0, 6, 0, 5, 0, 6, 0])
    assert last_move_won(state) and is_terminal(state)
    assert 0 not in legal_moves(state)  # the winning column is now full


# --- 9-10: full board — win beats draw / true draw --------------------------

def test_09_win_on_final_stone_beats_draw():
    # 41 stones, no four anywhere, one empty cell (col 4, top row);
    # the forced 42nd stone completes O's four on the top row.
    rows = [
        "XXOO.OX",
        "XOXXXOO",
        "OXOOOXO",
        "XOXXOXO",
        "OXOOXOX",
        "XOXXOXX",
    ]
    state = grid_to_state(rows, to_move="O")
    position, mask = state
    first, second = position ^ mask, position  # X derived, O to move
    assert bin(mask).count("1") == 41
    assert not has_won(first) and not has_won(second)  # win-free before
    assert legal_moves(state) == [4]
    final = play(state, 4)
    assert last_move_won(final)
    assert not is_draw(final)  # win takes precedence on a full board


def test_10_draw_full_board_no_winner():
    # alternating 3+3 column blocks: provably four-free when complete
    pair = lambda a, b: [a, b] * 3 + [b, a] * 3
    moves = (
        pair(0, 1)
        + pair(2, 3)
        + [6, 5, 6, 5, 6, 5, 4, 6, 4, 6, 4, 6, 5, 4, 5, 4, 5, 4]
    )
    state = replay_no_premature_end(moves)
    assert ply(state) == 42
    assert is_draw(state) and is_terminal(state) and not last_move_won(state)
    assert legal_moves(state) == []


# --- 11-13: non-wins, wrap safety, illegal input -----------------------------

def test_11_three_in_a_row_is_not_a_win():
    state = from_moves("001122")
    assert not is_terminal(state) and not last_move_won(state)


def test_12_no_wrap_across_columns():
    # P1: top of column 0 (r3,r4,r5) + bottom of column 1. In a layout
    # without the sentinel row these bits are adjacent and fake a vertical.
    state = replay_no_premature_end([1, 0, 5, 0, 3, 0, 0, 5, 0, 3, 0])
    assert not is_terminal(state)


def test_13_out_of_range_column_raises():
    with pytest.raises(ValueError):
        play(INITIAL, 7)
    with pytest.raises(ValueError):
        play(INITIAL, -1)


# --- 14: state invariants ----------------------------------------------------

def test_14_play_invariants():
    rng = np.random.default_rng(0)
    state = INITIAL
    for step in range(30):
        moves = legal_moves(state)
        if is_terminal(state):
            break
        col = int(rng.choice(moves))
        new = play(state, col)
        assert new[0] & ~new[1] == 0  # position is a subset of mask
        assert new[1].bit_count() == state[1].bit_count() + 1
        assert new[0] == state[0] ^ state[1]  # next player's stones unchanged
        state = new


# --- 15-16: horizontal flip --------------------------------------------------

def test_15_flip_involution_and_mirror():
    state = from_moves("0112")
    assert flip_horizontal(flip_horizontal(state)) == state
    assert flip_horizontal(from_moves("0")) == from_moves("6")
    assert flip_horizontal(INITIAL) == INITIAL


def test_16_flip_matches_plane_flip():
    state = from_moves("331245")  # asymmetric position
    np.testing.assert_array_equal(
        encode(flip_horizontal(state)), encode(state)[:, :, ::-1]
    )


# --- 17: exact plane encoding ------------------------------------------------

def test_17_encode_planes_exact():
    # after "34": P1 stone at col 3, P2 stone at col 4, P1 to move
    planes = encode(from_moves("34"))
    assert planes.shape == (3, 6, 7) and planes.dtype == np.float32
    expected0 = np.zeros((6, 7), dtype=np.float32)
    expected0[5, 3] = 1.0  # row 5 = bottom row in top-first convention
    expected1 = np.zeros((6, 7), dtype=np.float32)
    expected1[5, 4] = 1.0
    np.testing.assert_array_equal(planes[0], expected0)  # to-move = P1
    np.testing.assert_array_equal(planes[1], expected1)
    np.testing.assert_array_equal(planes[2], np.ones((6, 7), dtype=np.float32))

    # after "3": P2 to move, so plane0 = P2's (empty), plane2 = zeros
    planes = encode(from_moves("3"))
    assert planes[0].sum() == 0
    assert planes[1][5, 3] == 1.0
    np.testing.assert_array_equal(planes[2], np.zeros((6, 7), dtype=np.float32))


# --- 18-19: immediate-win and must-block detection ---------------------------

def test_18_winning_moves_detection():
    # open-ended three on the bottom row: wins on both sides
    assert winning_moves(from_moves("112233")) == [0, 4]
    # three with one side blocked by the wall
    assert winning_moves(from_moves("001122")) == [3]
    # vertical three
    assert winning_moves(from_moves("343434")) == [3]
    assert winning_moves(INITIAL) == []


def test_19_must_block_detection():
    # P2 to move; P1 threatens to complete c0-c3 on the bottom row at col 3
    state = from_moves("00112")
    assert winning_moves(state) == []  # P2 itself has no win
    assert opponent_winning_moves(state) == [3]
    blocked = play(state, 3)
    assert not is_terminal(blocked)
    assert winning_moves(blocked) == []  # threat is gone for P1


# --- 20: exact bit layout ----------------------------------------------------

def test_20_replay_exact_bitboards():
    # moves 3,3,4,1 -> P1: bits 21, 28; P2: bits 22, 7; P1 to move
    position, mask = from_moves("3341")
    assert position == (1 << 21) | (1 << 28)
    assert mask == (1 << 21) | (1 << 22) | (1 << 28) | (1 << 7)
    assert ply((position, mask)) == 4
