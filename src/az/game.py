"""Connect Four (6x7) bitboard engine, Pascal Pons / John Tromp layout.

Board state is an immutable pair of Python ints ``(position, mask)``:

- ``position``: stones of the player **to move** (perspective flips every ply)
- ``mask``:     stones of both players

Bit index = ``col * 7 + row`` with **row 0 = bottom**. Each column spans 7
bits: rows 0..5 are playable, row 6 is a sentinel that stays empty so that
shift-based win detection never wraps across columns and the ``mask +
BOTTOM`` drop trick never carries into the next column.

The opponent's stones are always ``position ^ mask``.

Numpy plane encoding (:func:`encode`) uses the opposite row convention,
**row 0 = top**, matching how boards are rendered on screen.
"""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
H1 = ROWS + 1  # bits per column, incl. sentinel

BOTTOM = tuple(1 << (c * H1) for c in range(COLS))
TOP = tuple(1 << (c * H1 + ROWS - 1) for c in range(COLS))  # top playable cell
COL_BITS = (1 << H1) - 1  # 0b1111111
FULL_MASK = 0
for _c in range(COLS):
    FULL_MASK |= (COL_BITS >> 1) << (_c * H1)  # 42 playable bits

_SHIFTS = np.arange(COLS * H1, dtype=np.int64)

# state of a fresh game
INITIAL = (0, 0)


def ply(state: tuple[int, int]) -> int:
    """Number of stones on the board."""
    return state[1].bit_count()


def legal_moves(state: tuple[int, int]) -> list[int]:
    """Columns that can still be played."""
    mask = state[1]
    return [c for c in range(COLS) if not mask & TOP[c]]


def legal_mask(state: tuple[int, int]) -> np.ndarray:
    """Boolean array of length 7, True where the column is playable."""
    mask = state[1]
    return np.array([not mask & TOP[c] for c in range(COLS)], dtype=bool)


def play(state: tuple[int, int], col: int) -> tuple[int, int]:
    """Drop a stone in ``col``; returns the new state (perspective flipped).

    Raises ``ValueError`` on a full column or out-of-range column.
    """
    position, mask = state
    if not 0 <= col < COLS:
        raise ValueError(f"column {col} out of range")
    if mask & TOP[col]:
        raise ValueError(f"column {col} is full")
    return position ^ mask, mask | (mask + BOTTOM[col])


def has_won(stones: int) -> bool:
    """True if the given stone bitboard contains four in a row."""
    # vertical, horizontal, diagonal /, diagonal \  (shifts 1, 7, 8, 6)
    for d in (1, H1, H1 + 1, H1 - 1):
        m = stones & (stones >> d)
        if m & (m >> (2 * d)):
            return True
    return False


def last_move_won(state: tuple[int, int]) -> bool:
    """True if the player who just moved (i.e. NOT the one to move) won."""
    position, mask = state
    return has_won(position ^ mask)


def is_draw(state: tuple[int, int]) -> bool:
    """Full board with no winner. Check :func:`last_move_won` first."""
    return state[1] == FULL_MASK and not last_move_won(state)


def is_terminal(state: tuple[int, int]) -> bool:
    return last_move_won(state) or state[1] == FULL_MASK


def is_win_move(state: tuple[int, int], col: int) -> bool:
    """True if the player to move wins immediately by playing ``col``."""
    position, mask = state
    if mask & TOP[col]:
        return False
    stone = (mask + BOTTOM[col]) & (COL_BITS << (col * H1))
    return has_won(position | stone)


def winning_moves(state: tuple[int, int]) -> list[int]:
    """All columns that win on the spot for the player to move."""
    return [c for c in legal_moves(state) if is_win_move(state, c)]


def opponent_winning_moves(state: tuple[int, int]) -> list[int]:
    """Columns where the OPPONENT would win if it were their turn.

    These are the threats the player to move must block (or beat).
    """
    position, mask = state
    return winning_moves((position ^ mask, mask))


def flip_horizontal(state: tuple[int, int]) -> tuple[int, int]:
    """Mirror the board left-right (column c <-> column 6-c)."""
    position, mask = state
    fp = fm = 0
    for c in range(COLS):
        shift_src = c * H1
        shift_dst = (COLS - 1 - c) * H1
        fp |= ((position >> shift_src) & COL_BITS) << shift_dst
        fm |= ((mask >> shift_src) & COL_BITS) << shift_dst
    return fp, fm


def encode(state: tuple[int, int]) -> np.ndarray:
    """Encode as float32 planes (3, 6, 7), row 0 = TOP row.

    plane 0: stones of the player to move
    plane 1: stones of the opponent
    plane 2: all ones if the player to move is the first player, else zeros
    """
    position, mask = state
    opponent = position ^ mask

    def grid(bits: int) -> np.ndarray:
        arr = ((bits >> _SHIFTS) & 1).astype(np.float32).reshape(COLS, H1)
        # arr[col, row] with row 0 = bottom; drop sentinel, transpose, top-first
        return arr[:, :ROWS].T[::-1]

    planes = np.empty((3, ROWS, COLS), dtype=np.float32)
    planes[0] = grid(position)
    planes[1] = grid(opponent)
    planes[2] = 1.0 if mask.bit_count() % 2 == 0 else 0.0
    return planes


def from_moves(moves: str) -> tuple[int, int]:
    """Replay a game from a string of 0-indexed column digits, e.g. "3345"."""
    state = INITIAL
    for ch in moves:
        state = play(state, int(ch))
    return state


def to_string(state: tuple[int, int]) -> str:
    """ASCII board for debugging; X = first player, O = second, row 0 on top."""
    position, mask = state
    first = position if mask.bit_count() % 2 == 0 else position ^ mask
    second = first ^ mask
    lines = []
    for r in range(ROWS - 1, -1, -1):  # print top row first
        row = []
        for c in range(COLS):
            bit = 1 << (c * H1 + r)
            row.append("X" if first & bit else "O" if second & bit else ".")
        lines.append(" ".join(row))
    return "\n".join(lines)
