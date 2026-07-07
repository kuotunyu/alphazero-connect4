"""Shared rendering: board images (Space app + GIF) and the Elo curve.

Uses matplotlib's object-oriented API with the Agg canvas — no pyplot, so
it is thread-safe (Gradio handlers run in threads) and leaks no figures.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from az import game

BOARD_COLOR = "#1f4e9c"
P1_COLOR = "#e63946"  # first player: red
P2_COLOR = "#f4c430"  # second player: yellow
EMPTY_COLOR = "#f4f6fb"


def render_board(state, last_move: int | None = None,
                 title: str | None = None) -> np.ndarray:
    """Render a position to an RGB uint8 array (row 0 of the board on top)."""
    position, mask = state
    first = position if mask.bit_count() % 2 == 0 else position ^ mask
    second = first ^ mask

    fig = Figure(figsize=(4.2, 3.9), dpi=110)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.55, game.COLS - 0.45)
    ax.set_ylim(-0.55, game.ROWS - 0.45)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-0.55, -0.55), game.COLS + 0.1, game.ROWS + 0.1,
                           color=BOARD_COLOR, zorder=0))

    last_cell = None
    if last_move is not None:
        height = ((mask >> (last_move * game.H1)) & 0x3F).bit_count()
        last_cell = (last_move, height - 1)

    for c in range(game.COLS):
        for r in range(game.ROWS):  # bitboard row, 0 = bottom
            bit = 1 << (c * game.H1 + r)
            color = (P1_COLOR if first & bit
                     else P2_COLOR if second & bit else EMPTY_COLOR)
            edge = "black" if last_cell == (c, r) else "none"
            ax.add_patch(Circle((c, r), 0.42, facecolor=color,
                                edgecolor=edge, linewidth=2.0, zorder=1))

    ax.set_xticks(range(game.COLS), [str(i + 1) for i in range(game.COLS)])
    ax.set_yticks([])
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.3)

    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def plot_elo(history: list[dict], out_path) -> None:
    """Elo curve + anchor win rates from the training history rows."""
    if not history:
        return
    it = [row["iteration"] for row in history]
    fig = Figure(figsize=(8, 6), dpi=110)
    FigureCanvasAgg(fig)

    ax = fig.add_subplot(211)
    ax.plot(it, [row["elo"] for row in history], "-o", ms=3, color="#1f4e9c")
    for row in history:
        if row["gated"]:
            ax.axvline(row["iteration"], color="#2a9d8f", alpha=0.25, lw=1)
    ax.set_ylabel("Elo (random agent = 0)")
    ax.set_title("Elo vs. fixed anchors (green lines: new best gated in)")
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(212, sharex=ax)
    for key, label, color in [
        ("wr_random", "MCTS+net vs random", "#457b9d"),
        ("wr_policy_random", "raw policy vs random", "#e76f51"),
        ("wr_mcts200", "vs pure MCTS-200", "#2a9d8f"),
        ("wr_best", "vs current best", "#8d99ae"),
    ]:
        ax2.plot(it, [row[key] for row in history], "-o", ms=2.5,
                 label=label, color=color)
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("score")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
