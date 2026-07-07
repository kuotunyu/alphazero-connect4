"""Connect Four vs. an AlphaZero-style agent — Gradio Space.

Downloads the trained weights from the public model repo at startup and
runs MCTS + the policy/value net on CPU. Per-session game state lives in
gr.State (plain data only); the model is a read-only module-level global.
"""

from __future__ import annotations

import functools
import os

import gradio as gr
import numpy as np
import torch

from az import game
from az.mcts import MCTS
from az.model import NetEvaluator, PolicyValueNet, create_model
from az.viz import render_board

MODEL_REPO = os.environ.get("MODEL_REPO", "steven0226/alphazero-connect4")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
LOCAL_WEIGHTS = os.environ.get("AZ_LOCAL_WEIGHTS")  # local dev: path to best.pt

DIFFICULTIES = [("簡單 · 50 sims", 50), ("中等 · 200 sims", 200),
                ("困難 · 800 sims（CPU 思考約 10 秒）", 800)]

torch.set_num_threads(2)


def load_model() -> PolicyValueNet:
    if LOCAL_WEIGHTS:  # local development against a training checkpoint
        payload = torch.load(LOCAL_WEIGHTS, map_location="cpu",
                             weights_only=True)
        model = create_model(payload["config"])
        model.load_state_dict(payload["model"])
        return model
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    weights = hf_hub_download(MODEL_REPO, "model.safetensors",
                              revision=MODEL_REVISION)
    config = hf_hub_download(MODEL_REPO, "config.json",
                             revision=MODEL_REVISION)
    with open(config) as f:
        model = create_model(json.load(f))
    model.load_state_dict(load_file(weights))
    return model


MODEL = load_model().eval()
EVALUATOR = NetEvaluator(MODEL, device="cpu")


def initial_state(human_first: bool = True) -> dict:
    return {"position": 0, "mask": 0, "over": False, "result": "",
            "human_first": human_first, "last_move": None}


def board_state(s: dict) -> tuple[int, int]:
    return s["position"], s["mask"]


def ai_win_prob(s: dict) -> float:
    """Value-head estimate of the AI's win probability at the current state."""
    state = board_state(s)
    if s["over"]:
        if "平手" in s["result"]:
            return 0.5
        return 1.0 if "AI 獲勝" in s["result"] else 0.0
    _, values = EVALUATOR.evaluate_batch([state])
    v = float(values[0])  # for the player to move, in [-1, 1]
    human_to_move = (game.ply(state) % 2 == 0) == s["human_first"]
    p_mover = (v + 1.0) / 2.0
    return 1.0 - p_mover if human_to_move else p_mover


def status_text(s: dict) -> str:
    if s["over"]:
        return f"### {s['result']}"
    human_to_move = (game.ply(board_state(s)) % 2 == 0) == s["human_first"]
    who = "🔴" if game.ply(board_state(s)) % 2 == 0 else "🟡"
    return f"### 輪到{'你' if human_to_move else ' AI'} {who}"


def finish_if_over(s: dict, mover_is_human: bool) -> None:
    state = board_state(s)
    if game.last_move_won(state):
        s["over"] = True
        s["result"] = "你獲勝了！" if mover_is_human else "AI 獲勝！"
    elif game.is_draw(state):
        s["over"] = True
        s["result"] = "平手。"


def ai_move(s: dict, sims: int) -> None:
    if s["over"]:
        return
    mcts = MCTS(EVALUATOR, sims, noise_frac=0.0,
                rng=np.random.default_rng())
    col = mcts.best_move(board_state(s))
    s["position"], s["mask"] = game.play(board_state(s), col)
    s["last_move"] = col
    print(f"[game] ply {game.ply(board_state(s)):2d}  AI    -> col {col + 1}",
          flush=True)
    finish_if_over(s, mover_is_human=False)


def render_all(s: dict):
    prob = ai_win_prob(s)
    label = {"AI 🤖": prob, "你": 1.0 - prob}
    return (render_board(board_state(s), s["last_move"]), s,
            label, status_text(s))


def on_new_game(side: str, sims: int):
    s = initial_state(human_first=side.startswith("先手"))
    if not s["human_first"]:
        ai_move(s, int(sims))
    return render_all(s)


def on_drop(col: int, s: dict, sims: int):
    state = board_state(s)
    if s["over"] or col not in game.legal_moves(state):
        return render_all(s)
    s["position"], s["mask"] = game.play(state, col)
    s["last_move"] = col
    print(f"[game] ply {game.ply(board_state(s)):2d}  human -> col {col + 1}",
          flush=True)
    finish_if_over(s, mover_is_human=True)
    ai_move(s, int(sims))
    if s["over"]:
        print("[game] result:",
              s["result"].encode("ascii", "backslashreplace").decode(),
              flush=True)
    return render_all(s)


with gr.Blocks(title="Connect4 Arena — AlphaZero") as demo:
    gr.Markdown(
        "# 四子棋 Connect Four — 挑戰 AlphaZero 式 agent\n"
        "自我對弈訓練的策略/價值網路 + MCTS。勝率條是網路 value head "
        "對當前局面的即時評估。模型："
        f"[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})"
    )
    session = gr.State(initial_state())

    with gr.Row():
        with gr.Column(scale=3):
            board = gr.Image(render_board(game.INITIAL), type="numpy",
                             label="棋盤", interactive=False)
            with gr.Row():
                buttons = [gr.Button(str(c + 1), min_width=40,
                                     elem_id=f"col-{c}")
                           for c in range(game.COLS)]
        with gr.Column(scale=2):
            status = gr.Markdown(status_text(initial_state()))
            value_label = gr.Label(label="value head 勝率評估",
                                   value={"AI 🤖": 0.5, "你": 0.5})
            side = gr.Radio(["先手（紅）", "後手（黃）"], value="先手（紅）",
                            label="你的棋色")
            sims = gr.Radio(DIFFICULTIES, value=200, label="難度（MCTS 模擬次數）")
            new_game = gr.Button("新對局 / 重開", variant="primary",
                                 elem_id="new-game")

    outputs = [board, session, value_label, status]
    new_game.click(on_new_game, inputs=[side, sims], outputs=outputs)
    for c, btn in enumerate(buttons):
        btn.click(functools.partial(on_drop, c), inputs=[session, sims],
                  outputs=outputs)

if __name__ == "__main__":
    demo.launch()
