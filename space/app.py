"""Connect Four vs. an AlphaZero-style agent — Gradio Space.

Downloads the trained weights from the public model repo at startup and
runs MCTS + the policy/value net on CPU. Per-session game state lives in
gr.State (plain data only); the model is a read-only module-level global.
"""

from __future__ import annotations

import functools
import os
from html import escape

import gradio as gr
import numpy as np
import torch

from az import game
from az.mcts import MCTS
from az.model import NetEvaluator, PolicyValueNet, create_model
from az.viz import render_board
from model_assets import download_model_assets, resolve_model_source

MODEL_REPO, MODEL_REVISION = resolve_model_source()
LOCAL_WEIGHTS = os.environ.get("AZ_LOCAL_WEIGHTS")  # local dev: path to best.pt

DIFFICULTIES = [("簡單 · 50", 50), ("中等 · 200", 200),
                ("困難 · 800", 800)]

APP_CSS = """
:root {
  --az-ink: #172033;
  --az-muted: #4e5b6c;
  --az-blue: #183c73;
  --az-blue-soft: #edf3fa;
  --az-orange: #e9680b;
  --az-orange-soft: #fff7ed;
  --az-line: #c8ced8;
  --az-surface: #ffffff;
  --az-canvas: #f5f7f9;
}

.gradio-container {
  width: 100% !important;
  max-width: 1660px !important;
  margin-inline: auto !important;
  padding: 16px clamp(20px, 2vw, 32px) 24px !important;
  box-sizing: border-box;
  overflow-x: hidden;
  color: var(--az-ink);
  background: var(--az-canvas);
  font-family: "Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif;
  font-size: 20px !important;
}

.gradio-container .main,
.gradio-container main.contain {
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
}

.gradio-container p,
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container .wrap,
.gradio-container .label-wrap {
  font-size: 20px !important;
  line-height: 1.45;
}

#app-header {
  padding: 6px 2px 16px;
  border-bottom: 4px solid var(--az-blue);
}

#app-header h1 {
  margin: 0;
  color: var(--az-ink);
  font-size: 36px !important;
  line-height: 1.2;
  letter-spacing: -0.025em;
}

#app-header p,
#app-header a,
#app-header code {
  font-size: 20px !important;
  line-height: 1.5;
}

#app-header p { margin: 8px 0 0; color: var(--az-muted); }
#app-header a { color: var(--az-blue); font-weight: 700; }
#app-header code { color: var(--az-blue); background: transparent; padding: 0; }

#game-layout {
  gap: 18px;
  align-items: flex-start;
  margin-top: 18px;
}

#board-panel,
#control-panel,
#position-evaluation,
#board {
  border-radius: 0 !important;
  box-shadow: none !important;
}

#board-panel {
  min-width: 0;
  gap: 9px;
  flex: 62 1 0 !important;
}

#board {
  overflow: hidden;
  border: 1px solid var(--az-line);
  background: var(--az-surface);
}

#board .image-container {
  width: 100% !important;
  border-radius: 0 !important;
}

#board .image-frame {
  width: 100% !important;
}

#board img {
  width: 100% !important;
  height: auto !important;
  object-fit: contain !important;
  border-radius: 0 !important;
}

#column-actions {
  display: grid !important;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 7px;
}

.column-button {
  width: 100% !important;
  min-width: 0 !important;
  min-height: 52px !important;
  padding: 0 !important;
  border: 1px solid #b8c0cc !important;
  border-radius: 4px !important;
  color: var(--az-ink) !important;
  background: var(--az-surface) !important;
  font-size: 22px !important;
  font-weight: 800 !important;
}

.column-button:hover {
  border-color: var(--az-orange) !important;
  color: #943b07 !important;
  background: var(--az-orange-soft) !important;
}

.column-button:focus-visible,
#new-game:focus-visible,
.game-control input:focus-visible + span {
  outline: 3px solid #0b6bcb !important;
  outline-offset: 2px;
}

#control-panel {
  min-width: 420px;
  flex: 38 1 0 !important;
  gap: 13px;
  padding: 0 18px 18px;
  border: 1px solid var(--az-line);
  border-top: 4px solid var(--az-blue);
  background: var(--az-surface);
}

#turn-status {
  margin: 0 -18px;
  padding: 14px 18px 13px;
  border-bottom: 1px solid var(--az-line);
  background: var(--az-blue-soft);
}

#turn-status h3 {
  margin: 0;
  color: var(--az-blue);
  font-size: 28px !important;
  line-height: 1.3;
}

#turn-status .progress-text {
  color: var(--az-blue) !important;
  font-size: 0 !important;
  font-weight: 800;
}

#turn-status .progress-text::before {
  content: "AI 思考中…";
  font-size: 20px;
}

#position-evaluation {
  padding: 10px 0 12px;
  border: 0;
  border-top: 1px solid var(--az-line);
  border-bottom: 1px solid var(--az-line);
}

#position-evaluation .output-class,
#position-evaluation .confidence,
#position-evaluation .label {
  font-size: 30px !important;
}

#position-evaluation > label {
  position: static !important;
  display: block;
  padding: 0 0 8px !important;
  color: var(--az-ink);
  background: transparent !important;
  font-size: 20px !important;
  font-weight: 800;
}

#position-evaluation > label svg,
#position-evaluation .output-class { display: none !important; }

#position-evaluation .confidence-set:first-of-type meter {
  background: var(--az-blue) !important;
}

#position-evaluation .confidence-set:last-of-type meter {
  background: #f2c94c !important;
}

.game-control {
  border-radius: 0 !important;
  box-shadow: none !important;
}

.game-control label,
.game-control span,
.game-control .wrap {
  font-size: 20px !important;
}

.game-control .wrap {
  border-radius: 4px !important;
}

#side-control .wrap,
#difficulty-control .wrap {
  display: grid !important;
  gap: 8px !important;
}

#side-control .wrap { grid-template-columns: repeat(2, minmax(0, 1fr)); }
#difficulty-control .wrap { grid-template-columns: repeat(3, minmax(0, 1fr)); }

#side-control label,
#difficulty-control label {
  min-width: 0 !important;
  min-height: 48px !important;
  justify-content: center;
  border-radius: 4px !important;
  touch-action: manipulation;
}

#new-game {
  min-height: 56px;
  border: 0 !important;
  border-radius: 4px !important;
  color: #ffffff !important;
  background: var(--az-orange) !important;
  font-size: 22px !important;
  font-weight: 800 !important;
}

#new-game:hover { background: #c95408 !important; }
#new-game:disabled,
.column-button:disabled { cursor: not-allowed; opacity: 0.58; }

#thinking-note {
  padding-top: 10px;
  border-top: 1px solid #dfe3e9;
}

#thinking-note p {
  margin: 0;
  color: var(--az-muted);
  font-size: 20px !important;
  line-height: 1.45;
}

@media (max-width: 900px) {
  .gradio-container { padding: 12px !important; }
  #app-header h1 { font-size: 31px !important; }
  #game-layout { flex-direction: column; }
  #board-panel,
  #control-panel {
    min-width: 0;
    flex: 1 1 auto !important;
  }
  #control-panel { border-left: 1px solid var(--az-line); }
  #column-actions { gap: 4px; }
  .column-button { min-height: 54px !important; }
}
"""

torch.set_num_threads(2)


def load_model() -> PolicyValueNet:
    if LOCAL_WEIGHTS:  # local development against a training checkpoint
        payload = torch.load(LOCAL_WEIGHTS, map_location="cpu",
                             weights_only=True)
        model = create_model(payload["config"])
        model.load_state_dict(payload["model"])
        return model
    from safetensors.torch import load_file
    import json

    assets = download_model_assets()
    with assets.config_path.open(encoding="utf-8") as f:
        model = create_model(json.load(f))
    model.load_state_dict(load_file(assets.weights_path))
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
        message = s["result"]
    else:
        human_to_move = (
            (game.ply(board_state(s)) % 2 == 0) == s["human_first"]
        )
        who = "🔴" if game.ply(board_state(s)) % 2 == 0 else "🟡"
        message = f"輪到{'你' if human_to_move else ' AI'} {who}"
    return (
        '<div role="status" aria-live="polite" aria-atomic="true">'
        f"<h3>{escape(message)}</h3>"
        "</div>"
    )


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
        "# 四子棋 Connect Four — 挑戰 AlphaZero\n"
        "選擇欄位落子，AlphaZero 將以策略／價值網路與 MCTS 回應。  \n"
        "模型："
        f"[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) "
        f"@ `{MODEL_REVISION[:12]}`",
        elem_id="app-header",
    )
    session = gr.State(initial_state())

    with gr.Row(elem_id="game-layout"):
        with gr.Column(scale=5, elem_id="board-panel"):
            board = gr.Image(
                render_board(game.INITIAL),
                type="numpy",
                show_label=False,
                interactive=False,
                buttons=[],
                elem_id="board",
            )
            with gr.Row(elem_id="column-actions"):
                buttons = [gr.Button(str(c + 1), min_width=40,
                                     elem_id=f"col-{c}",
                                     elem_classes=["column-button"])
                           for c in range(game.COLS)]
        with gr.Column(scale=3, elem_id="control-panel"):
            status = gr.HTML(
                status_text(initial_state()),
                elem_id="turn-status",
            )
            value_label = gr.Label(
                label="局面評估",
                value={"AI 🤖": 0.5, "你": 0.5},
                elem_id="position-evaluation",
            )
            side = gr.Radio(["先手（紅）", "後手（黃）"], value="先手（紅）",
                            label="你的棋色",
                            elem_id="side-control",
                            elem_classes=["game-control"])
            sims = gr.Radio(DIFFICULTIES, value=200,
                            label="AI 強度（MCTS 模擬次數）",
                            elem_id="difficulty-control",
                            elem_classes=["game-control"])
            new_game = gr.Button("開始新對局", variant="primary",
                                 elem_id="new-game")
            gr.Markdown("AI 思考時會保留棋盤；困難模式約需 10 秒。",
                        elem_id="thinking-note")

    outputs = [board, session, value_label, status]
    new_game.click(on_new_game, inputs=[side, sims], outputs=outputs,
                   show_progress="minimal", show_progress_on=status)
    for c, btn in enumerate(buttons):
        btn.click(functools.partial(on_drop, c), inputs=[session, sims],
                  outputs=outputs, show_progress="minimal",
                  show_progress_on=status)

def launch_app() -> None:
    demo.launch(css=APP_CSS)


if __name__ == "__main__":
    launch_app()
