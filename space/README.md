---
title: Connect4 Arena — AlphaZero
emoji: 🔴
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
models:
- steven0226/alphazero-connect4
---

# Connect4 Arena

Play Connect Four against an AlphaZero-style agent trained from scratch by
self-play + MCTS. The win-probability bar shows the network's value-head
evaluation of the current position in real time.

- **難度** = MCTS 模擬次數（50 / 200 / 800）；越高越強、思考越久（CPU 推理）。
- 可選先手（紅）或後手（黃）。
- 權重從公開 model repo
  [steven0226/alphazero-connect4](https://huggingface.co/steven0226/alphazero-connect4)
  於啟動時下載。
