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
  於啟動時下載，而且 `config.json` 與 `model.safetensors` 固定在同一個
  [model commit](https://huggingface.co/steven0226/alphazero-connect4/tree/0ba2361fe4044af9f6bfadfa89997b46191077c7)。
- 核心 Python 套件來自
  [GitHub source repository](https://github.com/kuotunyu/alphazero-connect4) 的 canonical
  `src/az`；發布腳本在上傳前生成自含式 `space/az` bundle。
