# AlphaZero Connect Four — FULL run summary

- 來源：`results/elo_full.csv`（HF model repo 公開 `elo.csv` 的本地鏡像）
- 150 iterations，19,200 局 self-play，7.92 小時
- 59/150 iterations 通過 gating

| 指標 | 起點 | 終點 | 補充 |
|---|---:|---:|---|
| Elo | 973.8 | **1625.6** | 峰值 1625.6 首次出現在 iter 62；末 50 iter 平均 1593.6 |
| vs MCTS-200 | 50.0% | **100.0%** | 末 50 iter 平均 98.8%；iter 69 起每輪 ≥90% |
| 裸 policy vs random | 75.0% | **100.0%** | 不依賴搜尋的學習訊號 |
| Policy loss | 1.8528 | 1.0421 | |
| Value loss | 0.8098 | 0.1565 | |

重算：

```bash
python scripts/summarize_full_run.py
python scripts/plot_elo.py results/elo_full.csv assets/elo_curve_full.png
```
