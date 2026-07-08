# alphazero-connect4 — v1 實作計畫

## Context

從零實作 AlphaZero 式自我對弈訓練一個 Connect Four（6×7）agent：本機（Windows / RTX 2070 8GB）負責核心邏輯開發、單元測試與 SMOKE 級管線驗證；正式重訓放 Colab A100（過夜 6–10 小時預算）；最終部署成 Hugging Face Space 讓任何人上網對戰。程式碼組織為 Python 套件（src/ 佈局，package 名 `az`），訓練入口同時支援 `alphazero_connect4_colab_train.ipynb` 薄封裝與 `python -m az.train` 直跑。計畫經三路設計評審（引擎+MCTS／訓練管線+Colab／打包+Space 部署）定案。

**已與使用者確認：**
- HF 使用者名稱 `steven0226` → model repo `steven0226/alphazero-connect4`、Space `steven0226/connect4-arena`
- 本機 GPU 實測 RTX 2070 8GB（非 4090）→ 本機只跑開發/測試/SMOKE，夠用
- **GitHub 發佈暫緩**（gh CLI 未裝）：git 仍本機 init + commit 完整歷史，之後隨時可補 `gh repo create`
- Python 環境：專案內 `.venv`（Python 3.11.7）+ CUDA 版 torch（cu121 index，PyPI 的 Windows wheel 是 CPU-only）
- 專案根目錄 = 目前資料夾，不開子資料夾

**環境事實：** 資料夾全空、無 torch、無 gh/huggingface-cli、非 git repo。路徑含中文+全形括號——第一步就驗證 `pip install -e .` + pytest 全綠；若出現詭異 build 錯誤立即整包搬到純 ASCII 路徑（如 `C:\dev\alphazero-connect4`），不浪費時間 debug 路徑。

## 定案的關鍵設計決策（評審結論）

| 決策點 | 結論 |
|---|---|
| 自我對弈用哪個網 | **最新網（latest）**；gating（>55%）只決定 `best.pt`（部署與 vs-best 錨點），不回饋自我對弈——40 局 gate 雜訊大（σ≈7.9%），綁 champion 會因假陰性凍結資料生成 |
| 並行架構 | 同步 tick 制 pool：N 局各選一葉 → 湊一批 GPU forward → 展開回傳。每樹每 tick 一葉 → **不需 virtual loss** |
| 樹重用 | v1 不做（噪聲污染/複雜度），留 flag 給 v2；**例外：Space 對戰版可做跨步子樹重用**（省近半 sims，無訓練正確性問題） |
| π 訓練目標 | 永遠 = root 訪問次數比例（τ=1）；溫度只影響「實際走哪步」 |
| z 視角 | 「該 state 輪到走的人」視角 ±1/0，逐 ply 變號 |
| 對稱增強 | **採樣時**隨機水平翻轉（p=0.5，planes 沿欄軸翻 + π[::-1]），buffer 只存原始 |
| 網路寬度 | **預設 6 blocks × 96 filters（~1.1M 參數）**，config 留 128；96 對 CPU Space 延遲友善，Connect4 容量綽綽有餘 |
| 非法欄 mask | 模型輸出 raw logits；MCTS 展開、訓練 loss、Space 選步三處都以 `-1e9`（勿用 -inf，防 NaN）mask 後 softmax——訓練/推理一致 |
| Loss | 軟目標 CE `-(π·log_softmax(masked_logits)).sum(1).mean()` + MSE(z,v) + **AdamW** decoupled weight_decay 1e-4（不能用 nn.CrossEntropyLoss；Adam 的 weight_decay 是壞掉的 L2） |
| 發佈權重格式 | **safetensors + config.json**（torch≥2.6 `weights_only=True` 會拒載含雜物的 .pt；無 pickle 風險）；訓練期 checkpoint 仍用 .pt |
| Gradio | **6.x**（現行主線）；版本由 Space README `sdk_version` 控制，**不寫進 space/requirements.txt**；requirements 加 `--extra-index-url .../whl/cpu` 防拉 2.5GB CUDA torch |
| buffer 進 checkpoint | 進（緊湊編碼：棋盤 int8 + π float16 + z int8 ≈ 60B/位置，150k ≈ ~20MB）；不存則 resume 後 policy 易崩 |

## 專案佈局

```
（目前資料夾 = repo root）
├── pyproject.toml            # setuptools, src layout；[tool.setuptools.packages.find] where=["src"] 必寫
├── .gitignore                # checkpoints/ runs/ *.pt *.npz .venv/ __pycache__/；!assets/*.gif !assets/*.png
├── src/az/
│   ├── game.py               # bitboard 引擎
│   ├── model.py              # ResNet policy+value
│   ├── mcts.py               # PUCT + Evaluator 抽象（Net / Rollout / Uniform）
│   ├── selfplay.py           # 同步 tick 批次自我對弈 pool
│   ├── replay.py             # replay buffer（緊湊編碼、採樣時增強）
│   ├── train.py              # __main__ 訓練迴圈、checkpoint/resume、--max-hours
│   ├── arena.py              # 錨點對戰、gating、Elo MLE、CSV
│   ├── elo.py                # logistic MLE、錨點鏈
│   ├── viz.py                # 棋盤渲染（Figure+Agg，Space 與 GIF 共用）
│   └── config.py             # dataclass；SMOKE / FULL preset
├── tests/test_game.py        # 18 案
├── tests/test_mcts.py
├── scripts/
│   ├── make_gif.py           # 自我對弈一局 → assets/demo.gif（<2MB）
│   ├── plot_elo.py           # elo.csv → assets/elo_curve.png
│   ├── calibrate_anchor.py   # 一次性梯子法校準純 MCTS-200 Elo
│   ├── push_model.py         # safetensors+config+card+曲線 → HF model repo
│   └── deploy_space.py       # 複製 src/az → space/az 後 upload_folder 到 Space
├── space/                    # app.py、requirements.txt、README.md（sdk metadata）
├── alphazero_connect4_colab_train.ipynb
├── README.md
├── assets/
└── checkpoints/              # gitignored
```

## 核心模組設計

### 1. `az/game.py` — bitboard（Pascal Pons 標準佈局）
- bit index = `col*7 + row`，row 0 = 底列，每欄 7 bits（6 + 1 **哨兵列**，防跨欄 wrap 假連四）。狀態 = `(position, mask)` **Python 原生 int tuple**（可 hash、無 np.uint64 型別提升陷阱）；`position` = 當前輪到者的棋，對手 = `position ^ mask`。
- 落子：先檢查 `(mask & TOP[col])==0`；`new_mask = mask | (mask + BOTTOM[col])`、`new_position = position ^ mask`（換手）。勝負判定**對剛落子方**：d ∈ {1直, 7橫, 8「/」, 6「\」}：`m = p&(p>>d); win = m&(m>>2d)`。先判勝再判平（第 42 手可能是勝手）。
- `encode()` → float32 3×6×7：向量化位元展開（`(p >> np.arange(49)) & 1`），**約定 row 0 = 頂列**（配合渲染，寫進 docstring）；plane0 我方、plane1 對方、plane2 = 全 1（先手輪）/全 0。
- 水平翻轉：資料增強在 numpy 平面層做（`np.flip(planes,-1)` + `π[::-1]`）；bitboard 層 `flip()` 也實作，供交叉測試 `encode(flip(s)) == np.flip(encode(s),-1)`。
- 輔助：`winning_moves()`（即勝欄列表）、`from_moves("3341")`、`is_draw`、popcount 手數。

### 2. `az/model.py`
- Stem Conv3×3(3→C)+BN+ReLU；6× res block（Conv-BN-ReLU-Conv-BN+skip→ReLU）；policy head Conv1×1 C→32+BN+ReLU→Flatten→Linear(1344→7) **raw logits**；value head Conv1×1 C→32+BN+ReLU→Flatten→Linear→64→ReLU→Linear→1→tanh（當前輪到者視角）。C=96 預設 / 128 可選；SMOKE 用 3 blocks × 64。
- 推理一律 `model.eval()` + `torch.inference_mode()`（BN 忘了 eval 是經典沉默 bug）。

### 3. `az/mcts.py` + `az/selfplay.py`
- 節點用**陣列式**儲存：N/W/P 各為長度 7 的 np.float32 陣列，PUCT 一行向量式 argmax（naive 物件樹會把吞吐砍半）。`Q + c_puct·P·√N_parent/(1+N_child)`，c_puct=1.5，未訪子 Q=0。
- Evaluator 抽象：`evaluate_batch(states) -> (priors[B,7] 已 mask 歸一, values[B]∈[-1,1] 當前輪到者視角)`。三實作：`NetEvaluator`（批次 GPU，整批 numpy 回傳）、`RolloutEvaluator`（均勻 prior + 1 次隨機 rollout，即純 MCTS 錨點，CPU 即可）、`UniformEvaluator`（prior 均勻、value=0，測試用）。噪聲/溫度/sims 全是建構參數——self-play、arena、Space、tests 共用同一類別。
- 自我對弈 pool：128 局 lockstep；每 tick 每局選一條路徑到葉；終局葉直接真實結果 backprop（不進 batch）；非終局葉湊批 → **每 tick 恰一次 H2D/D2H** → 展開+回傳（沿路徑逐層 `v = -v`；終局葉 = 剛被將死方視角 **-1**）。局終回填 z、**立刻補新局進池保持批滿**；新局的根記得先展開+加噪聲。
- Root Dirichlet：展開**後**混 `P' = 0.75P + 0.25·Dir(α=1.0)`，只在根、噪聲維度 = 合法步數。溫度：ply<10 按訪問數 τ=1 採樣，之後 argmax（τ→0 直接 argmax 實作，別算 N^(1/0.01) 溢位）。
- A100 上自我對弈推理可 bf16 autocast；`torch.compile` 可選、先 profile。

### 4. `az/train.py`
- 每 iteration：latest 網自我對弈 G 局 → buffer → 訓練 K steps → arena → checkpoint。CLI：`python -m az.train --preset smoke|full --ckpt-dir DIR [--resume auto] [--max-hours H]`（`--max-hours` 讓訓練優雅收尾，把 push 時間留在 Colab 額度內）。
- 超參數定案（replay ratio ≈ 3–5，防過擬合 buffer / policy 熵崩）：

| 參數 | SMOKE（2070） | FULL（A100） |
|---|---|---|
| 網路 | 3 blocks × 64 | 6 blocks × 96（可選 128） |
| games/iter（=並行數） | 24 | 128 |
| sims/move | 64 | 160 |
| buffer | 20k 位置 | 150k 位置 |
| batch / steps/iter | 128 / 16 | 256 / 64 |
| optimizer | AdamW lr=1e-3, wd=1e-4（constant，過夜無人值守不搞排程） | 同左 |
- Checkpoint（原子寫 + `ckpt_last.pt`/`ckpt_prev.pt` 雙檔輪替，Colab 先寫本機再 copy 到 Drive）：model、optimizer、best_model、iteration、buffer（緊湊編碼）、elo_history、config（不合直接報錯）。`best.pt`/safetensors 獨立存。elo_history 存 ckpt 內、**每 iteration 全量重寫 CSV**（保證 resume 後一致）。`--resume auto`：有 ckpt_last 就續（損毀退 prev），沒有就全新——notebook 永遠帶此旗標，斷線 SOP = 重跑同一 cell。

### 5. `az/arena.py` + `az/elo.py`
- 每 iteration 對三錨點各 40 局（先後手各 20，和局 0.5 分）：(a) 隨機、(b) 純 MCTS-200（RolloutEvaluator）、(c) 目前 best。**雙方前 6 ply τ=1 採樣、之後 argmax、全程無噪聲**——否則兩個確定性 agent 同色 20 局是同一局棋。錨點設定跨 iteration 凍結（Elo 才可比）。網對網走批次推理路徑。
- Gating：vs best 分數 >22/40 → 升級 `best.pt`。文件如實記載統計限制（真 50% 的網有 ~21% 機率誤過，真 60% 有 ~31% 被擋；因自我對弈用 latest，誤判無累積傷害）。
- Elo：random ≡ 0；`calibrate_anchor.py` 用**梯子法**（random↔MCTS-16↔MCTS-64↔MCTS-200，相鄰各 200–400 局）一次性校準 MCTS-200 絕對 Elo 後凍結寫進 config（直接打 random 會撞 100% 截斷）。每 iteration 對已知錨點做 **1 維 logistic MLE**（凹函數，二分法十行）估 Elo；勝率夾到 [1/(2N), 1−1/(2N)]。舊 best 被換下時以當時 Elo 凍結加入錨點鏈，避免後期全飽和。
- CSV schema：`iteration, wallclock_s, wr_random, wr_mcts200, wr_best, elo, gated, buffer_size, loss_policy, loss_value, entropy_policy`。**另記「裸 policy vs random」勝率**（不搜尋直接 argmax policy）——MCTS+隨機權重本來就贏 random 90%+，裸 policy 從 ~60% 爬到 90%+ 才是「網路在學」最直接的證據。

### 6. `alphazero_connect4_colab_train.ipynb`
- Cell：①參數（`SMOKE_TEST` 唯一手改旗標、RUN_NAME、HF_USERNAME=steven0226；SMOKE/FULL 兩個 config dict 選一傳 CLI）→ ②git clone 到 /content + `pip install -e .`（torch 用 Colab 內建，**絕不 pin 版本蓋掉 CUDA build**）→ ③掛 Drive，ckpt 目錄 `MyDrive/alphazero-connect4/<run>/` → ④`userdata.get("HF_TOKEN")` → ⑤`pytest tests/ -q` 煙霧驗證抓環境漂移 → ⑥`subprocess.Popen([sys.executable,"-m","az.train",...,"--resume","auto","--max-hours","8"])`，log 落 Drive（cell 直跑會被 websocket 拖死；真正斷線防護 = Drive ckpt + resume）→ ⑦監控 cell：tail log + 每 5 分鐘畫 elo.csv → ⑧收尾：`try: proc.wait(); push_to_hf(...) finally: runtime.unassign()`（push 失敗不能擋 unassign；Drive 是真相來源，可事後本機補 push）。
- **A100 預算估算（寫進 notebook）**：每 iter ≈ 128 局 × ~25 手 × 160 sims ≈ 512k 次葉評估；A100 有效吞吐（含 Python 樹操作開銷）≈ 15–30k evals/s → 自我對弈 25–50s + arena 30–60s + 訓練 <10s ≈ **1.5–3 min/iter → 7 小時 ≈ 140–280 iterations，規劃目標 150 iter、--max-hours 8 兜底**。SMOKE 模式（迷你網+8局/iter，<10 分鐘）兼作 A100 吞吐實測，跑 3 iter 量 sec/iter 外推後再開正式 run。

### 7. `space/`（Gradio 6.x，CPU basic）
- 模型 module-level 載入一次：`hf_hub_download("steven0226/alphazero-connect4", "model.safetensors", revision=<pin>)`（公開免 token；**pin revision** 防重訓後 Space 默默換腦）+ config.json 重建網 → eval() → `torch.set_num_threads(2)`。
- `gr.State` 只存純資料 dict（board/player/over/history）——**絕不放 tensor 或 MCTS 樹**；全域變數存棋局 = 多訪客互相污染。7 欄按鈕用 `functools.partial` 綁 col（lambda 晚綁定 = 7 顆全下同欄）。滿欄/終局 handler early-return。
- 棋盤渲染 `az/viz.py`：`matplotlib.figure.Figure` + `FigureCanvasAgg`（不用 pyplot——非 thread-safe 且會洩漏 figure OOM）→ RGBA buffer → PIL。
- 顯示 value head 勝率評估（規格要求的展示點；輪到者視角換算成「AI 勝率」條，符號別反）。難度三檔 50/200/800 sims（gr.Radio 呈現）；先後手選擇（選後手則開局 AI 先走）；終局判定 + 重開。
- CPU 延遲：50 sims <1s、200 ≈ 2–4s、800 ≈ 6–15s → 緩解：步內 NN 評估快取（board hash → (π,v)，省 20–40%）+ 跨步子樹重用 + UI 標註「困難檔思考約 10 秒」。
- `space/requirements.txt`：`--extra-index-url https://download.pytorch.org/whl/cpu` + torch/numpy/matplotlib/huggingface_hub；**gradio 不寫**（由 README `sdk: gradio` + `sdk_version: 6.x` 控制）。README metadata 含 `models: ["steven0226/alphazero-connect4"]`。
- 部署：`deploy_space.py` 複製 `src/az` → `space/az`（自帶副本，不用 git 依賴）→ `create_repo(..., repo_type="space", space_sdk="gradio", exist_ok=True)` + `upload_folder(..., repo_type="space", ignore_patterns=["__pycache__/*"])`。**先本機跑 app、我自己與 agent 對戰一局並回報過程結果，才 push。**

### 8. 文件與加分
- `make_gif.py`：最佳權重自我對弈一局（200 sims）→ viz.py 逐半步渲染 → PIL 存 GIF（dpi≤80、≤45 幀、<2MB、末幀停格高亮連四）→ `assets/demo.gif`，同時 upload 到 model repo root（model card 相對路徑引用，只放 GitHub 路徑會在 HF 破圖）。
- README：demo GIF 首屏 → AlphaZero 三要素白話（MCTS 用網路引導搜尋 → 搜尋結果當訓練標籤 → 網路變強反哺搜尋的閉環）→ mermaid 訓練閉環圖（SelfPlay→(s,π,z)→Buffer→Train→Gating→Best↺；旁支 Anchors→Elo；部署支線 Best→HF→Space）→ 專案結構 → 本機/Colab 兩種訓練方式 → 真實 Elo 曲線與數據表 → Space 連結與截圖佔位。
- Model card（**不渲染 mermaid**，用文字+預渲染圖）：架構表、方法與超參、真實訓練規模、Elo 表+曲線、用法三行示意、限制（含 gating 統計雜訊；若未穩定打贏純 MCTS-200 錨點→如實寫限制+改進方向）。

## 實作順序（里程碑）

1. **環境+骨架**：`.venv`、cu121 torch、numpy/pytest/matplotlib/huggingface_hub；pyproject（`where=["src"]`）+ `pip install -e ".[dev]"`；在非 repo 目錄驗證 `import az` 指到 site-packages 連結；git init + .gitignore。路徑問題此時就會暴露。
2. **game.py + test_game.py（18 案）**全綠——含反 wrap、不對稱斜向、頂列橫連、第 42 手勝、交叉翻轉測試。
3. **mcts.py（Uniform/Rollout evaluator）+ test_mcts.py** 全綠——一步殺與必須擋各以 Uniform 與 Rollout(seed) 兩種 evaluator 驗證，π 正確欄質量 ≥0.7；再加 model.py 與 NetEvaluator。
4. **selfplay/replay/train/arena/elo/config** + `calibrate_anchor.py` → **本機 SMOKE（2070）30–50 iter**：驗證裸 policy vs random 勝率明顯上升（~60%→90%+）、vs MCTS-200 上升、Elo 曲線生成、中斷後 `--resume auto` 接續無縫。
5. **alphazero_connect4_colab_train.ipynb**（以 SMOKE 實測吞吐校準 A100 估算數字）。
6. **space/app.py 本機跑**（先用 SMOKE 權重）→ 我與 agent 實際對弈一局、回報棋譜與勝率顯示 → 使用者提供 HF_TOKEN（write）後：push model repo（safetensors+config+card）+ push Space。正式權重等使用者跑完 Colab 後以 `push_model.py` 更新、bump Space revision。
7. **make_gif.py、plot_elo.py、README、model card**；git commit 完整歷史。（GitHub 發佈暫緩；使用者決定後補 `gh repo create alphazero-connect4 --public --source=. --push` + topics。）

## 驗證方式

- `pytest tests/` 全綠（Windows 本機；notebook 內同套測試在 Colab 再跑一次）。
- SMOKE 訓練真實數據：elo.csv、裸 policy 勝率上升、`--resume` 中斷續跑實測、elo_curve.png 生成。
- Space：本機實際對弈一局全流程（落子/勝率條/終局/重開/先後手/三檔難度），貼結果；push 後開 Space build log 確認起得來。
- 所有訓練數據與 Elo 皆真實執行結果；達不到穩定打贏純 MCTS-200 → 如實寫進限制章節。

## 風險與備案

- **中文路徑**：一有詭異錯誤即整包遷移純 ASCII 路徑（成本一次 git mv，早做早安心）。
- **2070 SMOKE 吞吐**：估 ~30–60s/iter；不足則再縮網/局數——SMOKE 只驗證管線與學習訊號。
- **HF free CPU 800 sims 偏慢（6–15s）**：評估快取+子樹重用+UI 預期管理；仍不行降最高檔為 400。
- **Colab 斷線**：Drive ckpt 雙檔輪替 + `--resume auto`，重跑 notebook 即接續；push 包 try、unassign 在 finally。
- **40 局 gating 雜訊**：已用 latest-net 自我對弈解耦；文件如實記載。
