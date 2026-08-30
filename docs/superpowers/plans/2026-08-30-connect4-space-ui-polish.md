# Connect4 Space UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a clearer, denser, non-template-like Connect Four Space whose player-facing text is at least 20px and whose AI wait state never covers the board.

**Architecture:** Keep Gradio 6.20 and the existing callback/API boundary. Add one `APP_CSS` design-token block and stable `elem_id`/`elem_classes` hooks in `space/app.py`; switch all click events from the default full progress overlay to Gradio's minimal progress mode. Verify the source contract in pytest, then verify computed styles and interactions in a real Chromium browser before publishing the same `space/` bundle.

**Tech Stack:** Python 3.11, Gradio 6.20.0, PyTorch CPU inference, pytest, Playwright Chromium, GitHub Actions, Hugging Face Spaces.

## Global Constraints

- Every player-facing text element controlled by the Space must have a computed font size of at least `20px`.
- Desktop heading is `36px`; narrow-screen heading is `31px`; turn status is `28px`; key probability is `30px`; column controls are `22px`.
- General controls use at most `4px` corner radius; card-like layout containers remain square. Circular game pieces and tiny status dots are exempt.
- Keep the existing model revision `0ba2361fe4044af9f6bfadfa89997b46191077c7` unchanged.
- Keep callback inputs, output order `[board, session, value_label, status]`, public API names, game rules, MCTS counts, and model behavior unchanged.
- Keep `src/az` canonical and generate `space/az` only through `scripts/deploy_space.py`.
- Do not add animation, audio, accounts, leaderboard, history, analysis charts, or a frontend build toolchain.
- At `900px` and below, use one column with no horizontal scrolling.
- Do not modify any repository outside this AlphaZero project.

---

### Task 1: Accessible visual system and compact layout

**Files:**
- Modify: `tests/test_space_release.py:13-79`
- Modify: `space/app.py:23-170`

**Interfaces:**
- Consumes: existing Gradio components and callback output order `[board, session, value_label, status]`.
- Produces: `APP_CSS: str`, stable hooks `app-header`, `game-layout`, `board-panel`, `board`, `column-actions`, `control-panel`, `turn-status`, `position-evaluation`, `game-control`, `new-game`, and `thinking-note`.

- [ ] **Step 1: Write the failing source-contract test**

Add to `tests/test_space_release.py`:

```python
def load_space_app_source() -> str:
    return (ROOT / "space" / "app.py").read_text(encoding="utf-8")


def test_space_ui_contract_is_large_type_compact_and_responsive():
    source = load_space_app_source()

    for hook in (
        "app-header",
        "game-layout",
        "board-panel",
        "column-actions",
        "control-panel",
        "turn-status",
        "position-evaluation",
        "game-control",
        "thinking-note",
    ):
        assert hook in source

    assert "APP_CSS" in source
    assert "font-size: 20px" in source
    assert "font-size: 22px" in source
    assert "font-size: 28px" in source
    assert "font-size: 30px" in source
    assert "font-size: 36px" in source
    assert "font-size: 31px" in source
    assert "border-radius: 4px" in source
    assert "@media (max-width: 900px)" in source
    assert "overflow-x: hidden" in source
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  -m pytest tests/test_space_release.py::test_space_ui_contract_is_large_type_compact_and_responsive -q
```

Expected: FAIL because `space/app.py` does not yet define `APP_CSS` or the required hooks.

- [ ] **Step 3: Add the visual tokens and responsive CSS**

Add a single `APP_CSS` string below `DIFFICULTIES` in `space/app.py`. It must implement these exact public tokens and rules:

```python
APP_CSS = """
:root {
  --az-ink: #172033;
  --az-muted: #4e5b6c;
  --az-blue: #183c73;
  --az-orange: #f97316;
  --az-orange-soft: #fff7ed;
  --az-line: #c8ced8;
  --az-surface: #ffffff;
  --az-canvas: #f5f7f9;
}
.gradio-container {
  max-width: 1240px !important;
  padding: 16px 20px 24px !important;
  overflow-x: hidden;
  color: var(--az-ink);
  font-size: 20px !important;
}
.gradio-container p,
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container .wrap,
.gradio-container .label-wrap { font-size: 20px !important; }
#app-header h1 { font-size: 36px !important; line-height: 1.2; }
#app-header p { font-size: 20px !important; line-height: 1.5; }
#game-layout { gap: 18px; align-items: stretch; }
#board-panel, #control-panel, #position-evaluation {
  border-radius: 0 !important;
  box-shadow: none !important;
}
#board-panel { min-width: 0; }
#board { border: 1px solid var(--az-line); border-radius: 0 !important; }
#column-actions { gap: 7px; }
.column-button {
  min-height: 52px !important;
  border-radius: 4px !important;
  font-size: 22px !important;
  font-weight: 800 !important;
}
#control-panel {
  border-left: 4px solid var(--az-blue);
  padding: 0 18px 18px;
  background: var(--az-surface);
}
#turn-status h3 { font-size: 28px !important; line-height: 1.3; }
#position-evaluation .output-class { font-size: 30px !important; }
.game-control { border-radius: 0 !important; }
.game-control label, .game-control span { font-size: 20px !important; }
#new-game { min-height: 56px; border-radius: 4px !important; font-size: 22px !important; }
#thinking-note p { font-size: 20px !important; color: var(--az-muted); }
@media (max-width: 900px) {
  .gradio-container { padding: 12px !important; }
  #app-header h1 { font-size: 31px !important; }
  #game-layout { flex-direction: column; }
  #control-panel { border-left: 0; border-top: 4px solid var(--az-blue); }
  .column-button { min-height: 54px !important; }
}
"""
```

Adjust selectors after the first browser inspection only if Gradio's rendered DOM requires a more specific selector. Do not lower any required font size.

- [ ] **Step 4: Apply stable hooks and compact copy to the Gradio tree**

Change the Blocks declaration and components to this structure while keeping existing values and outputs:

```python
with gr.Blocks(title="Connect4 Arena — AlphaZero", css=APP_CSS) as demo:
    gr.Markdown(
        "# 四子棋 Connect Four — 挑戰 AlphaZero\n"
        "選擇欄位落子，AlphaZero 將以策略／價值網路與 MCTS 回應。  \n"
        f"模型：[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) "
        f"@ `{MODEL_REVISION[:12]}`",
        elem_id="app-header",
    )
    session = gr.State(initial_state())

    with gr.Row(elem_id="game-layout"):
        with gr.Column(scale=5, elem_id="board-panel"):
            board = gr.Image(
                render_board(game.INITIAL), type="numpy", show_label=False,
                interactive=False, elem_id="board",
            )
            with gr.Row(elem_id="column-actions"):
                buttons = [
                    gr.Button(
                        str(c + 1), min_width=40, elem_id=f"col-{c}",
                        elem_classes=["column-button"],
                    )
                    for c in range(game.COLS)
                ]
        with gr.Column(scale=3, elem_id="control-panel"):
            status = gr.Markdown(status_text(initial_state()), elem_id="turn-status")
            value_label = gr.Label(
                label="局面評估", value={"AI 🤖": 0.5, "你": 0.5},
                elem_id="position-evaluation",
            )
            side = gr.Radio(
                ["先手（紅）", "後手（黃）"], value="先手（紅）",
                label="你的棋色", elem_classes=["game-control"],
            )
            sims = gr.Radio(
                DIFFICULTIES, value=200, label="AI 強度（MCTS 模擬次數）",
                elem_classes=["game-control"],
            )
            new_game = gr.Button(
                "開始新對局", variant="primary", elem_id="new-game",
            )
            gr.Markdown(
                "AI 思考時會保留棋盤，請稍候片刻。",
                elem_id="thinking-note",
            )
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 6: Run the complete release-boundary test file**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  -m pytest tests/test_space_release.py -q
```

Expected: all tests in `tests/test_space_release.py` pass.

- [ ] **Step 7: Commit the accessible layout**

```powershell
git add -- space/app.py tests/test_space_release.py
git commit -m "feat: polish accessible Space layout"
```

---

### Task 2: Preserve the board during AI computation

**Files:**
- Modify: `tests/test_space_release.py`
- Modify: `space/app.py:166-170`

**Interfaces:**
- Consumes: existing callbacks `on_new_game(side: str, sims: int)` and `on_drop(col: int, s: dict, sims: int)`.
- Produces: the same eight public event endpoints and the same four outputs, configured with `show_progress="minimal"`.

- [ ] **Step 1: Write the failing progress-contract test**

Add:

```python
def test_space_events_keep_outputs_visible_during_ai_compute():
    source = load_space_app_source()

    assert source.count('show_progress="minimal"') == 2
    assert 'show_progress="full"' not in source
    assert "AI 思考時會保留棋盤" in source
    assert "outputs = [board, session, value_label, status]" in source
```

The count is two because one declaration configures the new-game event and the declaration inside the button loop configures all seven column events.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  -m pytest tests/test_space_release.py::test_space_events_keep_outputs_visible_during_ai_compute -q
```

Expected: FAIL because the click declarations still use Gradio's default full progress mode.

- [ ] **Step 3: Configure lightweight progress on every click event**

Use the exact event declarations:

```python
new_game.click(
    on_new_game,
    inputs=[side, sims],
    outputs=outputs,
    show_progress="minimal",
)
for c, btn in enumerate(buttons):
    btn.click(
        functools.partial(on_drop, c),
        inputs=[session, sims],
        outputs=outputs,
        show_progress="minimal",
    )
```

Do not use custom JavaScript, timers, streamed callbacks, or a new endpoint.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Run all tests**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' -m pytest -q
```

Expected: the previous 39 tests plus the two new UI contract tests pass.

- [ ] **Step 6: Commit the wait-state fix**

```powershell
git add -- space/app.py tests/test_space_release.py
git commit -m "fix: keep board visible during AI turns"
```

---

### Task 3: Local browser and responsive verification

**Files:**
- Generate but do not commit: `space/az/`
- Create temporarily under ignored workspace: `.superpowers/ui-smoke.py`
- Create temporarily under ignored workspace: `.superpowers/ui-desktop.png`
- Create temporarily under ignored workspace: `.superpowers/ui-mobile.png`
- Modify only if evidence requires it: `space/app.py`

**Interfaces:**
- Consumes: local Gradio URL, rendered hooks from Task 1, and the current pinned model.
- Produces: computed-style evidence and desktop/mobile screenshots; no new product API.

- [ ] **Step 1: Rebuild the deployment bundle without uploading**

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  scripts/deploy_space.py --dry-run
```

Expected: `space/az` is rebuilt from `src/az`; no Hugging Face upload occurs.

- [ ] **Step 2: Inspect the managed-server helper interface**

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  'C:\Users\3Hml\.codex\skills\webapp-testing\scripts\with_server.py' --help
```

Expected: help output describes `--server`, `--port`, and the child command separator. Use that exact syntax in Step 4; do not hand-roll server lifecycle management.

- [ ] **Step 3: Install Playwright in the disposable verification environment and write the acceptance script before visual fixes**

```powershell
uv pip install --python `
  'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' playwright
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  -m playwright install chromium
```

Create `.superpowers/ui-smoke.py` with assertions for the approved requirements:

```python
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def assert_minimum_type(page):
    sizes = page.locator(".gradio-container :is(p,label,button,h1,h2,h3,.wrap)").evaluate_all(
        "els => els.filter(e => e.offsetParent !== null && e.textContent.trim()).map(e => "
        "parseFloat(getComputedStyle(e).fontSize))"
    )
    assert sizes and min(sizes) >= 20, min(sizes)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    desktop = browser.new_page(viewport={"width": 1440, "height": 900})
    desktop.goto("http://127.0.0.1:7861", wait_until="networkidle")
    assert_minimum_type(desktop)
    assert desktop.locator("#game-layout").evaluate("e => e.scrollWidth <= e.clientWidth")
    desktop.screenshot(path=ROOT / ".superpowers/ui-desktop.png", full_page=True)

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto("http://127.0.0.1:7861", wait_until="networkidle")
    assert_minimum_type(mobile)
    assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
    mobile.screenshot(path=ROOT / ".superpowers/ui-mobile.png", full_page=True)

    board_image = desktop.locator("#board img").first
    before = board_image.get_attribute("src")
    desktop.get_by_role("button", name="4", exact=True).click()
    desktop.wait_for_timeout(100)
    assert desktop.locator("#board").is_visible()
    desktop.wait_for_function(
        "before => document.querySelector('#board img')?.getAttribute('src') !== before",
        before,
    )
    desktop.wait_for_function(
        "() => document.querySelector('#turn-status')?.textContent.includes('輪到你')"
    )
    browser.close()
```

- [ ] **Step 4: Run the local Space and browser acceptance script under managed lifecycle**

```powershell
$env:GRADIO_SERVER_PORT='7861'
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  'C:\Users\3Hml\.codex\skills\webapp-testing\scripts\with_server.py' `
  --server "D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe scripts/run_space_local.py" `
  --port 7861 `
  -- `
  'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  .superpowers/ui-smoke.py
```

Expected: exit code `0`, no horizontal overflow, minimum computed font size at least `20`, the board remains visible while a move is pending, and both screenshots are created.

- [ ] **Step 5: Inspect desktop and mobile screenshots once as a batch**

Open both screenshots together. Check: board prominence, status hierarchy, no clipped radio text, restrained corners, no empty lower-right card area, and no browser console errors. If a selector misses Gradio internals, adjust only `APP_CSS`, rerun Task 1 tests and this script once, then stop polishing.

- [ ] **Step 6: Verify canonical/deployment parity**

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' -c `
  "from pathlib import Path; import hashlib; a=Path('src/az'); b=Path('space/az'); rel=lambda p:{x.relative_to(p) for x in p.rglob('*.py')}; assert rel(a)==rel(b); assert all(hashlib.sha256((a/r).read_bytes()).digest()==hashlib.sha256((b/r).read_bytes()).digest() for r in rel(a)); print('BUNDLE_PARITY=ok')"
```

Expected: `BUNDLE_PARITY=ok`.

- [ ] **Step 7: Commit only evidence-driven CSS corrections, if any**

If Step 5 required a correction:

```powershell
git add -- space/app.py tests/test_space_release.py
git commit -m "fix: refine Space responsive styles"
```

If no correction was required, do not create an empty commit.

---

### Task 4: End-to-end verification and publication

**Files:**
- Verify: `.github/workflows/ci.yml`
- Verify: `space/README.md`
- Upload through: `scripts/deploy_space.py`
- Do not upload: `.superpowers/`, `docs/superpowers/`, local screenshots, checkpoints, or model files.

**Interfaces:**
- Consumes: committed UI source and the generated deployment bundle.
- Produces: synchronized GitHub `main`, successful hosted CI, and a live HF Space at the existing URL.

- [ ] **Step 1: Run fresh local verification**

```powershell
git diff --check
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' -m pytest -q
uv pip check --python 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
git status --short --branch
```

Expected: all tests pass, installed packages are compatible, and only intentional commits are ahead of `origin/main`.

- [ ] **Step 2: Push GitHub and wait for hosted CI**

```powershell
git push origin main
gh run list --repo kuotunyu/alphazero-connect4 --workflow ci.yml --limit 1
```

Wait for the run attached to the pushed head SHA. Expected: `status=completed`, `conclusion=success`, and zero annotations.

- [ ] **Step 3: Verify the model pin before Space deployment**

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' -c `
  "from huggingface_hub import HfApi; s=HfApi().model_info('steven0226/alphazero-connect4').sha; assert s=='0ba2361fe4044af9f6bfadfa89997b46191077c7'; print('MODEL_SHA='+s)"
```

Expected: exact pinned SHA. Abort deployment if it differs.

- [ ] **Step 4: Deploy the existing Space bundle**

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  scripts/deploy_space.py
```

Expected: a new commit on `steven0226/connect4-arena`; no model repo modification.

- [ ] **Step 5: Wait for runtime readiness and execute live smoke**

Verify the Space runtime reaches `RUNNING`, root returns HTTP `200`, `/config` exposes exactly 21 components and 8 dependencies, and the config contains `0ba2361fe404`.

Then run a Gradio client game:

```python
from pathlib import Path
from gradio_client import Client

client = Client("https://steven0226-connect4-arena.hf.space", verbose=False)
start = client.predict("先手（紅）", 50, api_name="/on_new_game")
moved = client.predict(50, api_name="/partial_3")
assert Path(start[0]).read_bytes() != Path(moved[0]).read_bytes()
assert "輪到你" in moved[2]
assert abs(sum(float(x["confidence"]) for x in moved[1]["confidences"]) - 1.0) < 1e-6
```

Expected: new game, player column 4, AI reply, board update, player turn, and normalized probabilities all pass.

- [ ] **Step 6: Perform one bounded live visual confirmation**

At desktop and mobile viewport widths, confirm computed text sizes remain at least `20px`, there is no horizontal overflow, and clicking a column does not replace the board with the full orange loading spinner shown in the original report.

- [ ] **Step 7: Record final evidence**

Report GitHub commits, hosted CI URL, HF Space commit SHA, model SHA, local test count, desktop/mobile viewport results, and explicit publish `GO` or `NO-GO`. Do not claim completion unless every preceding verification has fresh evidence.
