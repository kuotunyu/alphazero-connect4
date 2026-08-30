# Connect4 Space Post-Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three verified Space accessibility/UI boundary defects without changing the Connect4 game, model, MCTS, layout, or deployment architecture.

**Architecture:** Keep the existing Gradio `Blocks` tree, four-output callback contract, Python board renderer, and centralized `APP_CSS`. Make three isolated changes in `space/app.py`: disable the output image toolbar, enlarge radio-label hit areas, and render the existing status string through a polite atomic HTML live region. Protect each change with a release-boundary test before implementation.

**Tech Stack:** Python 3.11, Gradio 6.20.0, pytest 8+, PowerShell, uv, Chromium through the in-app Browser, Hugging Face Space dry-run tooling.

## Global Constraints

- Work only in the local AlphaZero Connect4 repository.
- Keep `MODEL_REVISION` at `0ba2361fe4044af9f6bfadfa89997b46191077c7`.
- Do not modify `src/az`, generated `space/az`, MCTS, game rules, value evaluation, or session state.
- Keep Gradio at `6.20.0`; add no package or frontend dependency.
- Keep all player-facing Space text at a computed size of at least `20px`.
- Keep the current 1660px container, 62:38 desktop split, 900px breakpoint, low-radius visual language, and four callback outputs.
- Do not push GitHub, deploy Hugging Face Space, retrain a model, or change any remote setting.
- Use TDD for each behavioral or UI-contract change and preserve a clean working tree after verification.

---

## File Map

- `tests/test_space_release.py`: owns Gradio component, CSS, model pin, callback, and deployment-bundle release contracts.
- `space/app.py`: owns the Space component tree, `status_text`, callbacks, and centralized `APP_CSS`.
- `README.md`: reports the current automated test inventory and local verification command.
- `scripts/deploy_space.py`: read-only during this work; rebuilds ignored `space/az` from canonical `src/az` for the dry-run gate.
- `space/model_assets.py`: read-only during this work; owns the immutable model revision.

### Task 1: Remove the nonessential board image toolbar

**Files:**
- Modify: `tests/test_space_release.py:126-188`
- Modify: `space/app.py:433-440`
- Test: `tests/test_space_release.py`

**Interfaces:**
- Consumes: module-level `board: gr.Image` constructed by `load_space_app_module(monkeypatch)`.
- Produces: `board.buttons == []`; the board remains a non-interactive NumPy image output.

- [ ] **Step 1: Write the failing board-toolbar contract**

Add this test after `test_space_ui_contract_is_large_type_compact_and_responsive`:

```python
def test_space_board_omits_nonessential_image_toolbar(monkeypatch):
    app = load_space_app_module(monkeypatch)

    assert app.board.interactive is False
    assert app.board.buttons == []
```

- [ ] **Step 2: Run the focused test and confirm the expected failure**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest `
  tests/test_space_release.py::test_space_board_omits_nonessential_image_toolbar -q
```

Expected: FAIL because Gradio supplies the default `download`, `share`, and `fullscreen` buttons.

- [ ] **Step 3: Disable the toolbar at the component boundary**

Add the explicit `buttons` argument to the existing board component:

```python
board = gr.Image(
    render_board(game.INITIAL),
    type="numpy",
    show_label=False,
    interactive=False,
    buttons=[],
    elem_id="board",
)
```

- [ ] **Step 4: Run focused and release-boundary tests**

```powershell
& $python -m pytest `
  tests/test_space_release.py::test_space_board_omits_nonessential_image_toolbar -q
& $python -m pytest tests/test_space_release.py -q
```

Expected: both commands PASS; click dependencies remain eight and every event retains four outputs.

- [ ] **Step 5: Commit the isolated toolbar change**

```powershell
git add -- tests/test_space_release.py space/app.py
git commit -m "fix: remove nonessential Space image toolbar"
```

### Task 2: Raise radio-option touch targets to 48px

**Files:**
- Modify: `tests/test_space_release.py:126-188`
- Modify: `space/app.py:263-268`
- Test: `tests/test_space_release.py`

**Interfaces:**
- Consumes: `APP_CSS` applied by `launch_app()` and the existing `#side-control` and `#difficulty-control` element IDs.
- Produces: full radio labels with `min-height: 48px !important` and `touch-action: manipulation`.

- [ ] **Step 1: Write the failing CSS contract**

Add this test after the board-toolbar contract:

```python
def test_space_game_control_labels_are_48px_touch_targets(monkeypatch):
    app = load_space_app_module(monkeypatch)
    launch_options = {}
    monkeypatch.setattr(
        app.demo,
        "launch",
        lambda **options: launch_options.update(options),
    )

    app.launch_app()
    css = launch_options["css"]

    assert re.search(
        r"#side-control label,\s*#difficulty-control label\s*\{"
        r"[^}]*min-height:\s*48px !important"
        r"[^}]*touch-action:\s*manipulation",
        css,
        re.DOTALL,
    )
```

- [ ] **Step 2: Run the focused test and confirm the expected failure**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest `
  tests/test_space_release.py::test_space_game_control_labels_are_48px_touch_targets -q
```

Expected: FAIL because the existing selector has no `min-height` or `touch-action` declaration.

- [ ] **Step 3: Add the minimum touch-target declarations**

Change the existing selector in `APP_CSS` to:

```css
#side-control label,
#difficulty-control label {
  min-width: 0 !important;
  min-height: 48px !important;
  justify-content: center;
  border-radius: 4px !important;
  touch-action: manipulation;
}
```

- [ ] **Step 4: Run focused and release-boundary tests**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest `
  tests/test_space_release.py::test_space_game_control_labels_are_48px_touch_targets -q
& $python -m pytest tests/test_space_release.py -q
```

Expected: both commands PASS and the existing 20px, breakpoint, focus, and low-radius contracts remain green.

- [ ] **Step 5: Commit the touch-target change**

```powershell
git add -- tests/test_space_release.py space/app.py
git commit -m "fix: enlarge Space option touch targets"
```

### Task 3: Make status updates a polite atomic live region

**Files:**
- Modify: `tests/test_space_release.py:190-206`
- Modify: `space/app.py:8-21`
- Modify: `space/app.py:358-365`
- Modify: `space/app.py:446-447`
- Test: `tests/test_space_release.py`

**Interfaces:**
- Consumes: `status_text(s: dict) -> str` and the fourth output of `render_all(s)`.
- Produces: escaped internal status HTML containing one `<h3>` inside `role="status"`, `aria-live="polite"`, and `aria-atomic="true"`; the callback output count and position stay unchanged.

- [ ] **Step 1: Write the failing live-region contract**

Add this test before `test_space_events_keep_outputs_visible_during_ai_compute`:

```python
def test_space_turn_status_is_a_polite_atomic_live_region(monkeypatch):
    app = load_space_app_module(monkeypatch)
    status_config = app.status.get_config()

    assert status_config["name"] == "html"
    initial_value = str(status_config["value"])
    assert 'role="status"' in initial_value
    assert 'aria-live="polite"' in initial_value
    assert 'aria-atomic="true"' in initial_value
    assert "輪到你" in initial_value
```

- [ ] **Step 2: Run the focused test and confirm the expected failure**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest `
  tests/test_space_release.py::test_space_turn_status_is_a_polite_atomic_live_region -q
```

Expected: FAIL because the current component name is `markdown` and its value has no live-region attributes.

- [ ] **Step 3: Render safe status HTML without changing game state**

Add the standard-library import:

```python
from html import escape
```

Replace `status_text` with:

```python
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
```

Replace only the status component constructor:

```python
status = gr.HTML(
    status_text(initial_state()),
    elem_id="turn-status",
)
```

- [ ] **Step 4: Run the live-region and callback contract tests**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest `
  tests/test_space_release.py::test_space_turn_status_is_a_polite_atomic_live_region `
  tests/test_space_release.py::test_space_events_keep_outputs_visible_during_ai_compute -q
```

Expected: two tests PASS; every click event still has one `show_progress_on` target and four outputs.

- [ ] **Step 5: Run the entire release-boundary test file**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest tests/test_space_release.py -q
```

Expected: eight tests PASS.

- [ ] **Step 6: Commit the live-region change**

```powershell
git add -- tests/test_space_release.py space/app.py
git commit -m "fix: announce Space turn status accessibly"
```

### Task 4: Synchronize documentation and run the clean local release gate

**Files:**
- Modify: `README.md:69`
- Modify: `README.md:83`
- Verify: `pyproject.toml`
- Verify: `space/README.md`
- Verify: `space/model_assets.py`
- Verify: `scripts/deploy_space.py`
- Test: `tests/`

**Interfaces:**
- Consumes: 36 existing core/result tests and eight Space release-boundary tests after Tasks 1–3.
- Produces: truthful README counts, a compatible clean environment, a regenerated ignored deployment bundle, and full local release evidence.

- [ ] **Step 1: Confirm the collected test count before changing documentation**

```powershell
$python = 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest --collect-only -q
```

Expected: `44 tests collected`.

- [ ] **Step 2: Update the two stale README counts**

Change the repository tree description to:

```text
tests/             36 個核心／結果測試 + 8 個 Space release-boundary 測試
```

Change the local verification command comment to:

```text
pytest tests -q                      # 44 tests：36 既有 + 8 release-boundary
```

- [ ] **Step 3: Commit the factual documentation correction**

```powershell
git add -- README.md
git commit -m "docs: update Space release test count"
```

- [ ] **Step 4: Build a disposable clean Python 3.11 environment**

```powershell
$auditEnv = 'D:\AI-Portfolio\.codex-tmp\alphazero-hardening-20260830'
$auditPython = Join-Path $auditEnv 'Scripts\python.exe'
uv venv $auditEnv --python 3.11
uv pip install --python $auditPython `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  '.[dev,hub]' 'gradio==6.20.0'
uv pip check --python $auditPython
```

Expected: installation and `pip check` succeed with Gradio 6.20.0 and no broken requirements.

- [ ] **Step 5: Run complete tests and dependency vulnerability audit**

```powershell
$auditPython = 'D:\AI-Portfolio\.codex-tmp\alphazero-hardening-20260830\Scripts\python.exe'
& $auditPython -m pytest tests -q
uvx pip-audit . --format columns --progress-spinner off
```

Expected: 44 tests PASS and `No known vulnerabilities found`.

- [ ] **Step 6: Regenerate the deployment bundle without uploading**

```powershell
$auditPython = 'D:\AI-Portfolio\.codex-tmp\alphazero-hardening-20260830\Scripts\python.exe'
& $auditPython scripts/deploy_space.py --dry-run
git diff --no-index -- src/az space/az
if ($LASTEXITCODE -eq 0) { 'canonical/bundle: identical' } else { exit 1 }
```

Expected: the script prints `dry run — space/ ready, not uploading` and the canonical/bundle comparison is identical.

- [ ] **Step 7: Verify immutable release metadata and tracked state**

```powershell
rg -n "sdk_version: 6\.20\.0" space/README.md
rg -n "0ba2361fe4044af9f6bfadfa89997b46191077c7" `
  space/model_assets.py tests/test_space_release.py
git diff --check
git status --short
```

Expected: the Gradio pin and model SHA each match exactly, `git diff --check` is clean, and no generated or temporary artifact appears in tracked status.

### Task 5: Perform bounded local browser acceptance

**Files:**
- Verify: `space/app.py`
- Verify: `scripts/run_space_local.py`
- No tracked file is created or modified.

**Interfaces:**
- Consumes: the local Space started with `checkpoints/smoke/best.pt` and the in-app Browser control capability.
- Produces: desktop, 200%-equivalent, mobile, keyboard, touch-target, live-region, interaction, and console evidence for the final local GO/NO-GO decision.

- [ ] **Step 1: Start the local Space in a managed terminal session**

Run from the repository root with the clean environment:

```powershell
$auditPython = 'D:\AI-Portfolio\.codex-tmp\alphazero-hardening-20260830\Scripts\python.exe'
& $auditPython scripts/run_space_local.py checkpoints/smoke/best.pt
```

Wait for the local Gradio URL and keep the terminal session ID for cleanup. Do not start a visible background PowerShell window.

- [ ] **Step 2: Inspect desktop layout and the three changed boundaries**

Using the in-app Browser at a 1440px-wide viewport, verify:

```text
#board has no download/share/fullscreen toolbar buttons
#side-control label and #difficulty-control label each measure at least 48px high
#turn-status contains [role="status"][aria-live="polite"][aria-atomic="true"]
all player-facing computed font sizes are at least 20px
document.scrollWidth <= document.clientWidth
```

Expected: every assertion is true and no vacated toolbar strip remains above the board.

- [ ] **Step 3: Verify keyboard and responsive behavior**

At 640px width as a 200%-equivalent desktop check and at 390px mobile width, verify no horizontal overflow. Tab through a column button, both radio groups, and the new-game button; each active control must have a visible focus treatment. Confirm all seven column buttons remain reachable and the radio-label hit areas remain at least 48px high.

- [ ] **Step 4: Exercise all three MCTS settings**

For each of 50, 200, and 800 simulations:

```text
select the difficulty
start a new game
play one legal column
wait for the AI response
confirm the board remains visible while processing
confirm the final visible status says 輪到你 or reports a valid game result
```

Expected: three flows complete without a callback error, stale spinner, blank board, or changed game semantics.

- [ ] **Step 5: Check browser errors and stop the server**

Inspect console messages recorded during the session. Expected: no new application error. Stop the exact managed terminal session, close only the temporary audit tab, and confirm the repository remains clean.

- [ ] **Step 6: Record the local publish decision**

Report:

```text
Git commit SHAs for Tasks 1–4
44/44 pytest result
pip check result
pip-audit result
canonical src/az vs space/az result
desktop, 200%-equivalent, and mobile measurements
50/200/800 interaction results
unchanged model revision
local publish GO or NO-GO
```

The only acceptable `GO` means the local branch is reviewable and eligible for a separately authorized GitHub/Hugging Face release. It does not authorize or perform remote publication.
