# Connect4 Space Wide Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use substantially more of a wide desktop viewport while preserving the approved large-type, low-complexity mobile and desktop interface.

**Architecture:** Keep the Gradio component tree and all event handlers unchanged. Add one CSS contract to the existing release-boundary test, then make a minimal `APP_CSS` change that widens and centers the outer container, applies a controlled 62:38 desktop split, and lets the board image fill its panel without distortion.

**Tech Stack:** Python 3.11, Gradio 6.20.0, pytest, CSS, Playwright, GitHub Actions, Hugging Face Hub.

## Global Constraints

- Desktop `.gradio-container` maximum width is exactly `1660px` and remains centered.
- At `1920px`, the container width is at least 80% of the viewport.
- Above `900px`, the board/control target ratio is `62:38`; at or below `900px`, retain the current single-column order.
- Desktop board image width is at least 90% of the board panel without cropping or aspect-ratio distortion.
- Player-facing text remains at least `20px`; the existing restrained radius and color system stay unchanged.
- Do not add JavaScript, Gradio components, endpoints, game behavior, model changes, or new runtime dependencies.
- Keep the immutable model revision `0ba2361fe4044af9f6bfadfa89997b46191077c7` unchanged.

---

## File Map

- `tests/test_space_release.py`: owns the rendered Gradio/CSS release contract.
- `space/app.py`: owns the Space component tree and `APP_CSS`; only its CSS string changes.
- `.superpowers/wide-layout-smoke.py`: ignored, disposable Playwright evidence script created for local and live verification, then removed.

### Task 1: Add the wide-layout release contract

**Files:**
- Modify: `tests/test_space_release.py:125-162`
- Test: `tests/test_space_release.py`

**Interfaces:**
- Consumes: `load_space_app_module(monkeypatch)` and `app.launch_app()` from the existing test helper.
- Produces: selector-scoped CSS assertions that fail while the production CSS is still capped at `1240px`.

- [ ] **Step 1: Extend the existing UI contract with selector-scoped assertions**

Add `import re`, then append these assertions to
`test_space_ui_contract_is_large_type_compact_and_responsive`. They inspect
the CSS delivered through the real Gradio launch boundary and cannot be
satisfied by an unrelated `width: 100%` rule:

```python
    assert re.search(
        r"\.gradio-container\s*\{[^}]*max-width:\s*1660px[^}]*"
        r"margin-inline:\s*auto",
        css,
        re.DOTALL,
    )
    assert re.search(r"#board-panel\s*\{[^}]*flex:\s*62 1 0", css, re.DOTALL)
    assert re.search(r"#control-panel\s*\{[^}]*flex:\s*38 1 0", css, re.DOTALL)
    assert re.search(
        r"#board img\s*\{[^}]*width:\s*100% !important[^}]*"
        r"height:\s*auto !important",
        css,
        re.DOTALL,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  -m pytest tests/test_space_release.py::test_space_ui_contract_is_large_type_compact_and_responsive -q
```

Expected: FAIL because `max-width: 1660px` is absent and the current CSS still contains `max-width: 1240px`.

### Task 2: Implement the controlled wide layout

**Files:**
- Modify: `space/app.py:39-116,154-160,278-285`
- Test: `tests/test_space_release.py`

**Interfaces:**
- Consumes: the existing IDs `#game-layout`, `#board-panel`, `#board`, and `#control-panel`.
- Produces: an unchanged `demo` component tree and public event API with revised responsive CSS only.

- [ ] **Step 1: Make the smallest CSS change that satisfies the contract**

Change the outer container and desktop panel rules to this shape:

```css
.gradio-container {
  width: 100% !important;
  max-width: 1660px !important;
  margin-inline: auto !important;
  padding: 16px clamp(20px, 2vw, 32px) 24px !important;
}

#board-panel {
  min-width: 0;
  gap: 9px;
  flex: 62 1 0 !important;
}

#control-panel {
  min-width: 420px;
  flex: 38 1 0 !important;
}

#board .image-container {
  width: 100% !important;
}

#board img {
  width: 100% !important;
  height: auto !important;
  object-fit: contain !important;
}
```

Inside the existing `@media (max-width: 900px)` block, restore safe mobile sizing:

```css
  #board-panel,
  #control-panel {
    min-width: 0;
    flex: 1 1 auto !important;
  }
```

- [ ] **Step 2: Run the focused test and verify GREEN**

Run the focused pytest command from Task 1.

Expected: PASS.

- [ ] **Step 3: Run the complete test suite**

Run:

```powershell
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' -m pytest -q
```

Expected: `41 passed`.

- [ ] **Step 4: Commit the tested layout change**

```powershell
git add -- tests/test_space_release.py space/app.py
git commit -m "fix: use wide Space layout"
```

### Task 3: Verify the rendered responsive layout

**Files:**
- Create temporarily: `.superpowers/wide-layout-smoke.py`
- Verify: `space/app.py`

**Interfaces:**
- Consumes: a local `scripts/run_space_local.py` server and the rendered Gradio DOM.
- Produces: measured viewport, container, panel and image evidence; no tracked files.

- [ ] **Step 1: Start the Space locally against the existing smoke checkpoint**

Use the existing release environment and a clean local port:

```powershell
$env:GRADIO_SERVER_PORT='7861'
& 'D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe' `
  scripts/run_space_local.py checkpoints/smoke/best.pt
```

- [ ] **Step 2: Create one bounded Playwright measurement script**

The script must open the local URL at `390x844`, `1024x900`, `1440x900`, and
`1920x970`, then assert:

```python
assert document_scroll_width <= viewport_width + 1
assert minimum_player_facing_font_px >= 20
if viewport_width == 1920:
    assert container_width / viewport_width >= 0.80
if viewport_width >= 1024:
    assert board_image_width / board_panel_width >= 0.90
```

It must also capture desktop and mobile screenshots in `.superpowers/`, click
one column at 800 simulations, assert `#board` remains visible after 250 ms,
and report page/console errors.

- [ ] **Step 3: Inspect desktop and mobile screenshots once**

Confirm in one batched review that the board is not cropped, the control panel
is not stretched into a sparse form, the seven column buttons align with the
board, and the mobile layout retains the approved hierarchy. If the batch
shows defects, make one CSS correction batch, rerun the contract/full tests,
and perform only one final screenshot confirmation.

- [ ] **Step 4: Remove the disposable script and screenshots**

Delete only the exact `.superpowers/wide-layout-smoke.py` and generated image
files after recording their measurements. Do not recursively delete
`.superpowers` because it contains the approved brainstorming evidence.

### Task 4: Release through the existing gates

**Files:**
- Verify: repository and generated `space/az` bundle
- Remote update after all gates: GitHub `main`, then `steven0226/connect4-arena`

**Interfaces:**
- Consumes: `scripts/deploy_space.py`, GitHub Actions `ci.yml`, immutable model revision, and the live Space API.
- Produces: synchronized GitHub `main` and one new running Space commit.

- [ ] **Step 1: Run the complete local release gate**

```powershell
$python='D:\AI-Portfolio\.codex-tmp\alphazero-release-20260830\Scripts\python.exe'
& $python -m pytest -q
uv pip check --python $python
& $python scripts/deploy_space.py --dry-run
git diff --check
git status --short --branch
```

Expected: `41 passed`, 69 compatible packages, dry-run bundle success, no diff
errors, and no uncommitted tracked files.

- [ ] **Step 2: Push GitHub and require hosted CI success**

```powershell
git push origin main
gh run list --workflow ci.yml --branch main --limit 3 `
  --json databaseId,headSha,status,conclusion,url
```

Wait for the run whose `headSha` equals local `HEAD`; require `conclusion` to
be `success`. Do not deploy the Space if it fails.

- [ ] **Step 3: Verify the immutable model and deploy the Space**

Use `HfApi().model_info(..., revision=MODEL_REVISION)` and assert the returned
SHA is exactly `0ba2361fe4044af9f6bfadfa89997b46191077c7`, then run:

```powershell
& $python scripts/deploy_space.py
```

Wait until `HfApi().get_space_runtime("steven0226/connect4-arena").stage` is
`RUNNING` and record the new Space commit SHA.

- [ ] **Step 4: Run live API and browser smoke tests**

Require root HTTP 200, 21 components, 8 dependencies, the exact model pin,
one successful new game and move, plus the same 390/1024/1440/1920 layout
assertions used locally. Confirm the live browser console has no errors.

- [ ] **Step 5: Verify final synchronization and report**

```powershell
git status --short --branch
git rev-list --left-right --count origin/main...main
```

Expected: clean `main`, `0 0`, successful GitHub CI, and a `RUNNING` Space.
Report the Git commit, CI URL, Space commit, model pin, test counts and final
publish GO/NO-GO.
