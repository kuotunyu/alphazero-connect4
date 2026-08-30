"""Release-boundary tests for the self-contained Hugging Face Space."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from scripts.deploy_space import bundle_az


ROOT = Path(__file__).resolve().parents[1]


def load_model_assets_module():
    module_path = ROOT / "space" / "model_assets.py"
    spec = importlib.util.spec_from_file_location("space_model_assets", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_space_app_module(monkeypatch):
    import sys

    class StubModel:
        def load_state_dict(self, state):
            return None

        def eval(self):
            return self

    class StubEvaluator:
        def __init__(self, model, device):
            self.model = model
            self.device = device

    torch_stub = ModuleType("torch")
    torch_stub.load = lambda *args, **kwargs: {"config": {}, "model": {}}
    torch_stub.set_num_threads = lambda threads: None

    model_stub = ModuleType("az.model")
    model_stub.PolicyValueNet = StubModel
    model_stub.create_model = lambda config: StubModel()
    model_stub.NetEvaluator = StubEvaluator

    viz_stub = ModuleType("az.viz")
    viz_stub.render_board = lambda *args, **kwargs: None

    monkeypatch.setenv("AZ_LOCAL_WEIGHTS", "ui-test.pt")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "az.model", model_stub)
    monkeypatch.setitem(sys.modules, "az.viz", viz_stub)
    monkeypatch.syspath_prepend(str(ROOT / "space"))

    module_path = ROOT / "space" / "app.py"
    spec = importlib.util.spec_from_file_location("space_app_for_ui_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_model_source_pins_one_immutable_revision_for_all_assets(tmp_path):
    model_assets = load_model_assets_module()
    calls = []

    def downloader(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        return str(tmp_path / filename)

    assets = model_assets.download_model_assets(downloader=downloader, environ={})

    assert assets.repo_id == "steven0226/alphazero-connect4"
    assert assets.revision == "0ba2361fe4044af9f6bfadfa89997b46191077c7"
    assert assets.config_path == tmp_path / "config.json"
    assert assets.weights_path == tmp_path / "model.safetensors"
    assert calls == [
        (
            "steven0226/alphazero-connect4",
            "config.json",
            "0ba2361fe4044af9f6bfadfa89997b46191077c7",
        ),
        (
            "steven0226/alphazero-connect4",
            "model.safetensors",
            "0ba2361fe4044af9f6bfadfa89997b46191077c7",
        ),
    ]


def test_model_revision_override_must_also_be_immutable():
    model_assets = load_model_assets_module()

    with pytest.raises(ValueError, match="40-character commit SHA"):
        model_assets.resolve_model_source({"MODEL_REVISION": "main"})


def test_bundle_az_replaces_deployment_copy_from_canonical_source(tmp_path):
    source = tmp_path / "src" / "az"
    destination = tmp_path / "space" / "az"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "__init__.py").write_text("SOURCE = 'canonical'\n", encoding="utf-8")
    (source / "game.py").write_text("ROWS = 6\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "game.pyc").write_bytes(b"cache")
    (destination / "stale.py").write_text("STALE = True\n", encoding="utf-8")

    bundled = bundle_az(source=source, destination=destination)

    assert bundled == destination
    assert (destination / "__init__.py").read_text(encoding="utf-8") == (
        "SOURCE = 'canonical'\n"
    )
    assert (destination / "game.py").read_text(encoding="utf-8") == "ROWS = 6\n"
    assert not (destination / "stale.py").exists()
    assert not (destination / "__pycache__").exists()


def test_space_ui_contract_is_large_type_compact_and_responsive(monkeypatch):
    app = load_space_app_module(monkeypatch)
    config = app.demo.get_config_file()
    props = [component["props"] for component in config["components"]]
    hooks = {item.get("elem_id") for item in props}
    classes = {
        name
        for item in props
        for name in item.get("elem_classes", [])
    }

    assert {
        "app-header",
        "game-layout",
        "board-panel",
        "column-actions",
        "control-panel",
        "turn-status",
        "position-evaluation",
        "thinking-note",
    } <= hooks
    assert {"column-button", "game-control"} <= classes

    launch_options = {}
    monkeypatch.setattr(app.demo, "launch", lambda **options: launch_options.update(options))
    app.launch_app()
    css = launch_options["css"]
    for rule in (
        "font-size: 20px",
        "font-size: 22px",
        "font-size: 28px",
        "font-size: 30px",
        "font-size: 36px",
        "font-size: 31px",
        "border-radius: 4px",
        "@media (max-width: 900px)",
        "overflow-x: hidden",
    ):
        assert rule in css

    assert re.search(
        r"\.gradio-container\s*\{[^}]*max-width:\s*1660px[^}]*"
        r"margin-inline:\s*auto",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"#game-layout\s*\{[^}]*align-items:\s*flex-start", css, re.DOTALL
    )
    assert re.search(r"#board-panel\s*\{[^}]*flex:\s*62 1 0", css, re.DOTALL)
    assert re.search(r"#control-panel\s*\{[^}]*flex:\s*38 1 0", css, re.DOTALL)
    assert re.search(
        r"#board \.image-frame\s*\{[^}]*width:\s*100% !important",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"#board img\s*\{[^}]*width:\s*100% !important[^}]*"
        r"height:\s*auto !important",
        css,
        re.DOTALL,
    )


def test_space_board_omits_nonessential_image_toolbar(monkeypatch):
    app = load_space_app_module(monkeypatch)

    assert app.board.interactive is False
    assert app.board.buttons == []


def test_space_events_keep_outputs_visible_during_ai_compute(monkeypatch):
    app = load_space_app_module(monkeypatch)
    config = app.demo.get_config_file()
    click_events = [
        dependency
        for dependency in config["dependencies"]
        if any(event == "click" for _, event in dependency["targets"])
    ]

    assert len(click_events) == 8
    assert all(event["show_progress"] == "minimal" for event in click_events)
    assert all(len(event["show_progress_on"] or []) == 1 for event in click_events)
    assert all(len(event["outputs"]) == 4 for event in click_events)
    assert any(
        "AI 思考時會保留棋盤" in str(component["props"].get("value", ""))
        for component in config["components"]
    )
