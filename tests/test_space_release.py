"""Release-boundary tests for the self-contained Hugging Face Space."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
