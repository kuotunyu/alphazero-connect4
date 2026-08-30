"""Resolve the immutable Hugging Face model assets used by the Space."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Mapping, NamedTuple


DEFAULT_MODEL_REPO = "steven0226/alphazero-connect4"
DEFAULT_MODEL_REVISION = "0ba2361fe4044af9f6bfadfa89997b46191077c7"
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class ModelAssets(NamedTuple):
    repo_id: str
    revision: str
    config_path: Path
    weights_path: Path


def resolve_model_source(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Return a model repo and immutable Hub commit revision."""
    source = os.environ if environ is None else environ
    repo_id = source.get("MODEL_REPO", DEFAULT_MODEL_REPO)
    revision = source.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError("MODEL_REVISION must be a 40-character commit SHA")
    return repo_id, revision


def download_model_assets(
    *,
    downloader: Callable[..., str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModelAssets:
    """Download config and weights from the same immutable Hub revision."""
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    repo_id, revision = resolve_model_source(environ)
    config_path = Path(
        downloader(repo_id, "config.json", revision=revision)
    )
    weights_path = Path(
        downloader(repo_id, "model.safetensors", revision=revision)
    )
    return ModelAssets(
        repo_id=repo_id,
        revision=revision,
        config_path=config_path,
        weights_path=weights_path,
    )
