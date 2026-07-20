"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config and resolve its relative dataset path locally."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    raw_root = Path(str(config["path"]))
    candidates = [
        Path.cwd() / raw_root,
        path.parent.parent / raw_root,
        path.parent / raw_root,
    ]
    resolved_root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if resolved_root is None:
        raise FileNotFoundError(
            f"Dataset path {raw_root} was not found relative to {Path.cwd()} or {path.parent}"
        )
    config["path"] = str(resolved_root.resolve())

    names = config.get("names")
    if isinstance(names, dict):
        config["names"] = [names[index] for index in range(int(config["nc"]))]
    return config
