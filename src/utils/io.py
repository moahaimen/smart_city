from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_results_dirs(root: str | Path) -> dict[str, Path]:
    root_path = ensure_dir(root)
    return {
        "root": root_path,
        "models": ensure_dir(root_path / "models"),
        "figures": ensure_dir(root_path / "figures"),
        "tables": ensure_dir(root_path / "tables"),
        "logs": ensure_dir(root_path / "logs"),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
