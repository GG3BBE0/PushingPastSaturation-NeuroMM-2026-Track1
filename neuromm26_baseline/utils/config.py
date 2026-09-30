"""Configuration loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


ENV_PATH = ".env"


def _load_dotenv(dotenv_path: str = ENV_PATH) -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    default_config = config.pop("default_config", None)
    if not default_config:
        return config

    default_path = Path(default_config)
    if not default_path.is_absolute() and not default_path.exists():
        default_path = (path.parent / default_path).resolve()
    base = _resolve_config(default_path)
    return _deep_merge(base, config)


def _parse_override_value(raw_value: str) -> Any:
    return yaml.safe_load(raw_value)


def _set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Override key must not be empty.")

    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def parse_override_strings(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = _parse_override_value(raw_value.strip())
    return overrides


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        _set_nested_value(config, key, value)
    return config


def load_config(path: str) -> dict[str, Any]:
    _load_dotenv()

    config_path = Path(path).resolve()
    config = _resolve_config(config_path)

    paths = config.setdefault("paths", {})
    paths["dataset_root"] = os.getenv("NEUROMM_DATASET_ROOT", paths.get("dataset_root", "neuromm26_datasets"))
    paths["output_root"] = os.getenv("NEUROMM_OUTPUT_ROOT", paths.get("output_root", "neuromm26_results"))
    paths["model_root"] = os.getenv("NEUROMM_MODEL_ROOT", paths.get("model_root", "models"))
    paths["eeg_feature_root"] = os.getenv(
        "NEUROMM_EEG_FEATURE_ROOT",
        paths.get("eeg_feature_root", "neuromm26_datasets/processed/features/eeg"),
    )
    paths["video_feature_root"] = os.getenv(
        "NEUROMM_VIDEO_FEATURE_ROOT",
        paths.get("video_feature_root", "neuromm26_datasets/processed/features"),
    )

    dataset = config.setdefault("dataset", {})
    dataset["eeg_feature_name"] = os.getenv(
        "NEUROMM_EEG_FEATURE_NAME",
        dataset.get("eeg_feature_name", "eeg"),
    )
    dataset["video_feature_name"] = os.getenv(
        "NEUROMM_VIDEO_FEATURE_NAME",
        dataset.get("video_feature_name"),
    )
    return config
