"""Configuration loading helpers."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def runs_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["runs_root"])


def gens_dir(cfg: dict, benchmark: str, model_key: str, template: str) -> Path:
    return runs_root(cfg) / benchmark / model_key / "gens" / template


def graded_dir(cfg: dict, benchmark: str, model_key: str, template: str) -> Path:
    return runs_root(cfg) / benchmark / model_key / "graded" / template


def aggregate_path(cfg: dict, benchmark: str, model_key: str, template: str) -> Path:
    return runs_root(cfg) / "aggregates" / f"{benchmark}.{model_key}.{template}.json"


def subset_path(cfg: dict, subset_name: str) -> Path:
    return runs_root(cfg) / "subsets" / f"{subset_name}.json"


def stage2_path(cfg: dict, benchmark: str) -> Path:
    return runs_root(cfg) / "stage2" / f"{benchmark}.json"
