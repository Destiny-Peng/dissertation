"""Experiment-spec schema and matrix expansion for Localization Lab."""

from __future__ import annotations

import copy
import itertools
import math
import re
from typing import Any, Mapping, Sequence

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

DEFAULT_BASE = {
    "data": {
        "population": "failure_only",
        "success_ratio": 0.0,
    },
    "target": {
        "kind": "hard",
        "sigma_pre": 3.0,
        "sigma_post": 3.0,
        "tau_event": 20.0,
    },
    "model": {
        "hidden": 16,
    },
    "loss": {
        "name": "bce",
        "distance_weight": 1.0,
        "ranking_weight": 1.0,
        "ranking_margin": 1.0,
    },
    "training": {
        "device": "auto",
        "batch_size": 32,
        "epochs": 300,
        "patience": 35,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "grad_clip": 5.0,
        "seed": 17,
        "train_fraction": 0.70,
        "val_fraction": 0.15,
    },
}

BUILTIN_PRESETS = {
    "bilstm_default": {
        "schema_version": 1,
        "name": "bilstm_default",
        "base": copy.deepcopy(DEFAULT_BASE),
        "sweep": [],
        "repeats": 5,
        "stages": [],
    },
    "label_loss_default": {
        "schema_version": 1,
        "name": "label_loss_default",
        "base": copy.deepcopy(DEFAULT_BASE),
        "repeats": 5,
        "sweep": [],
        "stages": [
            {
                "name": "label_selection",
                "sweep": [
                    {
                        "path": "target",
                        "values": [
                            {"kind": "hard", "sigma_pre": 3.0, "sigma_post": 3.0, "tau_event": 20.0},
                            {"kind": "gaussian", "sigma_pre": 1.0, "sigma_post": 1.0, "tau_event": 20.0},
                            {"kind": "gaussian", "sigma_pre": 2.0, "sigma_post": 2.0, "tau_event": 20.0},
                            {"kind": "gaussian", "sigma_pre": 3.0, "sigma_post": 3.0, "tau_event": 20.0},
                            {"kind": "gaussian", "sigma_pre": 5.0, "sigma_post": 5.0, "tau_event": 20.0},
                        ],
                    }
                ],
                "select": {
                    "metric": "in_interval_rate_mean",
                    "mode": "max",
                    "tie_breakers": [
                        {"metric": "mae_samples_mean", "mode": "min"},
                        {"metric": "mse_samples_mean", "mode": "min"},
                    ],
                },
            },
            {
                "name": "loss_selection",
                "sweep": [
                    {
                        "path": "loss.name",
                        "values": [
                            "bce",
                            "temporal_softmax_ce",
                            "temporal_softmax_ce_distance",
                            "temporal_softmax_ce_squared_distance",
                            "temporal_softmax_ce_ranking",
                            "temporal_softmax_ce_distance_ranking",
                        ],
                    }
                ],
                "select": {
                    "metric": "in_interval_rate_mean",
                    "mode": "max",
                    "tie_breakers": [
                        {"metric": "mae_samples_mean", "mode": "min"},
                        {"metric": "mse_samples_mean", "mode": "min"},
                    ],
                },
            },
        ],
    },
    "success_ratio_default": {
        "schema_version": 1,
        "name": "success_ratio_default",
        "base": {
            **copy.deepcopy(DEFAULT_BASE),
            "data": {"population": "failure_success", "success_ratio": 0.0},
        },
        "repeats": 5,
        "sweep": [
            {"path": "data.success_ratio", "values": [0.0, 0.5, 1.0, 2.0]},
            {"path": "model.hidden", "values": [16, 32]},
        ],
        "stages": [],
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_path(config: dict[str, Any], path: str, value: Any) -> None:
    if not path or path.startswith(".") or path.endswith("."):
        raise ValueError(f"invalid sweep path: {path!r}")
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = copy.deepcopy(value)


def expand(base: Mapping[str, Any], sweep: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not sweep:
        return [copy.deepcopy(dict(base))]
    paths: list[str] = []
    values: list[list[Any]] = []
    for item in sweep:
        path = str(item.get("path") or "").strip()
        options = item.get("values")
        if not path or not isinstance(options, list) or not options:
            raise ValueError("each sweep dimension requires path and non-empty values")
        if path in paths:
            raise ValueError(f"duplicate sweep path: {path}")
        paths.append(path)
        values.append(options)
    result = []
    for combination in itertools.product(*values):
        config = copy.deepcopy(dict(base))
        for path, value in zip(paths, combination):
            set_path(config, path, value)
        result.append(config)
    return result


def _finite(value: Any, name: str, *, minimum: float | None = None, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be > 0")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def validate_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    target = config.get("target", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    training = config.get("training", {})

    population = str(data.get("population", "failure_only"))
    if population not in {"failure_only", "failure_success"}:
        raise ValueError("data.population must be failure_only or failure_success")
    ratio = _finite(data.get("success_ratio", 0.0), "data.success_ratio", minimum=0.0)
    if population == "failure_only" and ratio != 0:
        raise ValueError("failure_only requires data.success_ratio=0")

    kind = str(target.get("kind", "hard"))
    if kind not in {"hard", "gaussian"}:
        raise ValueError("target.kind must be hard or gaussian")
    _finite(target.get("tau_event", 20.0), "target.tau_event", positive=True)
    if kind == "gaussian":
        _finite(target.get("sigma_pre", 3.0), "target.sigma_pre", positive=True)
        _finite(target.get("sigma_post", 3.0), "target.sigma_post", positive=True)

    hidden = int(model.get("hidden", 16))
    if hidden < 1 or hidden > 512:
        raise ValueError("model.hidden must be between 1 and 512")

    loss_name = str(loss.get("name", "bce"))
    allowed_losses = {
        "bce",
        "temporal_softmax_ce",
        "temporal_softmax_ce_distance",
        "temporal_softmax_ce_squared_distance",
        "temporal_softmax_ce_ranking",
        "temporal_softmax_ce_distance_ranking",
    }
    if loss_name not in allowed_losses:
        raise ValueError(f"unsupported loss.name: {loss_name}")
    if population == "failure_success" and ratio > 0 and loss_name != "bce":
        raise ValueError("success-negative training currently requires BCE")
    for key in ("distance_weight", "ranking_weight", "ranking_margin"):
        _finite(loss.get(key, 1.0), f"loss.{key}", minimum=0.0)

    device = str(training.get("device", "auto"))
    if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", device):
        raise ValueError("training.device must be auto, cpu, cuda, or cuda:N")
    batch_size = int(training.get("batch_size", 32))
    if batch_size < 1 or batch_size > 128:
        raise ValueError("training.batch_size must be between 1 and 128")
    for key in ("epochs", "patience"):
        if int(training.get(key, 1)) < 1:
            raise ValueError(f"training.{key} must be >= 1")
    _finite(training.get("learning_rate", 0.003), "training.learning_rate", positive=True)
    _finite(training.get("weight_decay", 0.0001), "training.weight_decay", minimum=0.0)
    _finite(training.get("grad_clip", 5.0), "training.grad_clip", positive=True)
    train_fraction = _finite(training.get("train_fraction", 0.70), "training.train_fraction", positive=True)
    val_fraction = _finite(training.get("val_fraction", 0.15), "training.val_fraction", positive=True)
    if train_fraction >= 1 or val_fraction >= 1 or train_fraction + val_fraction >= 1:
        raise ValueError("training train/val fractions must be <1 and leave test data")


def normalize_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or "localization_experiment").strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError("experiment name must use letters, numbers, dot, underscore, or hyphen")
    repeats = int(raw.get("repeats", 5))
    if repeats < 1 or repeats > 50:
        raise ValueError("repeats must be between 1 and 50")
    base = deep_merge(DEFAULT_BASE, raw.get("base", {}))
    sweep = copy.deepcopy(raw.get("sweep", []))
    stages = copy.deepcopy(raw.get("stages", []))
    if not isinstance(sweep, list) or not isinstance(stages, list):
        raise ValueError("sweep and stages must be arrays")
    if stages and sweep:
        raise ValueError("use either top-level sweep or stages, not both")

    if stages:
        inherited = copy.deepcopy(base)
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("each stage must be an object")
            stage_name = str(stage.get("name") or "").strip()
            if not NAME_RE.fullmatch(stage_name):
                raise ValueError(f"invalid stage name: {stage_name!r}")
            stage_base = deep_merge(inherited, stage.get("base", {}))
            configs = expand(stage_base, stage.get("sweep", []))
            for config in configs:
                validate_config(config)
            inherited = stage_base
    else:
        for config in expand(base, sweep):
            validate_config(config)

    return {
        "schema_version": 1,
        "name": name,
        "base": base,
        "sweep": sweep,
        "stages": stages,
        "repeats": repeats,
    }


def estimate_runs(spec: Mapping[str, Any]) -> dict[str, int]:
    normalized = normalize_spec(spec)
    repeats = int(normalized["repeats"])
    if normalized["stages"]:
        configurations = sum(
            len(expand(deep_merge(normalized["base"], stage.get("base", {})), stage.get("sweep", [])))
            for stage in normalized["stages"]
        )
    else:
        configurations = len(expand(normalized["base"], normalized["sweep"]))
    return {
        "configurations": configurations,
        "training_runs": configurations * repeats,
    }
