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
        "signal_mode": "fused",
        "pca_components": 64,
        "source_run_root": "",
        "population": "failure_only",
        "success_ratio": 0.0,
        "challenge_set_name": "",
        "force_train_rollout_ids": [],
    },
    "target": {
        "kind": "hard",
        "sigma_pre": 3.0,
        "sigma_post": 3.0,
        "tau_event": 20.0,
    },
    "model": {
        "hidden": 16,
        "num_layers": 1,
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
        "parallel_workers": 4,
        "epochs": 300,
        "patience": 35,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "grad_clip": 5.0,
        "seed": 17,
        "split_seed": None,
        "vary_model_seed": True,
        "vary_split_seed": True,
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
        "variants": [],
        "repeats": 5,
        "stages": [],
    },
    "input_signal_default": {
        "schema_version": 1,
        "name": "input_signal_default",
        "base": copy.deepcopy(DEFAULT_BASE),
        "sweep": [{
            "path": "data.signal_mode",
            "values": ["incremental", "forward", "backward", "fused", "perspectives_6d", "fused_perspectives_8d"],
        }],
        "variants": [],
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
                "sweep": [],
                "variants": [
                    {
                        "name": "hard",
                        "set": {
                            "target.kind": "hard",
                            "target.sigma_pre": 3.0,
                            "target.sigma_post": 3.0,
                            "target.tau_event": 20.0,
                        },
                    },
                    {
                        "name": "gaussian_sigma_1",
                        "set": {
                            "target.kind": "gaussian",
                            "target.sigma_pre": 1.0,
                            "target.sigma_post": 1.0,
                            "target.tau_event": 20.0,
                        },
                    },
                    {
                        "name": "gaussian_sigma_2",
                        "set": {
                            "target.kind": "gaussian",
                            "target.sigma_pre": 2.0,
                            "target.sigma_post": 2.0,
                            "target.tau_event": 20.0,
                        },
                    },
                    {
                        "name": "gaussian_sigma_3",
                        "set": {
                            "target.kind": "gaussian",
                            "target.sigma_pre": 3.0,
                            "target.sigma_post": 3.0,
                            "target.tau_event": 20.0,
                        },
                    },
                    {
                        "name": "gaussian_sigma_5",
                        "set": {
                            "target.kind": "gaussian",
                            "target.sigma_pre": 5.0,
                            "target.sigma_post": 5.0,
                            "target.tau_event": 20.0,
                        },
                    },
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
        "variants": [],
        "stages": [],
    },
}


LATENT_COMPARISON_PRESET_NAMES = ["latent_input_default"]
_latent_base = copy.deepcopy(DEFAULT_BASE)
_latent_base["data"].update(signal_mode="robodopamine_latent", pca_components=64)
_latent_base["training"]["split_seed"] = 17
_latent_variants = []
for _dimensions in (64, 32, 128):
    for _mode, _suffix in (
        ("robodopamine_latent", "latent"),
        ("robodopamine_latent_plus_fused", "latent_plus_fused"),
    ):
        _latent_variants.append({
            "name": f"pca{_dimensions}_{_suffix}",
            "set": {"data.signal_mode": _mode, "data.pca_components": _dimensions},
        })
_latent_variants.append({
    "name": "fused_progress",
    "set": {"data.signal_mode": "fused", "data.pca_components": 64},
})
BUILTIN_PRESETS["latent_input_default"] = {
    "schema_version": 1,
    "name": "latent_input_default",
    "label": "Latent input comparison (PCA-64 default)",
    "base": _latent_base,
    "sweep": [],
    "variants": _latent_variants,
    "stages": [],
    "repeats": 5,
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


def expand(
    base: Mapping[str, Any],
    sweep: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand independent sweep dimensions and optional coupled variants.

    Sweep dimensions form a Cartesian product. A variant is a named set of paths
    that move together, e.g. target.kind + sigma_pre + sigma_post + tau_event.
    If both are present, each independent sweep combination is crossed with each
    coupled variant.
    """
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

    combinations = list(itertools.product(*values)) if values else [()]
    variant_rows = list(variants or [])

    result: list[dict[str, Any]] = []
    for combination in combinations:
        swept = copy.deepcopy(dict(base))
        for path, value in zip(paths, combination):
            set_path(swept, path, value)

        if not variant_rows:
            result.append(swept)
            continue

        for variant in variant_rows:
            if not isinstance(variant, Mapping):
                raise ValueError("each variant must be an object")
            variant_name = str(variant.get("name") or "").strip()
            if not NAME_RE.fullmatch(variant_name):
                raise ValueError(f"invalid variant name: {variant_name!r}")
            assignments = variant.get("set", {})
            if not isinstance(assignments, Mapping) or not assignments:
                raise ValueError(f"variant {variant_name!r} requires a non-empty set object")
            config = copy.deepcopy(swept)
            for path, value in assignments.items():
                set_path(config, str(path), value)
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

    signal_mode = str(data.get("signal_mode", "fused"))
    if signal_mode not in {
        "incremental", "forward", "backward", "fused", "perspectives_6d", "fused_perspectives_8d",
        "robodopamine_latent", "robodopamine_latent_plus_fused"
    }:
        raise ValueError(
            "data.signal_mode must be incremental, forward, backward, fused, "
            "perspectives_6d, fused_perspectives_8d, robodopamine_latent, or robodopamine_latent_plus_fused"
        )
    if not isinstance(data.get("source_run_root", ""), str):
        raise ValueError("data.source_run_root must be a project-relative baseline run path")
    components = data.get("pca_components", 64)
    if isinstance(components, bool) or int(components) != components or not 1 <= int(components) <= 2560:
        raise ValueError("data.pca_components must be between 1 and 2560")
    population = str(data.get("population", "failure_only"))
    if population not in {"failure_only", "failure_success"}:
        raise ValueError("data.population must be failure_only or failure_success")
    ratio = _finite(data.get("success_ratio", 0.0), "data.success_ratio", minimum=0.0)
    if population == "failure_only" and ratio != 0:
        raise ValueError("failure_only requires data.success_ratio=0")
    forced_ids = data.get("force_train_rollout_ids", [])
    if not isinstance(forced_ids, list) or any(not isinstance(value, str) or not value for value in forced_ids):
        raise ValueError("data.force_train_rollout_ids must be an array of rollout-id strings")
    challenge_name = str(data.get("challenge_set_name", "") or "")
    if challenge_name and not NAME_RE.fullmatch(challenge_name):
        raise ValueError("data.challenge_set_name is invalid")

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
    num_layers = int(model.get("num_layers", 1))
    if num_layers < 1 or num_layers > 8:
        raise ValueError("model.num_layers must be between 1 and 8")

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
    parallel_workers = int(training.get("parallel_workers", 4))
    if parallel_workers < 1 or parallel_workers > 8:
        raise ValueError("training.parallel_workers must be between 1 and 8")
    for key in ("epochs", "patience"):
        if int(training.get(key, 1)) < 1:
            raise ValueError(f"training.{key} must be >= 1")
    _finite(training.get("learning_rate", 0.003), "training.learning_rate", positive=True)
    _finite(training.get("weight_decay", 0.0001), "training.weight_decay", minimum=0.0)
    _finite(training.get("grad_clip", 5.0), "training.grad_clip", positive=True)
    try:
        int(training.get("seed", 17))
    except (TypeError, ValueError) as exc:
        raise ValueError("training.seed must be an integer") from exc
    split_seed = training.get("split_seed")
    if split_seed is not None:
        try:
            int(split_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.split_seed must be an integer or null") from exc
    for key in ("vary_model_seed", "vary_split_seed"):
        if not isinstance(training.get(key, True), bool):
            raise ValueError(f"training.{key} must be boolean")
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
    variants = copy.deepcopy(raw.get("variants", []))
    stages = copy.deepcopy(raw.get("stages", []))
    if not isinstance(sweep, list) or not isinstance(variants, list) or not isinstance(stages, list):
        raise ValueError("sweep, variants, and stages must be arrays")
    if stages and (sweep or variants):
        raise ValueError("use either top-level sweep/variants or stages, not both")

    if stages:
        inherited = copy.deepcopy(base)
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("each stage must be an object")
            stage_name = str(stage.get("name") or "").strip()
            if not NAME_RE.fullmatch(stage_name):
                raise ValueError(f"invalid stage name: {stage_name!r}")
            stage_base = deep_merge(inherited, stage.get("base", {}))
            configs = expand(
                stage_base,
                stage.get("sweep", []),
                stage.get("variants", []),
            )
            for config in configs:
                validate_config(config)
            inherited = stage_base
    else:
        for config in expand(base, sweep, variants):
            validate_config(config)

    return {
        "schema_version": 1,
        "name": name,
        "base": base,
        "sweep": sweep,
        "variants": variants,
        "stages": stages,
        "repeats": repeats,
    }


def estimate_runs(spec: Mapping[str, Any]) -> dict[str, int]:
    normalized = normalize_spec(spec)
    repeats = int(normalized["repeats"])
    if normalized["stages"]:
        configurations = sum(
            len(expand(
                deep_merge(normalized["base"], stage.get("base", {})),
                stage.get("sweep", []),
                stage.get("variants", []),
            ))
            for stage in normalized["stages"]
        )
    else:
        configurations = len(expand(
            normalized["base"],
            normalized["sweep"],
            normalized["variants"],
        ))
    return {
        "configurations": configurations,
        "training_runs": configurations * repeats,
    }
