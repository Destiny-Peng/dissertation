from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


EPSILONS = (-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10)
CONSECUTIVE_NS = tuple(range(1, 9))
KOFM_MS = (3, 4, 5, 6, 8)
MEAN_MS = (2, 3, 4, 5, 6, 8)
MEAN_THRESHOLDS = (-0.10, -0.05, -0.02, 0.00, 0.02, 0.05)
REGRESSION_MS = (2, 3, 4, 5, 6, 8)
REGRESSION_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
STAGNATION_DELTAS = (0.01, 0.02, 0.05, 0.10)
CLEAN_FPR_CONSTRAINTS = (0.05, 0.10, 0.20)
RECALL_SAMPLE_WINDOWS = (1, 3, 5, 10, 20)
PRE_ONSET_LOOKBACK_SAMPLES = 10

CONFIG_FIELDS = (
    "config_id",
    "detector_family",
    "epsilon",
    "n",
    "m",
    "k",
    "theta",
    "A",
    "delta",
    "parameters_json",
)


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def parse_score_percent(pred: Any) -> float | None:
    if not isinstance(pred, str):
        return None
    match = re.search(
        r"<score>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*</score>",
        pred,
    )
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def detect_hop_scale(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate saved incremental hop scale and normalize only percentage form."""
    raw_hops = [finite_number(row.get("hop"), "incremental hop") for row in rows]
    if not raw_hops:
        raise ValueError("Incremental prediction file contains no hop values")

    norm_errors: list[float] = []
    pct_errors: list[float] = []
    informative = 0
    for row, hop in zip(rows, raw_hops):
        percent = parse_score_percent(row.get("pred"))
        if percent is None:
            continue
        if abs(percent) > 1e-12 or abs(hop) > 1e-12:
            informative += 1
        norm_errors.append(abs(hop - percent / 100.0))
        pct_errors.append(abs(hop - percent))

    norm_mae = statistics.median(norm_errors) if norm_errors else None
    pct_mae = statistics.median(pct_errors) if pct_errors else None
    max_abs = max(abs(value) for value in raw_hops)

    if informative and norm_mae is not None and pct_mae is not None:
        if norm_mae <= pct_mae:
            source_scale, divisor = "normalized_-1_to_1", 1.0
        else:
            source_scale, divisor = "percentage_points_-100_to_100", 100.0
        method = "pred_score_crosscheck"
    elif max_abs <= 1.000001:
        source_scale, divisor, method = "normalized_-1_to_1", 1.0, "range_fallback"
    elif max_abs <= 100.000001:
        source_scale, divisor, method = (
            "percentage_points_-100_to_100",
            100.0,
            "range_fallback",
        )
    else:
        raise ValueError(
            "Incremental hop range is incompatible with normalized/percentage "
            f"semantics: max_abs={max_abs}"
        )

    normalized = [value / divisor for value in raw_hops]
    if any(abs(value) > 1.000001 for value in normalized):
        raise ValueError("Normalized incremental hop exceeds [-1, 1]")

    return {
        "source_scale": source_scale,
        "normalization_divisor": divisor,
        "detection_method": method,
        "max_abs_raw_hop": max_abs,
        "score_pair_count": len(norm_errors),
        "informative_score_pair_count": informative,
        "normalized_score_mae": norm_mae,
        "percentage_score_mae": pct_mae,
        "raw_hops": raw_hops,
        "normalized_hops": normalized,
    }


def config_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in ("epsilon", "n", "m", "k", "theta", "A", "delta")
        if config.get(key) is not None
    }


def make_config(config_id: str, family: str, **parameters: Any) -> dict[str, Any]:
    config = {
        "config_id": config_id,
        "detector_family": family,
        "epsilon": None,
        "n": None,
        "m": None,
        "k": None,
        "theta": None,
        "A": None,
        "delta": None,
    }
    config.update(parameters)
    config["parameters_json"] = json.dumps(
        config_parameters(config), sort_keys=True, separators=(",", ":")
    )
    return config


def config_row(config: Mapping[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in CONFIG_FIELDS}


def build_detector_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    index = 0

    def add(family: str, **parameters: Any) -> None:
        nonlocal index
        index += 1
        configs.append(make_config(f"cfg{index:04d}", family, **parameters))

    for epsilon in EPSILONS:
        for n in CONSECUTIVE_NS:
            add("consecutive", epsilon=epsilon, n=n)
    for epsilon in EPSILONS:
        for m in KOFM_MS:
            for k in range(math.ceil(0.6 * m), m + 1):
                add("k_of_m", epsilon=epsilon, m=m, k=k)
    for m in MEAN_MS:
        for theta in MEAN_THRESHOLDS:
            add("window_mean", m=m, theta=theta)
    for m in REGRESSION_MS:
        for threshold in REGRESSION_THRESHOLDS:
            add("cumulative_regression", m=m, A=threshold)
    for delta in STAGNATION_DELTAS:
        for n in CONSECUTIVE_NS:
            add("stagnation_consecutive", delta=delta, n=n)
    for delta in STAGNATION_DELTAS:
        for m in KOFM_MS:
            for k in range(math.ceil(0.6 * m), m + 1):
                add("stagnation_k_of_m", delta=delta, m=m, k=k)
    return configs


def detector_mask(hops: Sequence[float], config: Mapping[str, Any]) -> list[bool]:
    """Return detector state on the unchanged native sample grid."""
    family = str(config["detector_family"])
    values = [finite_number(value, "normalized hop") for value in hops]
    mask = [False] * len(values)

    if family in {"consecutive", "stagnation_consecutive"}:
        n = int(config["n"])
        run = 0
        for index, hop in enumerate(values):
            evidence = (
                hop <= float(config["epsilon"])
                if family == "consecutive"
                else abs(hop) <= float(config["delta"])
            )
            run = run + 1 if evidence else 0
            mask[index] = run >= n
        return mask

    if family in {"k_of_m", "stagnation_k_of_m"}:
        m, k = int(config["m"]), int(config["k"])
        evidence = [
            hop <= float(config["epsilon"])
            if family == "k_of_m"
            else abs(hop) <= float(config["delta"])
            for hop in values
        ]
        for index in range(m - 1, len(values)):
            mask[index] = sum(evidence[index - m + 1 : index + 1]) >= k
        return mask

    if family == "window_mean":
        m, theta = int(config["m"]), float(config["theta"])
        for index in range(m - 1, len(values)):
            window = values[index - m + 1 : index + 1]
            mask[index] = sum(window) / m <= theta
        return mask

    if family == "cumulative_regression":
        m, threshold = int(config["m"]), float(config["A"])
        for index in range(m - 1, len(values)):
            window = values[index - m + 1 : index + 1]
            mask[index] = (
                sum(max(0.0, -value) for value in window) >= threshold
            )
        return mask

    raise ValueError(f"Unknown detector family: {family}")


def positive_episode_count(mask: Sequence[bool]) -> int:
    return sum(
        bool(value) and (index == 0 or not mask[index - 1])
        for index, value in enumerate(mask)
    )


def evaluate_event(
    frames: Sequence[int],
    mask: Sequence[bool],
    observable_onset_frame: int,
    *,
    episode_end_frame: int | None = None,
    episode_end_source: str | None = None,
    lookback_samples: int = PRE_ONSET_LOOKBACK_SAMPLES,
) -> dict[str, Any]:
    """Evaluate one human failure episode on the native sample grid.

    The event window starts at the first native sample at/after observable onset.
    An annotated recovery/terminal boundary or the next event onset is exclusive:
    detections at or after that boundary do not count for this episode. If no
    explicit boundary exists, the last available native sample is included.
    """
    if len(frames) != len(mask):
        raise ValueError("frames/mask length mismatch")
    onset = int(observable_onset_frame)
    end = int(episode_end_frame) if episode_end_frame is not None else None
    if end is not None and end <= onset:
        raise ValueError(
            f"episode_end_frame must be after observable onset: onset={onset}, end={end}"
        )

    post = [
        index
        for index, frame in enumerate(frames)
        if frame >= onset and (end is None or frame < end)
    ]
    anchor = post[0] if post else None
    detection = next((index for index in post if mask[index]), None)

    pre = [index for index, frame in enumerate(frames) if frame < onset]
    early = [index for index in pre if mask[index]]
    lookback = pre[-lookback_samples:]
    lookback_positive = sum(bool(mask[index]) for index in lookback)

    result: dict[str, Any] = {
        "episode_end_frame": end,
        "episode_end_source": episode_end_source or ("rollout_end" if end is None else "explicit"),
        "episode_end_exclusive": end is not None,
        "onset_anchor_frame": frames[anchor] if anchor is not None else None,
        "eligible_post_onset_samples": len(post),
        "eventual_recall": detection is not None,
        "detected": detection is not None,
        "detection_frame": frames[detection] if detection is not None else None,
        "delay_samples": None,
        "delay_frames": None,
        "early_positive_any": bool(early),
        "early_positive_count": len(early),
        "earliest_early_positive_frame": frames[early[0]] if early else None,
        "latest_early_positive_frame": frames[early[-1]] if early else None,
        "pre_onset_lookback_n": len(lookback),
        "pre_onset_positive_count": lookback_positive,
        "pre_onset_positive_fraction": (
            lookback_positive / len(lookback) if lookback else None
        ),
        "pre_onset_any_positive": lookback_positive > 0,
    }
    if detection is not None and anchor is not None:
        # 1-based: first native sample at/after onset is +1 sample.
        result["delay_samples"] = detection - anchor + 1
        result["delay_frames"] = int(frames[detection]) - onset
    for window in RECALL_SAMPLE_WINDOWS:
        delay = result["delay_samples"]
        result[f"recall_at_{window}"] = bool(
            delay is not None and delay <= window
        )
    return result


def evaluate_failure_rollout_from_start(
    frames: Sequence[int],
    mask: Sequence[bool],
) -> dict[str, Any]:
    """Evaluate a terminal-failure rollout with no annotated failure event.

    No onset is synthesized. The failure condition is treated as present from
    the beginning of the saved native signal. Delay is 1-based in native samples
    and frame delay is measured from the first saved native frame.
    """
    if len(frames) != len(mask):
        raise ValueError("frames/mask length mismatch")
    detection = next(
        (index for index, positive in enumerate(mask) if positive),
        None,
    )
    start_frame = 0
    result: dict[str, Any] = {
        "rollout_start_frame": start_frame,
        "eligible_samples": len(frames),
        "detected": detection is not None,
        "eventual_recall": detection is not None,
        "first_alarm_frame": (
            int(frames[detection]) if detection is not None else None
        ),
        "delay_samples": (
            detection + 1 if detection is not None else None
        ),
        "delay_frames": (
            int(frames[detection]) - start_frame
            if detection is not None and start_frame is not None
            else None
        ),
    }
    for window in RECALL_SAMPLE_WINDOWS:
        delay = result["delay_samples"]
        result[f"recall_at_{window}"] = bool(
            delay is not None and delay <= window
        )
    return result


def describe(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    data = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not data:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }

    def percentile(q: float) -> float:
        if len(data) == 1:
            return data[0]
        position = (len(data) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return data[lower]
        weight = position - lower
        return data[lower] * (1.0 - weight) + data[upper] * weight

    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "median": statistics.median(data),
        "p90": percentile(0.90),
        "min": data[0],
        "max": data[-1],
    }


def aggregate_event_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    detected = [row for row in rows if row.get("detected")]
    sample_delay = describe(row.get("delay_samples") for row in detected)
    frame_delay = describe(row.get("delay_frames") for row in detected)
    lookback_n = sum(int(row.get("pre_onset_lookback_n") or 0) for row in rows)
    lookback_positive = sum(
        int(row.get("pre_onset_positive_count") or 0) for row in rows
    )
    result: dict[str, Any] = {
        "event_n": n,
        "detected_event_n": len(detected),
        "detected_event_fraction": len(detected) / n if n else None,
        "event_recall_eventual": (
            sum(bool(row.get("eventual_recall")) for row in rows) / n
            if n
            else None
        ),
        "median_delay_samples": sample_delay["median"],
        "mean_delay_samples": sample_delay["mean"],
        "p90_delay_samples": sample_delay["p90"],
        "median_delay_frames": frame_delay["median"],
        "mean_delay_frames": frame_delay["mean"],
        "p90_delay_frames": frame_delay["p90"],
        "early_positive_event_fraction": (
            sum(bool(row.get("early_positive_any")) for row in rows) / n
            if n
            else None
        ),
        "pre_onset_10_any_positive_fraction": (
            sum(bool(row.get("pre_onset_any_positive")) for row in rows) / n
            if n
            else None
        ),
        "pre_onset_10_positive_sample_fraction": (
            lookback_positive / lookback_n if lookback_n else None
        ),
        "pre_onset_10_sample_n": lookback_n,
    }
    for window in RECALL_SAMPLE_WINDOWS:
        result[f"event_recall_at_{window}"] = (
            sum(bool(row.get(f"recall_at_{window}")) for row in rows) / n
            if n
            else None
        )
    return result


def aggregate_clean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    native_n = sum(int(row.get("native_sample_n") or 0) for row in rows)
    positive_n = sum(int(row.get("positive_sample_n") or 0) for row in rows)
    episodes = [int(row.get("positive_episode_n") or 0) for row in rows]
    return {
        "clean_rollout_n": n,
        "clean_rollout_fpr": (
            sum(bool(row.get("any_positive")) for row in rows) / n
            if n
            else None
        ),
        "clean_positive_sample_fraction": (
            positive_n / native_n if native_n else None
        ),
        "clean_positive_episode_total": sum(episodes),
        "clean_positive_episodes_mean": sum(episodes) / n if n else None,
        "clean_positive_episodes_median": (
            statistics.median(episodes) if episodes else None
        ),
        "clean_native_sample_n": native_n,
    }


def recovery_metrics(
    frames: Sequence[int],
    raw_hops: Sequence[float],
    hops: Sequence[float],
    mask: Sequence[bool],
    recovery_frame: int,
    *,
    window_samples: int,
) -> dict[str, Any]:
    recovery = int(recovery_frame)
    pre_or_at = [i for i, frame in enumerate(frames) if frame <= recovery]
    state = pre_or_at[-1] if pre_or_at else None
    post = [i for i, frame in enumerate(frames) if frame >= recovery]
    first_post = post[0] if post else None
    first_negative = next((i for i in post if not mask[i]), None)

    before = [
        i for i, frame in enumerate(frames) if frame < recovery
    ][-window_samples:]
    after = [
        i for i, frame in enumerate(frames) if frame >= recovery
    ][:window_samples]

    def stats(indices: Sequence[int], values: Sequence[float], prefix: str) -> dict[str, Any]:
        data = [float(values[index]) for index in indices]
        return {
            f"{prefix}_n": len(data),
            f"{prefix}_mean": sum(data) / len(data) if data else None,
            f"{prefix}_median": statistics.median(data) if data else None,
            f"{prefix}_min": min(data) if data else None,
            f"{prefix}_max": max(data) if data else None,
        }

    clearance_samples = None
    clearance_frames = None
    if first_negative is not None and first_post is not None:
        clearance_samples = first_negative - first_post + 1
        clearance_frames = int(frames[first_negative]) - recovery

    return {
        "recovery_state_frame": frames[state] if state is not None else None,
        "detector_positive_at_recovery_state": (
            bool(mask[state]) if state is not None else None
        ),
        "first_native_frame_at_or_after_recovery": (
            frames[first_post] if first_post is not None else None
        ),
        "first_detector_negative_frame_at_or_after_recovery": (
            frames[first_negative] if first_negative is not None else None
        ),
        "clearance_delay_samples": clearance_samples,
        "clearance_delay_frames": clearance_frames,
        **stats(before, hops, "hop_before_recovery"),
        **stats(after, hops, "hop_after_recovery"),
        **stats(before, raw_hops, "stored_hop_before_recovery"),
        **stats(after, raw_hops, "stored_hop_after_recovery"),
    }
