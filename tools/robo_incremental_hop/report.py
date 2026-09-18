from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import (
    CLEAN_FPR_CONSTRAINTS,
    CONSECUTIVE_NS,
    EPSILONS,
    KOFM_MS,
    aggregate_clean_metrics,
    aggregate_event_metrics,
    config_row,
    detector_mask,
    evaluate_event,
    positive_episode_count,
    recovery_metrics,
)
from .io import project_relative


def evaluate_all_configs(
    configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    events_by_rollout: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for event in events:
        events_by_rollout[
            str(event["rollout_id"])
        ].append(event)

    clean_by_id = {
        str(row["rollout_id"]): row
        for row in clean_rollouts
    }
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []

    for config in configs:
        masks = {
            rollout_id: detector_mask(
                signal["hops"], config
            )
            for rollout_id, signal in signals.items()
        }
        config_events: list[dict[str, Any]] = []
        config_clean: list[dict[str, Any]] = []

        for rollout_id, rollout_events in (
            events_by_rollout.items()
        ):
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            mask = masks[rollout_id]
            for event in rollout_events:
                metrics = evaluate_event(
                    signal["frames"],
                    mask,
                    int(
                        event[
                            "observable_onset_frame"
                        ]
                    ),
                )
                row = {
                    **config_row(config),
                    **event,
                    **metrics,
                    "native_sample_n": len(
                        signal["frames"]
                    ),
                    "signal_source": (
                        project_relative(
                            signal["prediction_path"]
                        )
                    ),
                }
                config_events.append(row)
                event_rows.append(row)

        for rollout_id, clean in clean_by_id.items():
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            mask = masks[rollout_id]
            positive = [
                index
                for index, value in enumerate(mask)
                if value
            ]
            row = {
                **config_row(config),
                **clean,
                "native_sample_n": len(mask),
                "positive_sample_n": len(positive),
                "positive_sample_fraction": (
                    len(positive) / len(mask)
                    if mask
                    else None
                ),
                "any_positive": bool(positive),
                "first_positive_frame": (
                    signal["frames"][positive[0]]
                    if positive
                    else None
                ),
                "positive_episode_n": (
                    positive_episode_count(mask)
                ),
                "incremental_source": (
                    project_relative(
                        signal["prediction_path"]
                    )
                ),
            }
            config_clean.append(row)
            clean_rows.append(row)

        summary_rows.append(
            {
                **config_row(config),
                **aggregate_event_metrics(
                    config_events
                ),
                **aggregate_clean_metrics(
                    config_clean
                ),
            }
        )

    return summary_rows, event_rows, clean_rows


def select_best_configs(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in summary_rows:
        by_family[
            str(row["detector_family"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for family in sorted(by_family):
        for constraint in CLEAN_FPR_CONSTRAINTS:
            eligible = [
                row
                for row in by_family[family]
                if (
                    row.get("clean_rollout_fpr")
                    is not None
                    and float(
                        row["clean_rollout_fpr"]
                    )
                    <= constraint + 1e-12
                    and row.get(
                        "event_recall_at_3"
                    )
                    is not None
                )
            ]
            if not eligible:
                result.append(
                    {
                        "detector_family": family,
                        "clean_fpr_constraint": constraint,
                        "selection_status": (
                            "no_eligible_config"
                        ),
                    }
                )
                continue

            def rank(
                row: Mapping[str, Any],
            ) -> tuple[
                float, float, float, str
            ]:
                recall = float(
                    row.get(
                        "event_recall_at_3"
                    )
                    or 0.0
                )
                delay = row.get(
                    "median_delay_samples"
                )
                delay_key = (
                    float(delay)
                    if delay is not None
                    else math.inf
                )
                fpr = float(
                    row.get(
                        "clean_rollout_fpr"
                    )
                    or 0.0
                )
                return (
                    -recall,
                    delay_key,
                    fpr,
                    str(row["config_id"]),
                )

            best = min(eligible, key=rank)
            result.append(
                {
                    "clean_fpr_constraint": constraint,
                    "selection_status": "selected",
                    **dict(best),
                }
            )
    return result


def summarize_breakdowns(
    configs: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    clean_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)

    for row in event_rows:
        events_by_config[
            str(row["config_id"])
        ].append(row)
    for row in clean_rows:
        clean_by_config[
            str(row["config_id"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for config in configs:
        config_id = str(config["config_id"])
        events = events_by_config.get(
            config_id, []
        )
        clean = clean_by_config.get(
            config_id, []
        )
        groups: list[
            tuple[
                str,
                str,
                list[Mapping[str, Any]],
                list[Mapping[str, Any]],
            ]
        ] = [
            (
                "overall",
                "all",
                list(events),
                list(clean),
            )
        ]

        for failure_type in sorted(
            {
                str(row["failure_type"])
                for row in events
            }
        ):
            groups.append(
                (
                    "failure_type",
                    failure_type,
                    [
                        row
                        for row in events
                        if str(
                            row["failure_type"]
                        )
                        == failure_type
                    ],
                    [],
                )
            )

        for outcome in (
            "terminal_failure",
            "recovered_success",
        ):
            subset = [
                row
                for row in events
                if row.get("outcome")
                == outcome
            ]
            if subset:
                groups.append(
                    (
                        "outcome",
                        outcome,
                        subset,
                        [],
                    )
                )

        task_keys = sorted(
            {
                str(row["task_key"])
                for row in events
            }
            | {
                str(row["task_key"])
                for row in clean
            }
        )
        for task_key in task_keys:
            groups.append(
                (
                    "task",
                    task_key,
                    [
                        row
                        for row in events
                        if str(
                            row["task_key"]
                        )
                        == task_key
                    ],
                    [
                        row
                        for row in clean
                        if str(
                            row["task_key"]
                        )
                        == task_key
                    ],
                )
            )

        for (
            dimension,
            value,
            event_subset,
            clean_subset,
        ) in groups:
            result.append(
                {
                    **config_row(config),
                    "group_dimension": dimension,
                    "group_value": value,
                    **aggregate_event_metrics(
                        event_subset
                    ),
                    **aggregate_clean_metrics(
                        clean_subset
                    ),
                }
            )
    return result


def selected_unique_configs(
    best_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config_by_id = {
        str(config["config_id"]): dict(config)
        for config in configs
    }
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in best_rows:
        if (
            row.get("selection_status")
            != "selected"
        ):
            continue
        config_id = str(row["config_id"])
        if (
            config_id not in seen
            and config_id in config_by_id
        ):
            seen.add(config_id)
            selected.append(
                config_by_id[config_id]
            )
    return selected


def build_recovery_rows(
    selected_configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    window_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recovery_events = [
        event
        for event in events
        if event.get("recovery_frame")
        is not None
    ]
    for config in selected_configs:
        for event in recovery_events:
            signal = signals.get(
                str(event["rollout_id"])
            )
            if signal is None:
                continue
            mask = detector_mask(
                signal["hops"], config
            )
            rows.append(
                {
                    **config_row(config),
                    **event,
                    **recovery_metrics(
                        signal["frames"],
                        signal["raw_hops"],
                        signal["hops"],
                        mask,
                        int(
                            event[
                                "recovery_frame"
                            ]
                        ),
                        window_samples=(
                            window_samples
                        ),
                    ),
                }
            )
    return rows


def task_cross_validation(
    configs: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Tune on all other tasks and evaluate the selected config on one held-out task."""
    task_keys = sorted(
        {str(row["task_key"]) for row in event_rows}
        | {str(row["task_key"]) for row in clean_rows}
    )
    config_by_id = {
        str(config["config_id"]): config
        for config in configs
    }
    events_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    clean_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in event_rows:
        events_by_config[
            str(row["config_id"])
        ].append(row)
    for row in clean_rows:
        clean_by_config[
            str(row["config_id"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for held_out in task_keys:
        train_summary: list[dict[str, Any]] = []
        for config in configs:
            config_id = str(config["config_id"])
            train_events = [
                row
                for row in events_by_config.get(config_id, [])
                if str(row["task_key"]) != held_out
            ]
            train_clean = [
                row
                for row in clean_by_config.get(config_id, [])
                if str(row["task_key"]) != held_out
            ]
            train_summary.append(
                {
                    **config_row(config),
                    **aggregate_event_metrics(train_events),
                    **aggregate_clean_metrics(train_clean),
                }
            )

        selected = select_best_configs(train_summary)
        for row in selected:
            base = {
                "held_out_task": held_out,
                "train_task_n": max(0, len(task_keys) - 1),
                "detector_family": row["detector_family"],
                "clean_fpr_constraint": row[
                    "clean_fpr_constraint"
                ],
                "selection_status": row["selection_status"],
            }
            if row["selection_status"] != "selected":
                result.append(base)
                continue

            config_id = str(row["config_id"])
            heldout_events = [
                item
                for item in events_by_config.get(config_id, [])
                if str(item["task_key"]) == held_out
            ]
            heldout_clean = [
                item
                for item in clean_by_config.get(config_id, [])
                if str(item["task_key"]) == held_out
            ]
            heldout = {
                **aggregate_event_metrics(heldout_events),
                **aggregate_clean_metrics(heldout_clean),
            }
            config = config_by_id[config_id]
            result.append(
                {
                    **base,
                    **config_row(config),
                    "train_event_n": row.get("event_n"),
                    "train_event_recall_at_3": row.get(
                        "event_recall_at_3"
                    ),
                    "train_median_delay_samples": row.get(
                        "median_delay_samples"
                    ),
                    "train_clean_rollout_n": row.get(
                        "clean_rollout_n"
                    ),
                    "train_clean_rollout_fpr": row.get(
                        "clean_rollout_fpr"
                    ),
                    **{
                        f"heldout_{key}": value
                        for key, value in heldout.items()
                    },
                }
            )
    return result

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_tradeoff(
    summary_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    plt = _plt()
    fig, axis = plt.subplots(
        figsize=(8, 5), dpi=150
    )
    grouped: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in summary_rows:
        if (
            row.get("clean_rollout_fpr")
            is not None
            and row.get(
                "event_recall_at_3"
            )
            is not None
        ):
            grouped[
                str(
                    row[
                        "detector_family"
                    ]
                )
            ].append(row)

    for family, rows in sorted(
        grouped.items()
    ):
        axis.scatter(
            [
                float(
                    row[
                        "clean_rollout_fpr"
                    ]
                )
                for row in rows
            ],
            [
                float(
                    row[
                        "event_recall_at_3"
                    ]
                )
                for row in rows
            ],
            label=family,
            s=20,
            alpha=0.7,
        )
    for threshold in (
        CLEAN_FPR_CONSTRAINTS
    ):
        axis.axvline(
            threshold,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
    axis.set_xlabel(
        "clean-rollout false-positive rate"
    )
    axis.set_ylabel(
        "event recall @ 3 native samples"
    )
    axis.set_title(
        "Robo-Dopamine hop detector trade-off"
    )
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    x_key: str,
    y_key: str,
    metric: str,
    title: str,
    output: Path,
) -> None:
    plt = _plt()
    index = {
        (row.get(y_key), row.get(x_key)): (
            row.get(metric)
        )
        for row in rows
    }
    matrix = [
        [
            (
                float(index[(y, x)])
                if index.get((y, x))
                is not None
                else math.nan
            )
            for x in x_values
        ]
        for y in y_values
    ]
    fig, axis = plt.subplots(
        figsize=(
            max(6, len(x_values) * 0.8),
            max(4, len(y_values) * 0.45),
        ),
        dpi=150,
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(
        range(len(x_values))
    )
    axis.set_xticklabels(
        [str(value) for value in x_values]
    )
    axis.set_yticks(
        range(len(y_values))
    )
    axis.set_yticklabels(
        [str(value) for value in y_values]
    )
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    for row_index, values in enumerate(
        matrix
    ):
        for column_index, value in enumerate(
            values
        ):
            if math.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.colorbar(
        image, ax=axis, label=metric
    )
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def plot_detector_heatmaps(
    summary_rows: Sequence[Mapping[str, Any]],
    plot_dir: Path,
) -> None:
    consecutive = [
        row
        for row in summary_rows
        if row["detector_family"]
        == "consecutive"
    ]
    for metric, suffix in (
        (
            "event_recall_at_3",
            "recall_at_3",
        ),
        (
            "clean_rollout_fpr",
            "clean_fpr",
        ),
    ):
        heatmap(
            consecutive,
            x_values=CONSECUTIVE_NS,
            y_values=EPSILONS,
            x_key="n",
            y_key="epsilon",
            metric=metric,
            title=(
                "Consecutive non-progress/regression: "
                + suffix
            ),
            output=(
                plot_dir
                / f"consecutive_{suffix}_heatmap.png"
            ),
        )

    for m in KOFM_MS:
        rows = [
            row
            for row in summary_rows
            if (
                row["detector_family"]
                == "k_of_m"
                and int(row["m"]) == m
            )
        ]
        ks = list(
            range(
                math.ceil(0.6 * m),
                m + 1,
            )
        )
        for metric, suffix in (
            (
                "event_recall_at_3",
                "recall_at_3",
            ),
            (
                "clean_rollout_fpr",
                "clean_fpr",
            ),
        ):
            heatmap(
                rows,
                x_values=EPSILONS,
                y_values=ks,
                x_key="epsilon",
                y_key="k",
                metric=metric,
                title=(
                    f"k-of-m, m={m}: "
                    + suffix
                ),
                output=(
                    plot_dir
                    / (
                        f"k_of_m_m{m}_"
                        f"{suffix}_heatmap.png"
                    )
                ),
            )


def plot_delay_distributions(
    best_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    selected: list[
        tuple[str, str]
    ] = []
    seen: set[str] = set()
    for row in best_rows:
        if (
            row.get("selection_status")
            != "selected"
        ):
            continue
        config_id = str(row["config_id"])
        if config_id in seen:
            continue
        seen.add(config_id)
        label = (
            f"{row['detector_family']} "
            f"≤{int(round(float(row['clean_fpr_constraint']) * 100))}%"
        )
        selected.append(
            (config_id, label)
        )

    pairs = []
    for config_id, label in selected:
        values = [
            float(row["delay_samples"])
            for row in event_rows
            if (
                str(row["config_id"])
                == config_id
                and row.get(
                    "delay_samples"
                )
                is not None
            )
        ]
        if values:
            pairs.append(
                (label, values)
            )
    if not pairs:
        return

    plt = _plt()
    fig, axis = plt.subplots(
        figsize=(
            max(8, len(pairs) * 0.65),
            5,
        ),
        dpi=150,
    )
    axis.boxplot(
        [values for _, values in pairs],
        labels=[
            label for label, _ in pairs
        ],
        showfliers=False,
    )
    axis.set_ylabel(
        "post-onset delay (native samples, 1-based)"
    )
    axis.set_title(
        "Detection-delay distributions for selected configs"
    )
    axis.tick_params(
        axis="x",
        rotation=60,
        labelsize=7,
    )
    axis.grid(
        axis="y", alpha=0.2
    )
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def _positive_spans(
    frames: Sequence[int],
    mask: Sequence[bool],
) -> list[tuple[float, float]]:
    if not frames:
        return []
    if len(frames) == 1:
        boundaries = [
            frames[0] - 0.5,
            frames[0] + 0.5,
        ]
    else:
        boundaries = [
            frames[0]
            - (
                frames[1] - frames[0]
            )
            / 2.0
        ]
        boundaries.extend(
            (left + right) / 2.0
            for left, right in zip(
                frames, frames[1:]
            )
        )
        boundaries.append(
            frames[-1]
            + (
                frames[-1]
                - frames[-2]
            )
            / 2.0
        )

    spans: list[
        tuple[float, float]
    ] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (
            start is not None
            and (
                not value
                or index
                == len(mask) - 1
            )
        ):
            end = (
                index
                if (
                    value
                    and index
                    == len(mask) - 1
                )
                else index - 1
            )
            spans.append(
                (
                    boundaries[start],
                    boundaries[end + 1],
                )
            )
            start = None
    return spans


def selected_plot_configs(
    best_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in best_rows:
        if (
            row.get("selection_status")
            == "selected"
        ):
            by_family[
                str(
                    row[
                        "detector_family"
                    ]
                )
            ].append(row)

    config_by_id = {
        str(config["config_id"]): dict(config)
        for config in configs
    }
    preference = {
        0.10: 0,
        0.05: 1,
        0.20: 2,
    }
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        row = min(
            by_family[family],
            key=lambda item: preference.get(
                float(
                    item[
                        "clean_fpr_constraint"
                    ]
                ),
                99,
            ),
        )
        config = config_by_id.get(
            str(row["config_id"])
        )
        if config:
            selected.append(config)
    return selected


def choose_representative_rollouts(
    explicit: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> list[str]:
    if explicit:
        unknown = [
            rollout_id
            for rollout_id in explicit
            if rollout_id not in signals
        ]
        if unknown:
            raise ValueError(
                "Representative rollout(s) "
                "have no usable hop signal: "
                f"{unknown}"
            )
        return list(
            dict.fromkeys(explicit)
        )[:limit]

    ordered: list[str] = []
    groups = [
        [
            event
            for event in events
            if event.get(
                "recovery_frame"
            )
            is not None
        ],
        [
            event
            for event in events
            if event.get("outcome")
            == "terminal_failure"
        ],
        list(events),
    ]
    for group in groups:
        for event in sorted(
            group,
            key=lambda row: (
                str(
                    row[
                        "failure_type"
                    ]
                ),
                str(
                    row["rollout_id"]
                ),
                int(
                    row["event_index"]
                ),
            ),
        ):
            rollout_id = str(
                event["rollout_id"]
            )
            if (
                rollout_id in signals
                and rollout_id
                not in ordered
            ):
                ordered.append(
                    rollout_id
                )
            if len(ordered) >= limit:
                return ordered
    return ordered


def plot_representative_rollouts(
    configs: Sequence[Mapping[str, Any]],
    rollout_ids: Sequence[str],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    plot_dir: Path,
    signal_mode: str = "incremental",
) -> None:
    events_by_rollout: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for event in events:
        events_by_rollout[
            str(event["rollout_id"])
        ].append(event)

    plt = _plt()
    for rollout_id in rollout_ids:
        signal = signals[rollout_id]
        frames = signal["frames"]
        hops = signal["hops"]
        rollout_events = sorted(
            events_by_rollout.get(
                rollout_id, []
            ),
            key=lambda row: int(
                row["event_index"]
            ),
        )

        for config in configs:
            mask = detector_mask(
                hops, config
            )
            fig, axis = plt.subplots(
                figsize=(10, 4.5),
                dpi=150,
            )
            axis.plot(
                frames,
                hops,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"{signal_mode} hop",
            )
            axis.axhline(
                0.0,
                linewidth=1.0,
                linestyle="--",
                label="zero hop",
            )
            for left, right in _positive_spans(
                frames, mask
            ):
                axis.axvspan(
                    left,
                    right,
                    alpha=0.15,
                )

            for event in rollout_events:
                axis.axvline(
                    int(
                        event[
                            "observable_onset_frame"
                        ]
                    ),
                    linestyle="-",
                    linewidth=1.0,
                    label="observable onset",
                )
                if (
                    event.get(
                        "recovery_frame"
                    )
                    is not None
                ):
                    axis.axvline(
                        int(
                            event[
                                "recovery_frame"
                            ]
                        ),
                        linestyle=":",
                        linewidth=1.2,
                        label="recovery",
                    )

            axis.set_xlabel(
                "video frame index "
                "(native Robo-Dopamine samples)"
            )
            axis.set_ylabel(
                (
                    "incremental hop (normalized [-1,1])"
                    if signal_mode == "incremental"
                    else f"{signal_mode} hop (saved native scale)"
                )
            )
            axis.set_title(
                f"{rollout_id} · {signal_mode} · "
                f"{config['detector_family']} · "
                f"{config['parameters_json']}"
            )
            handles, labels = (
                axis.get_legend_handles_labels()
            )
            unique: dict[str, Any] = {}
            for handle, label in zip(
                handles, labels
            ):
                unique.setdefault(
                    label, handle
                )
            axis.legend(
                unique.values(),
                unique.keys(),
                fontsize=8,
                loc="best",
            )
            axis.grid(alpha=0.2)
            fig.tight_layout()

            output = (
                plot_dir
                / "representative"
                / (
                    f"{rollout_id}__"
                    f"{config['config_id']}.png"
                )
            )
            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fig.savefig(output)
            plt.close(fig)
