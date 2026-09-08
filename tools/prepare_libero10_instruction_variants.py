#!/usr/bin/env python3
"""Prepare a separate LIBERO-10 single-subtask diagnostic manifest.

The source manifest and rollout/media files are read only. The generated
records keep the original video path and provenance, but receive distinct
variant IDs and result namespaces so they cannot be confused with existing
full-instruction baseline outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_BDDL_ROOT = PROJECT_ROOT / "repos/LIBERO/libero/libero/bddl_files/libero_10"
DEFAULT_OFFICIAL_ROOTS = (
    PROJECT_ROOT / "repos/LIBERO/libero/libero/bddl_files/libero_90",
    PROJECT_ROOT / "repos/LIBERO/libero/libero/bddl_files/libero_object",
    PROJECT_ROOT / "repos/LIBERO/libero/libero/bddl_files/libero_goal",
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "tools/lf3r_annotator/instruction_variants/libero_10_v1"
)
SOURCE_MANIFEST_RELATIVE = "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
OUTPUT_NAMESPACE = "outputs/baselines/instruction_variants/libero_10"
VARIANTS = ("full_instruction", "subtask_a", "subtask_b")
DIAGNOSTIC_ROLE = "instruction_variant_diagnostic"
DIAGNOSTIC_PARTITION = "instruction_variant_diagnostic"
SCHEMA_VERSION = 1


def subtask(
    *,
    label: str,
    instruction: str,
    goal_atom: str,
    subject: str,
    source_object: str,
    target: str | None,
    source_target: str | None,
    official_language_required: bool,
    semantic_validation: str = "official_bddl_goal_atom",
    execution_order_note: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "instruction": instruction,
        "goal_atom": goal_atom,
        "subject": subject,
        "source_object": source_object,
        "target": target,
        "source_target": source_target,
        "official_language_required": official_language_required,
        "semantic_validation": semantic_validation,
        "execution_order_note": execution_order_note,
    }


# A/B labels are canonical labels, not claims about temporal execution order.
# Ordinary tasks follow source clause order. Task 8 follows object identifiers:
# moka_pot_1 is the right pot and moka_pot_2 is the left pot.
TASK_SPECS: dict[int, dict[str, Any]] = {
    0: {
        "bddl_file": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
        "full_instruction": "put both the alphabet soup and the tomato sauce in the basket",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="pick up the alphabet soup and put it in the basket",
                goal_atom="(In alphabet_soup_1 basket_1_contain_region)",
                subject="alphabet soup",
                source_object="alphabet_soup_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="pick up the tomato sauce and put it in the basket",
                goal_atom="(In tomato_sauce_1 basket_1_contain_region)",
                subject="tomato sauce",
                source_object="tomato_sauce_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
        ],
    },
    1: {
        "bddl_file": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl",
        "full_instruction": "put both the cream cheese box and the butter in the basket",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="pick up the cream cheese box and put it in the basket",
                goal_atom="(In cream_cheese_1 basket_1_contain_region)",
                subject="cream cheese box",
                source_object="cream_cheese_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="pick up the butter and put it in the basket",
                goal_atom="(In butter_1 basket_1_contain_region)",
                subject="butter",
                source_object="butter_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
        ],
    },
    2: {
        "bddl_file": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
        "full_instruction": "turn on the stove and put the moka pot on it",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="turn on the stove",
                goal_atom="(Turnon flat_stove_1)",
                subject="stove",
                source_object="flat_stove_1",
                target=None,
                source_target=None,
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="put the moka pot on the stove",
                goal_atom="(On moka_pot_1 flat_stove_1_cook_region)",
                subject="moka pot",
                source_object="moka_pot_1",
                target="stove",
                source_target="flat_stove_1_cook_region",
                official_language_required=True,
            ),
        ],
    },
    3: {
        "bddl_file": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
        "full_instruction": "put the black bowl in the bottom drawer of the cabinet and close it",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="put the black bowl in the bottom drawer of the cabinet",
                goal_atom="(In akita_black_bowl_1 white_cabinet_1_bottom_region)",
                subject="black bowl",
                source_object="akita_black_bowl_1",
                target="bottom drawer of the cabinet",
                source_target="white_cabinet_1_bottom_region",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="close the bottom drawer of the cabinet",
                goal_atom="(Close white_cabinet_1_bottom_region)",
                subject="bottom drawer of the cabinet",
                source_object="white_cabinet_1_bottom_region",
                target=None,
                source_target=None,
                official_language_required=True,
            ),
        ],
    },
    4: {
        "bddl_file": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        "full_instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="put the white mug on the left plate",
                goal_atom="(On porcelain_mug_1 plate_1)",
                subject="white mug",
                source_object="porcelain_mug_1",
                target="left plate",
                source_target="plate_1",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="put the yellow and white mug on the right plate",
                goal_atom="(On white_yellow_mug_1 plate_2)",
                subject="yellow and white mug",
                source_object="white_yellow_mug_1",
                target="right plate",
                source_target="plate_2",
                official_language_required=True,
            ),
        ],
    },
    5: {
        "bddl_file": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        "full_instruction": "pick up the book and place it in the back compartment of the caddy",
        "compatibility": "incompatible_single_atomic_goal",
        "incompatibility_reason": (
            "The official task has one unique goal atom. Splitting the surface "
            "phrase into 'pick up' and 'place' would create an invalid or "
            "underspecified standalone instruction."
        ),
        "order_basis": "not_applicable",
        "subtasks": [],
    },
    6: {
        "bddl_file": "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
        "full_instruction": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="put the white mug on the plate",
                goal_atom="(On porcelain_mug_1 plate_1)",
                subject="white mug",
                source_object="porcelain_mug_1",
                target="plate",
                source_target="plate_1",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="put the chocolate pudding to the right of the plate",
                goal_atom="(On chocolate_pudding_1 living_room_table_plate_right_region)",
                subject="chocolate pudding",
                source_object="chocolate_pudding_1",
                target="right of the plate",
                source_target="living_room_table_plate_right_region",
                official_language_required=True,
            ),
        ],
    },
    7: {
        "bddl_file": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
        "full_instruction": "put both the alphabet soup and the cream cheese box in the basket",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="pick up the alphabet soup and put it in the basket",
                goal_atom="(In alphabet_soup_1 basket_1_contain_region)",
                subject="alphabet soup",
                source_object="alphabet_soup_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
            subtask(
                label="subtask_b",
                instruction="pick up the cream cheese box and put it in the basket",
                goal_atom="(In cream_cheese_1 basket_1_contain_region)",
                subject="cream cheese box",
                source_object="cream_cheese_1",
                target="basket",
                source_target="basket_1_contain_region",
                official_language_required=True,
            ),
        ],
    },
    8: {
        "bddl_file": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
        "full_instruction": "put both moka pots on the stove",
        "compatibility": "compatible_two_atomic_goals_with_initial_condition",
        "order_basis": "official_object_identifier_order_only",
        "auxiliary_goal_atoms": ["(Turnon flat_stove_1)"],
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="put the right moka pot on the stove",
                goal_atom="(On moka_pot_1 flat_stove_1_cook_region)",
                subject="right moka pot",
                source_object="moka_pot_1",
                target="stove",
                source_target="flat_stove_1_cook_region",
                official_language_required=True,
                execution_order_note=(
                    "A is moka_pot_1, whose source init region is the right "
                    "moka-pot region; this is not an observed execution order."
                ),
            ),
            subtask(
                label="subtask_b",
                instruction="put the left moka pot on the stove",
                goal_atom="(On moka_pot_2 flat_stove_1_cook_region)",
                subject="left moka pot",
                source_object="moka_pot_2",
                target="stove",
                source_target="flat_stove_1_cook_region",
                official_language_required=False,
                semantic_validation="source_bddl_object_region_and_goal_atom",
                execution_order_note=(
                    "B is moka_pot_2, whose source init region is the left "
                    "moka-pot region; no official LIBERO-90 standalone file "
                    "for this left-object wording was found."
                ),
            ),
        ],
    },
    9: {
        "bddl_file": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
        "full_instruction": "put the yellow and white mug in the microwave and close it",
        "compatibility": "compatible_two_atomic_goals",
        "order_basis": "source_instruction_clause_order_only",
        "subtasks": [
            subtask(
                label="subtask_a",
                instruction="put the yellow and white mug in the microwave",
                goal_atom="(In white_yellow_mug_1 microwave_1_heating_region)",
                subject="yellow and white mug",
                source_object="white_yellow_mug_1",
                target="microwave",
                source_target="microwave_1_heating_region",
                official_language_required=False,
                semantic_validation="source_bddl_goal_atom_and_template",
            ),
            subtask(
                label="subtask_b",
                instruction="close the microwave",
                goal_atom="(Close microwave_1)",
                subject="microwave",
                source_object="microwave_1",
                target=None,
                source_target=None,
                official_language_required=True,
            ),
        ],
    },
}


class VariantError(RuntimeError):
    """Raised when the source or generated diagnostic manifest is invalid."""


def project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise VariantError(f"Path escapes project root: {path}") from error
    return path


def project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise VariantError(f"Path is outside project root: {path}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, content: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise VariantError(
            f"Refusing to overwrite existing diagnostic artifact: {project_relative(path)}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def extract_language(text: str, path: Path) -> str:
    match = re.search(r"\(:language\s+([^\n)]+)\)", text)
    if not match:
        raise VariantError(f"BDDL has no :language clause: {path}")
    return normalize(match.group(1))


def extract_balanced_form(text: str, marker: str, path: Path) -> str:
    start = text.find(marker)
    if start < 0:
        raise VariantError(f"BDDL has no {marker} form: {path}")
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise VariantError(f"Unbalanced {marker} form: {path}")


def extract_goal_atoms(text: str, path: Path) -> list[str]:
    section = extract_balanced_form(text, "(:goal", path)
    atoms = re.findall(r"\((In|On|Close|Turnon)\s+([^()]+)\)", section)
    if not atoms:
        raise VariantError(f"BDDL goal has no supported atomic predicates: {path}")
    return [normalize(f"({predicate} {arguments})") for predicate, arguments in atoms]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VariantError(f"Manifest does not exist: {project_relative(path)}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise VariantError(f"Invalid JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise VariantError(f"Manifest row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def validate_single_clause(instruction: str, task_id: int, label: str) -> None:
    normalized = normalize(instruction)
    if not normalized or normalized != instruction:
        raise VariantError(
            f"Instruction is not normalized lowercase text for task {task_id} {label}: "
            f"{instruction!r}"
        )
    if instruction[-1:] in ".;!?":
        raise VariantError(f"Instruction has trailing punctuation: {instruction!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9 ,'-]+", instruction):
        raise VariantError(
            f"Instruction contains unsupported grammar characters: {instruction!r}"
        )
    if not normalized.startswith(("pick up ", "put ", "turn on ", "close ")):
        raise VariantError(f"Instruction has no supported single-task verb: {instruction!r}")
    allowed_atomic_transfer = (
        normalized.startswith("pick up ")
        and " and put it " in normalized
        and normalized.count(" and ") == 1
    )
    allowed_compound_object_name = "yellow and white" in normalized
    if (
        " and " in normalized
        and not allowed_atomic_transfer
        and not allowed_compound_object_name
    ):
        raise VariantError(
            f"Instruction contains a second conjunction and is not single-subtask: "
            f"{instruction!r}"
        )


def load_official_languages(roots: Iterable[Path]) -> set[str]:
    languages: set[str] = set()
    for root in roots:
        if not root.is_dir():
            raise VariantError(f"Official BDDL root does not exist: {project_relative(root)}")
        for path in sorted(root.rglob("*.bddl")):
            languages.add(extract_language(path.read_text(encoding="utf-8"), path))
    return languages


def load_task_metadata(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise VariantError(f"Task metadata does not exist: {project_relative(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("libero_10")
    if not isinstance(tasks, dict):
        raise VariantError("Task metadata has no libero_10 mapping")
    return {int(task_id): str(description) for task_id, description in tasks.items()}


def validate_task_specs(
    *,
    bddl_root: Path,
    official_roots: Iterable[Path],
    task_metadata: dict[int, str],
) -> dict[int, dict[str, Any]]:
    official_languages = load_official_languages(official_roots)
    validated: dict[int, dict[str, Any]] = {}
    if set(TASK_SPECS) != set(range(10)):
        raise VariantError("TASK_SPECS must cover exactly LIBERO-10 tasks 0-9")

    for task_id in range(10):
        spec = TASK_SPECS[task_id]
        bddl_path = bddl_root / spec["bddl_file"]
        if not bddl_path.is_file():
            raise VariantError(f"Missing LIBERO-10 BDDL: {project_relative(bddl_path)}")
        text = bddl_path.read_text(encoding="utf-8")
        bddl_language = extract_language(text, bddl_path)
        goal_atoms = extract_goal_atoms(text, bddl_path)
        expected_full = normalize(spec["full_instruction"])
        if bddl_language != expected_full:
            raise VariantError(
                f"Task {task_id} full instruction disagrees with BDDL: "
                f"{bddl_language!r} != {expected_full!r}"
            )
        if task_metadata.get(task_id) != spec["full_instruction"]:
            raise VariantError(
                f"Task {task_id} full instruction disagrees with task_metadata.json"
            )

        subtask_results: list[dict[str, Any]] = []
        for item in spec["subtasks"]:
            label = str(item["label"])
            instruction = str(item["instruction"])
            validate_single_clause(instruction, task_id, label)
            goal_atom = normalize(str(item["goal_atom"]))
            if goal_atom not in goal_atoms:
                raise VariantError(
                    f"Task {task_id} {label} goal atom is absent from source BDDL: "
                    f"{item['goal_atom']}"
                )
            if str(item["source_object"]).lower() not in goal_atom:
                raise VariantError(
                    f"Task {task_id} {label} source object is not represented by goal atom"
                )
            source_target = item.get("source_target")
            if source_target and str(source_target).lower() not in goal_atom:
                raise VariantError(
                    f"Task {task_id} {label} target is not represented by goal atom"
                )
            if item.get("official_language_required") and normalize(instruction) not in official_languages:
                raise VariantError(
                    f"Task {task_id} {label} is not an official atomic language template: "
                    f"{instruction!r}"
                )
            subtask_results.append(
                {
                    "label": label,
                    "instruction": instruction,
                    "goal_atom": goal_atom,
                    "official_language_match": normalize(instruction) in official_languages,
                    "semantic_validation": item["semantic_validation"],
                    "source_object": item["source_object"],
                    "source_target": source_target,
                }
            )

        if spec["compatibility"].startswith("compatible") and len(subtask_results) != 2:
            raise VariantError(f"Task {task_id} must have exactly two subtask variants")
        if spec["compatibility"].startswith("incompatible") and subtask_results:
            raise VariantError(f"Incompatible task {task_id} unexpectedly has subtask variants")
        selected_atoms = [item["goal_atom"] for item in subtask_results]
        if len(selected_atoms) != len(set(selected_atoms)):
            raise VariantError(f"Task {task_id} A/B variants share the same goal atom")
        for auxiliary in spec.get("auxiliary_goal_atoms", []):
            if normalize(auxiliary) not in goal_atoms:
                raise VariantError(
                    f"Task {task_id} auxiliary goal atom is absent from source BDDL: {auxiliary}"
                )
        validated[task_id] = {
            "task_id": task_id,
            "bddl_path": bddl_path,
            "bddl_language": bddl_language,
            "goal_atoms": goal_atoms,
            "subtasks": subtask_results,
            "official_language_count": len(official_languages),
        }
    return validated


def source_record_fields(record: dict[str, Any]) -> tuple[str, ...]:
    return (
        "id",
        "task_suite",
        "task_id",
        "episode_index",
        "task_description",
        "ground_truth_outcome",
        "video_path",
        "csv_path",
        "dataset_role",
        "analysis_partition",
        "run_name",
    )


def select_source_rows(
    rows: list[dict[str, Any]], project_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suite_rows = [row for row in rows if row.get("task_suite") == "libero_10"]
    selected = [
        row
        for row in suite_rows
        if row.get("dataset_role") == "primary_natural"
        and row.get("analysis_partition") == "natural_observation"
    ]
    excluded = [row for row in suite_rows if row not in selected]
    if not selected:
        raise VariantError("No primary natural LIBERO-10 rows found")

    seen_ids: set[str] = set()
    for row in selected:
        rollout_id = row.get("id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise VariantError("Selected source row has no valid id")
        if rollout_id in seen_ids:
            raise VariantError(f"Duplicate source rollout ID: {rollout_id}")
        seen_ids.add(rollout_id)
        if not isinstance(row.get("task_id"), int):
            raise VariantError(f"Source row has no integer task_id: {rollout_id}")
        video_value = row.get("video_path")
        if not isinstance(video_value, str) or not video_value:
            raise VariantError(f"Source row has no video_path: {rollout_id}")
        video_path = project_path(video_value, project_root)
        if not video_path.is_file():
            raise VariantError(
                f"Source video is missing for {rollout_id}: "
                f"{project_relative(video_path, project_root)}"
            )
        if row.get("task_id") not in TASK_SPECS:
            raise VariantError(f"Unexpected LIBERO-10 task ID in source: {row.get('task_id')}")
    return (
        sorted(
            selected,
            key=lambda row: (
                int(row["task_id"]),
                int(row.get("episode_index", 0)),
                str(row["id"]),
            ),
        ),
        excluded,
    )


def make_variant_row(
    *,
    source: dict[str, Any],
    condition: str,
    subtask_item: dict[str, Any] | None,
    spec: dict[str, Any],
    validated_spec: dict[str, Any],
    source_manifest_path: Path,
    source_manifest_hash: str,
    bddl_hash: str,
    project_root: Path,
) -> dict[str, Any]:
    source_id = str(source["id"])
    full_instruction = str(source["task_description"])
    if condition == "full_instruction":
        instruction = full_instruction
        subtask_index = None
        subtask_label = None
        instruction_type = "original_full_instruction"
        instruction_condition = "original"
    else:
        if subtask_item is None:
            raise VariantError(f"Missing subtask item for condition {condition}")
        instruction = str(subtask_item["instruction"])
        subtask_index = 0 if condition == "subtask_a" else 1
        subtask_label = "A" if condition == "subtask_a" else "B"
        instruction_type = "counterfactual_single_subtask"
        instruction_condition = "counterfactual"

    variant_id = f"{source_id}--{condition}"
    result_namespace = f"{OUTPUT_NAMESPACE}/{condition}"
    row = dict(source)
    for field in source_record_fields(source):
        row[f"source_{field}"] = source.get(field)
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "variant_schema": "lf3r.libero10.instruction_variants.v1",
            "id": variant_id,
            "source_rollout_id": source_id,
            "source_manifest": project_relative(source_manifest_path, project_root),
            "source_manifest_sha256": source_manifest_hash,
            "source_record_sha256": sha256_json(source),
            "source_video_path": source.get("video_path"),
            "source_csv_path": source.get("csv_path"),
            "source_annotation_path": (
                f"annotations/failure_annotations/v1/records/{source_id}.json"
                if (
                    project_root
                    / f"annotations/failure_annotations/v1/records/{source_id}.json"
                ).is_file()
                else None
            ),
            "source_task_bddl": project_relative(
                validated_spec["bddl_path"], project_root
            ),
            "source_task_bddl_sha256": bddl_hash,
            "task_description": instruction,
            "instruction": instruction,
            "original_full_instruction": full_instruction,
            "instruction_variant": condition,
            "condition": condition,
            "instruction_type": instruction_type,
            "instruction_condition": instruction_condition,
            "counterfactual_instruction": (
                instruction if instruction_condition == "counterfactual" else None
            ),
            "task_variant_compatibility": spec["compatibility"],
            "subtask_label": subtask_label,
            "subtask_index": subtask_index,
            "subtask_subject": (
                subtask_item.get("subject") if subtask_item is not None else None
            ),
            "subtask_target": (
                subtask_item.get("target") if subtask_item is not None else None
            ),
            "subtask_goal_atom": (
                normalize(str(subtask_item["goal_atom"]))
                if subtask_item is not None
                else None
            ),
            "subtask_semantic_validation": (
                subtask_item.get("semantic_validation")
                if subtask_item is not None
                else None
            ),
            "subtask_order_basis": spec["order_basis"],
            "execution_order_assumption": False,
            "execution_order_note": (
                "A/B labels are canonical task/object labels only; no observed "
                "trajectory execution order is assumed."
                if subtask_item is None or not subtask_item.get("execution_order_note")
                else subtask_item["execution_order_note"]
            ),
            "dataset_role": DIAGNOSTIC_ROLE,
            "analysis_partition": DIAGNOSTIC_PARTITION,
            "source_dataset_role": source.get("dataset_role"),
            "source_analysis_partition": source.get("analysis_partition"),
            "output_namespace": result_namespace,
            "baseline_run_namespace": result_namespace,
            "baseline_output_dir": f"{result_namespace}/{source_id}",
            "result_separation": (
                "Do not merge this row or its outputs with the source full-instruction "
                "baseline namespace."
            ),
            "ground_truth_outcome_provenance": (
                "Copied from the source rollout/evaluator record; it is not a "
                "counterfactual outcome label."
            ),
        }
    )
    return row


def validate_generated_rows(
    *,
    source_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    validated_specs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {str(row["id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise VariantError("Source rows contain duplicate IDs")
    if len({str(row["id"]) for row in generated_rows}) != len(generated_rows):
        raise VariantError("Generated variant IDs are not unique")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        source_id = row.get("source_rollout_id")
        if source_id not in source_by_id:
            raise VariantError(f"Generated row has unknown source rollout: {source_id}")
        source = source_by_id[source_id]
        condition = row.get("condition")
        spec = TASK_SPECS[int(source["task_id"])]
        by_source[str(source_id)].append(row)
        if row.get("id") == source_id or not str(row["id"]).endswith(f"--{condition}"):
            raise VariantError(f"Variant ID is not safely namespaced: {row.get('id')}")
        if row.get("video_path") != source.get("video_path"):
            raise VariantError(f"Video path changed for source rollout: {source_id}")
        if row.get("source_video_path") != source.get("video_path"):
            raise VariantError(f"Source video link is not preserved: {source_id}")
        if row.get("original_full_instruction") != source.get("task_description"):
            raise VariantError(f"Original instruction was not preserved: {source_id}")
        if row.get("source_task_id") != source.get("task_id"):
            raise VariantError(f"Task ID provenance changed: {source_id}")
        if row.get("source_episode_index") != source.get("episode_index"):
            raise VariantError(f"Episode provenance changed: {source_id}")
        if row.get("dataset_role") != DIAGNOSTIC_ROLE:
            raise VariantError(f"Diagnostic row has source dataset role: {source_id}")
        if row.get("analysis_partition") != DIAGNOSTIC_PARTITION:
            raise VariantError(f"Diagnostic row has source analysis partition: {source_id}")
        if not str(row.get("output_namespace", "")).startswith(
            f"{OUTPUT_NAMESPACE}/"
        ):
            raise VariantError(f"Variant output namespace is not isolated: {source_id}")
        if f"/{condition}/" not in str(row.get("baseline_output_dir", "")):
            raise VariantError(f"Variant output directory is ambiguous: {source_id}")

        if condition == "full_instruction":
            if row.get("instruction_type") != "original_full_instruction":
                raise VariantError(f"Full row is not marked original: {source_id}")
            if row.get("task_description") != source.get("task_description"):
                raise VariantError(f"Full instruction changed: {source_id}")
        elif condition in ("subtask_a", "subtask_b"):
            index = 0 if condition == "subtask_a" else 1
            expected = spec["subtasks"][index]
            if row.get("task_description") != expected["instruction"]:
                raise VariantError(f"Subtask instruction mismatch: {source_id} {condition}")
            if row.get("instruction_type") != "counterfactual_single_subtask":
                raise VariantError(f"Subtask is not marked counterfactual: {source_id}")
            if row.get("execution_order_assumption") is not False:
                raise VariantError(f"Subtask claims execution order: {source_id} {condition}")
        else:
            raise VariantError(f"Unknown generated condition: {condition}")

    expected_conditions = {
        task_id: (
            ["full_instruction", "subtask_a", "subtask_b"]
            if spec["subtasks"]
            else ["full_instruction"]
        )
        for task_id, spec in TASK_SPECS.items()
    }
    for source_id, source in source_by_id.items():
        actual = sorted(row["condition"] for row in by_source.get(source_id, []))
        expected = sorted(expected_conditions[int(source["task_id"])])
        if actual != expected:
            raise VariantError(
                f"Variant set mismatch for {source_id}: actual={actual}, expected={expected}"
            )

    counts = Counter(str(row["condition"]) for row in generated_rows)
    task_counts = Counter(int(row["task_id"]) for row in generated_rows)
    compatible_tasks = [
        task_id for task_id, spec in TASK_SPECS.items() if spec["subtasks"]
    ]
    incompatible_tasks = [
        task_id for task_id, spec in TASK_SPECS.items() if not spec["subtasks"]
    ]
    return {
        "valid": True,
        "source_rollouts": len(source_rows),
        "generated_rows": len(generated_rows),
        "condition_counts": dict(sorted(counts.items())),
        "task_row_counts": {str(key): value for key, value in sorted(task_counts.items())},
        "compatible_tasks": compatible_tasks,
        "incompatible_tasks": incompatible_tasks,
        "compatible_rollouts": sum(
            int(row["task_id"]) in compatible_tasks for row in source_rows
        ),
        "validation_checks": [
            "Every source primary_natural LIBERO-10 rollout has one full_instruction row.",
            "Every compatible source rollout has exactly one subtask_a and one subtask_b row.",
            "Task 5 is retained as full_instruction only because its official BDDL has one unique goal atom.",
            "Every generated subtask goal atom is present in the corresponding official LIBERO-10 BDDL.",
            "Official atomic language templates are checked against the upstream BDDL language corpus.",
            "Non-template subtask wording is checked against its source BDDL object/goal atom and single-clause grammar.",
            "A/B labels are canonical and make no claim about observed trajectory execution order.",
            "Every variant preserves the source video path, task ID, episode ID, and full instruction.",
            "Every variant uses an isolated diagnostic role and output namespace.",
            "No source manifest, video, CSV, annotation, or existing baseline output is written by this tool.",
        ],
        "validated_task_spec_count": len(validated_specs),
    }


def json_ready_specs(
    validated_specs: dict[int, dict[str, Any]], project_root: Path
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for task_id, spec in TASK_SPECS.items():
        validated = validated_specs[task_id]
        payload[str(task_id)] = {
            "task_id": task_id,
            "full_instruction": spec["full_instruction"],
            "compatibility": spec["compatibility"],
            "incompatibility_reason": spec.get("incompatibility_reason"),
            "order_basis": spec["order_basis"],
            "source_bddl": project_relative(validated["bddl_path"], project_root),
            "source_bddl_sha256": sha256_file(validated["bddl_path"]),
            "bddl_language": validated["bddl_language"],
            "goal_atoms": validated["goal_atoms"],
            "auxiliary_goal_atoms": spec.get("auxiliary_goal_atoms", []),
            "subtasks": validated["subtasks"],
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and validate LIBERO-10 instruction-variant manifest"
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--bddl-root", type=Path, default=DEFAULT_BDDL_ROOT)
    parser.add_argument(
        "--official-bddl-root",
        type=Path,
        action="append",
        default=None,
        help="Additional official BDDL root; may be repeated.",
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/task_metadata.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the existing generated manifest without writing artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of artifacts only inside the new output namespace.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT.resolve()
    source_manifest = project_path(args.source_manifest, project_root)
    bddl_root = project_path(args.bddl_root, project_root)
    task_metadata_path = project_path(args.task_metadata, project_root)
    output_dir = project_path(args.output_dir, project_root)
    official_roots = tuple(
        project_path(path, project_root)
        for path in (args.official_bddl_root or DEFAULT_OFFICIAL_ROOTS)
    )

    source_manifest_hash = sha256_file(source_manifest)
    task_metadata = load_task_metadata(task_metadata_path)
    validated_specs = validate_task_specs(
        bddl_root=bddl_root,
        official_roots=official_roots,
        task_metadata=task_metadata,
    )
    all_rows = load_jsonl(source_manifest)
    source_rows, excluded_rows = select_source_rows(all_rows, project_root)

    generated_rows: list[dict[str, Any]] = []
    for source in source_rows:
        task_id = int(source["task_id"])
        spec = TASK_SPECS[task_id]
        validated = validated_specs[task_id]
        bddl_hash = sha256_file(validated["bddl_path"])
        generated_rows.append(
            make_variant_row(
                source=source,
                condition="full_instruction",
                subtask_item=None,
                spec=spec,
                validated_spec=validated,
                source_manifest_path=source_manifest,
                source_manifest_hash=source_manifest_hash,
                bddl_hash=bddl_hash,
                project_root=project_root,
            )
        )
        for subtask_item in spec["subtasks"]:
            generated_rows.append(
                make_variant_row(
                    source=source,
                    condition=str(subtask_item["label"]),
                    subtask_item=subtask_item,
                    spec=spec,
                    validated_spec=validated,
                    source_manifest_path=source_manifest,
                    source_manifest_hash=source_manifest_hash,
                    bddl_hash=bddl_hash,
                    project_root=project_root,
                )
            )

    validation = validate_generated_rows(
        source_rows=source_rows,
        generated_rows=generated_rows,
        validated_specs=validated_specs,
    )
    validation.update(
        {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_manifest": project_relative(source_manifest, project_root),
            "source_manifest_sha256": source_manifest_hash,
            "source_manifest_total_rows": len(all_rows),
            "source_libero10_rows": len(
                [row for row in all_rows if row.get("task_suite") == "libero_10"]
            ),
            "source_selected_rows": len(source_rows),
            "source_excluded_libero10_rows": len(excluded_rows),
            "source_excluded_ids": [
                row.get("id") for row in excluded_rows if row.get("id") is not None
            ],
            "output_namespace": OUTPUT_NAMESPACE,
            "output_directory": project_relative(output_dir, project_root),
            "schema_version": SCHEMA_VERSION,
        }
    )

    if args.check_only:
        existing_manifest = output_dir / "manifest.jsonl"
        existing_rows = load_jsonl(existing_manifest)
        if existing_rows != generated_rows:
            raise VariantError(
                "Existing diagnostic manifest differs from deterministic regenerated content"
            )
        print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
        print("LIBERO10_INSTRUCTION_VARIANTS_CHECK_OK")
        return 0

    existing_files = list(output_dir.iterdir()) if output_dir.is_dir() else []
    if existing_files and not args.force:
        raise VariantError(
            f"Output namespace is non-empty; use --force only to rebuild it: "
            f"{project_relative(output_dir)}"
        )

    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in generated_rows
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "libero10_instruction_variants",
        "source_manifest": project_relative(source_manifest, project_root),
        "source_manifest_sha256": source_manifest_hash,
        "source_task_metadata": project_relative(task_metadata_path, project_root),
        "output_namespace": OUTPUT_NAMESPACE,
        "output_directory": project_relative(output_dir, project_root),
        "conditions": list(VARIANTS),
        "source_selection": {
            "task_suite": "libero_10",
            "dataset_role": "primary_natural",
            "analysis_partition": "natural_observation",
        },
        "generated_at": validation["generated_at"],
        "source_files_untouched": True,
        "existing_baseline_outputs_untouched": True,
    }
    task_variants = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "libero10_instruction_variants",
        "source_manifest": project_relative(source_manifest, project_root),
        "source_manifest_sha256": source_manifest_hash,
        "output_namespace": OUTPUT_NAMESPACE,
        "label_semantics": (
            "subtask_a and subtask_b are canonical source-task/object labels. "
            "They do not encode which subtask the recorded trajectory executed first."
        ),
        "tasks": json_ready_specs(validated_specs, project_root),
    }
    readme = f"""LIBERO-10 instruction-variant diagnostic manifest

This directory is a separate diagnostic dataset description generated from
{SOURCE_MANIFEST_RELATIVE}. It does not replace or modify the source manifest,
videos, frame sidecars, annotations, or existing baseline outputs.

The source selection is task_suite=libero_10, dataset_role=primary_natural,
and analysis_partition=natural_observation. The source manifest hash is:

    {source_manifest_hash}

The generated manifest contains one full_instruction row for every selected
rollout. Tasks 0-4 and 6-9 are compatible with two independent atomic goals,
so each of their rollouts also has subtask_a and subtask_b rows. Task 5 is
retained as full_instruction only: its official BDDL has one unique goal atom,
and splitting the surface wording into pick up versus place would not produce
two valid standalone LIBERO task instructions.

subtask_a and subtask_b are canonical labels based on source clause order or,
for task 8, official object identifier order (moka_pot_1 right then
moka_pot_2 left). They make no assumption about which subtask was executed
first in the original video.

Every row keeps the original video_path, task ID, episode ID, source rollout
ID, and original_full_instruction. The actual evaluator instruction is in
task_description and instruction. Counterfactual rows are explicitly marked
with instruction_type=counterfactual_single_subtask,
condition=subtask_a|subtask_b, and
analysis_partition=instruction_variant_diagnostic.

Future outputs must use the isolated namespaces:

    outputs/baselines/instruction_variants/libero_10/full_instruction/
    outputs/baselines/instruction_variants/libero_10/subtask_a/
    outputs/baselines/instruction_variants/libero_10/subtask_b/

The manifest is not consumed by the existing full-instruction analysis by
default. Validate the deterministic artifact with:

    python3 tools/prepare_libero10_instruction_variants.py --check-only

No baseline inference is run by this preparation step.
"""
    atomic_write(output_dir / "manifest.jsonl", manifest_text, force=args.force)
    atomic_write(
        output_dir / "task_variants.json",
        json.dumps(task_variants, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        force=args.force,
    )
    atomic_write(
        output_dir / "validation_report.json",
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        force=args.force,
    )
    atomic_write(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        force=args.force,
    )
    atomic_write(output_dir / "README.md", readme, force=args.force)

    if sha256_file(source_manifest) != source_manifest_hash:
        raise VariantError("Source manifest changed while preparing variants")
    print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Manifest: {project_relative(output_dir / 'manifest.jsonl', project_root)}")
    print("LIBERO10_INSTRUCTION_VARIANTS_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VariantError as error:
        raise SystemExit(f"ERROR: {error}")
