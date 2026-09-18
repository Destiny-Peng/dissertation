"""External ProcVLM procedure ontology, semantic canonicalizer, and V1 tracker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class Action:
    id: str
    predicate: str
    object: str
    target: str | None = None
    text: str = ""
    history: str = ""


@dataclass(frozen=True)
class Chain:
    id: str
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Procedure:
    task_id: str
    task: str
    chains: tuple[Chain, ...]
    verb_aliases: dict[str, tuple[str, ...]]
    object_aliases: dict[str, tuple[str, ...]]
    target_aliases: dict[str, tuple[str, ...]]

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.id for chain in self.chains for action in chain.actions)


@dataclass(frozen=True)
class ParseResult:
    parsed_actions: tuple[str, ...]
    remaining_ids: tuple[str, ...]
    parse_valid: bool
    observed_stage: dict[str, int] | None
    source: str
    errors: tuple[str, ...]


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def _alias_tuple(raw: Any, fallback: str) -> tuple[str, ...]:
    values = raw if isinstance(raw, list) else []
    aliases = [normalize_text(value) for value in values if isinstance(value, str)]
    normalized_fallback = normalize_text(fallback)
    if normalized_fallback and normalized_fallback not in aliases:
        aliases.append(normalized_fallback)
    return tuple(alias for alias in aliases if alias)


def load_procedure(path: str | Path) -> Procedure:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("procedure config must contain a JSON object")

    task_id = str(raw.get("task_id", "")).strip()
    task = str(raw.get("task", "")).strip()
    if not task_id or not task:
        raise ValueError("procedure config requires non-empty task_id and task")

    raw_actions = raw.get("actions")
    raw_chains = raw.get("chains")
    if not isinstance(raw_actions, dict) or not raw_actions:
        raise ValueError("procedure config requires a non-empty actions object")
    if not isinstance(raw_chains, dict) or not raw_chains:
        raise ValueError("procedure config requires a non-empty chains object")

    verb_aliases = {
        str(key): _alias_tuple(value, str(key))
        for key, value in (raw.get("verb_aliases") or {}).items()
    }
    object_aliases = {
        str(key): _alias_tuple(value, str(key).replace("_", " "))
        for key, value in (raw.get("objects") or {}).items()
    }
    target_aliases = {
        str(key): _alias_tuple(value, str(key).replace("_", " "))
        for key, value in (raw.get("targets") or {}).items()
    }

    action_map: dict[str, Action] = {}
    for action_id, spec in raw_actions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"action {action_id!r} must be an object")
        aid = str(action_id).strip()
        predicate = str(spec.get("predicate", "")).strip()
        obj = str(spec.get("object", "")).strip()
        target_value = spec.get("target")
        target = str(target_value).strip() if target_value not in (None, "") else None
        text = str(spec.get("text", "")).strip()
        history = str(spec.get("history", "")).strip()
        if not aid or not predicate or not obj:
            raise ValueError(f"action {action_id!r} requires predicate and object")
        if predicate not in verb_aliases:
            verb_aliases[predicate] = _alias_tuple([], predicate)
        if obj not in object_aliases:
            object_aliases[obj] = _alias_tuple([], obj.replace("_", " "))
        if target and target not in target_aliases:
            target_aliases[target] = _alias_tuple([], target.replace("_", " "))
        action_map[aid] = Action(aid, predicate, obj, target, text, history)

    chains: list[Chain] = []
    seen: set[str] = set()
    for chain_id, ids in raw_chains.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"chain {chain_id!r} must contain action IDs")
        actions: list[Action] = []
        for action_id in ids:
            aid = str(action_id).strip()
            if aid not in action_map:
                raise ValueError(f"chain {chain_id!r} references unknown action {aid!r}")
            if aid in seen:
                raise ValueError(f"canonical action {aid!r} appears in multiple chains")
            seen.add(aid)
            actions.append(action_map[aid])
        chains.append(Chain(str(chain_id), tuple(actions)))

    unused = sorted(set(action_map) - seen)
    if unused:
        raise ValueError("actions not assigned to a chain: " + ", ".join(unused))

    return Procedure(
        task_id=task_id,
        task=task,
        chains=tuple(chains),
        verb_aliases=verb_aliases,
        object_aliases=object_aliases,
        target_aliases=target_aliases,
    )


def extract_remaining_section(answer: str) -> tuple[str, str]:
    text = str(answer or "")
    marker = re.search(r"the following actions are required\s*:\s*", text, flags=re.IGNORECASE)
    source = "remaining_section" if marker else "fallback_full_reasoning"
    if marker:
        text = text[marker.end():]
    text = re.split(r"<progress>|\btherefore\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip(), source


def _action_lines(section: str) -> list[str]:
    lines: list[str] = []
    for raw_line in section.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        if line:
            lines.append(line)
    if not lines and section.strip():
        lines = [section.strip()]
    return lines


def _contains_alias(text: str, aliases: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return any(alias and re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases)


def _canonicalize_line(line: str, procedure: Procedure) -> str | None:
    matches: list[str] = []
    for chain in procedure.chains:
        for action in chain.actions:
            if not _contains_alias(line, procedure.verb_aliases[action.predicate]):
                continue
            if not _contains_alias(line, procedure.object_aliases[action.object]):
                continue
            if action.target and not _contains_alias(line, procedure.target_aliases[action.target]):
                continue
            matches.append(action.id)
    return matches[0] if len(matches) == 1 else None


def _suffix_stage(chain: Chain, remaining: set[str]) -> int | None:
    ids = [action.id for action in chain.actions]
    for stage in range(len(ids) + 1):
        if remaining == set(ids[stage:]):
            return stage
    return None


def parse_remaining_actions(answer: str, procedure: Procedure) -> ParseResult:
    section, source = extract_remaining_section(answer)
    explicit_none = bool(
        re.search(r"\b(?:none|no actions? remain(?:ing)?|nothing remains?)\b", section, re.I)
    )
    parsed_actions: list[str] = []
    matched_ids: list[str] = []

    if not explicit_none:
        for line in _action_lines(section):
            action_id = _canonicalize_line(line, procedure)
            if action_id is not None:
                parsed_actions.append(line)
                matched_ids.append(action_id)

    errors: list[str] = []
    ids = set(matched_ids)
    if explicit_none and ids:
        errors.append("remaining-actions section contains both actions and an explicit none marker")
    if not explicit_none and not ids:
        errors.append("no canonical remaining action could be parsed")

    observed: dict[str, int] = {}
    if not errors:
        for chain in procedure.chains:
            chain_ids = {action.id for action in chain.actions}
            stage = _suffix_stage(chain, ids & chain_ids)
            if stage is None:
                errors.append(
                    f"chain {chain.id} violates suffix property: remaining={sorted(ids & chain_ids)}"
                )
            else:
                observed[chain.id] = stage

    ordered = tuple(action_id for action_id in procedure.action_ids if action_id in ids)
    return ParseResult(
        parsed_actions=tuple(parsed_actions),
        remaining_ids=ordered,
        parse_valid=not errors,
        observed_stage=observed if not errors else None,
        source=source,
        errors=tuple(errors),
    )


def task_history_text(procedure: Procedure, persistent_stage: dict[str, int]) -> str:
    lines: list[str] = []
    for chain in procedure.chains:
        stage = int(persistent_stage.get(chain.id, 0))
        if stage <= 0:
            continue
        stage = min(stage, len(chain.actions))
        action = chain.actions[stage - 1]
        if action.history:
            lines.append(action.history)
        elif action.text:
            lines.append(f"The task has already completed: {action.text}.")
    return "\n".join(lines)


def build_stateful_history_prompt(task: str, history_text: str) -> str:
    if not history_text.strip():
        return (
            f'Given the recent observation and the task "{task}", first infer the remaining atomic '
            "actions required to complete the task. Then estimate the current completion percentage "
            "and output it as a float wrapped by <progress> tags."
        )
    return (
        f'Given the recent observation and the task "{task}",\n\n'
        f"Task history:\n{history_text.strip()}\n\n"
        "First infer the remaining atomic actions required to complete the task. "
        "Then estimate the current completion percentage and output it as a float wrapped by "
        "<progress> tags."
    )


class StatefulProcedureTracker:
    """Forward-only 7-of-9 tracker over valid canonicalized ProcVLM observations."""

    def __init__(
        self,
        procedure: Procedure,
        *,
        support_threshold: int = 7,
        window_size: int = 9,
        max_forward_jump: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= support_threshold <= window_size:
            raise ValueError("support_threshold must be within window_size")
        if max_forward_jump != 1:
            raise ValueError("V1 supports max_forward_jump=1 only")
        self.procedure = procedure
        self.support_threshold = int(support_threshold)
        self.window_size = int(window_size)
        self.max_forward_jump = int(max_forward_jump)
        self.persistent_stage = {chain.id: 0 for chain in procedure.chains}
        self._history = {
            chain.id: deque(maxlen=self.window_size) for chain in procedure.chains
        }

    @property
    def confirmed_stage(self) -> dict[str, int]:
        """Compatibility alias for older readers."""
        return self.persistent_stage

    def update(self, frame_index: int, parsed: ParseResult) -> dict[str, Any]:
        if parsed.parse_valid and parsed.observed_stage is not None:
            for chain in self.procedure.chains:
                self._history[chain.id].append(
                    (int(frame_index), int(parsed.observed_stage[chain.id]))
                )

        supports: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        for chain in self.procedure.chains:
            history = self._history[chain.id]
            old = int(self.persistent_stage[chain.id])
            if old >= len(chain.actions):
                supports[chain.id] = {
                    "positive": 0,
                    "total": len(history),
                    "required": self.support_threshold,
                    "target_stage": old,
                }
                continue

            target_stage = old + 1
            positive = sum(1 for _, stage in history if stage >= target_stage)
            support = {
                "positive": positive,
                "total": len(history),
                "required": self.support_threshold,
                "target_stage": target_stage,
            }
            supports[chain.id] = support
            if len(history) < self.support_threshold or positive < self.support_threshold:
                continue

            new = min(target_stage, old + self.max_forward_jump)
            self.persistent_stage[chain.id] = new
            events.append(
                {
                    "chain": chain.id,
                    "from": old,
                    "to": new,
                    "support": f"{positive}/{len(history)}",
                    "frame_index": int(frame_index),
                }
            )
            # Do not reuse the same evidence for the next stage transition.
            history.clear()

        return {
            "persistent_stage": dict(self.persistent_stage),
            "confirmed_stage": dict(self.persistent_stage),
            "transition_support": supports,
            "state_update": events[0] if len(events) == 1 else (events or None),
            "transition_event": events[0] if len(events) == 1 else None,
            "transition_events": events,
            "valid_observation_count": max(
                (len(history) for history in self._history.values()), default=0
            ),
            "tracker_window_counts": {
                chain_id: len(history) for chain_id, history in self._history.items()
            },
        }
