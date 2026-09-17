"""Pure procedure-state logic for the training-free ProcVLM tracker."""

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
    text: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chain:
    id: str
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Procedure:
    task_id: str
    task: str
    chains: tuple[Chain, ...]

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.id for chain in self.chains for action in chain.actions)


@dataclass(frozen=True)
class ParseResult:
    remaining_ids: tuple[str, ...]
    parse_valid: bool
    observed_stage: dict[str, int] | None
    source: str
    errors: tuple[str, ...]


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def load_procedure(path: str | Path) -> Procedure:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("procedure config must contain a JSON object")
    task_id = str(raw.get("task_id", "")).strip()
    task = str(raw.get("task", "")).strip()
    raw_chains = raw.get("chains")
    if not task_id or not task:
        raise ValueError("procedure config requires non-empty task_id and task")
    if not isinstance(raw_chains, dict) or not raw_chains:
        raise ValueError("procedure config requires a non-empty chains object")

    chains: list[Chain] = []
    seen: set[str] = set()
    for chain_id, raw_actions in raw_chains.items():
        chain_name = str(chain_id).strip()
        if not chain_name or not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(f"chain {chain_id!r} must contain at least one action")
        actions: list[Action] = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                raise ValueError(f"chain {chain_name} contains a non-object action")
            action_id = str(raw_action.get("id", "")).strip()
            text = str(raw_action.get("text", "")).strip()
            if not action_id or not text:
                raise ValueError(f"chain {chain_name} contains an action without id/text")
            if action_id in seen:
                raise ValueError(f"duplicate canonical action id: {action_id}")
            seen.add(action_id)
            aliases = raw_action.get("aliases", []) or []
            if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
                raise ValueError(f"action {action_id} aliases must be a list of strings")
            actions.append(Action(action_id, text, tuple(a.strip() for a in aliases if a.strip())))
        chains.append(Chain(chain_name, tuple(actions)))
    return Procedure(task_id, task, tuple(chains))


def render_procedure(procedure: Procedure) -> str:
    lines: list[str] = []
    for chain in procedure.chains:
        lines.append(f"Chain {chain.id}:")
        lines.extend(f"[{action.id}] {action.text}" for action in chain.actions)
    lines += ["", "Dependencies:"]
    for chain in procedure.chains:
        lines.append(" -> ".join(action.id for action in chain.actions))
    if len(procedure.chains) > 1:
        lines.append("There is no required ordering between different chains unless stated above.")
    return "\n".join(lines)


def completed_action_ids(procedure: Procedure, confirmed_stage: dict[str, int]) -> list[str]:
    completed: list[str] = []
    for chain in procedure.chains:
        completed.extend(action.id for action in chain.actions[: int(confirmed_stage.get(chain.id, 0))])
    return completed


def build_procedure_prompt(
    procedure: Procedure,
    *,
    mode: str,
    confirmed_stage: dict[str, int] | None = None,
) -> str:
    if mode not in {"canonical", "stateful"}:
        raise ValueError(f"procedure prompt is not defined for mode={mode!r}")
    sections = [
        f"Original task:\n{procedure.task}",
        f"Canonical procedure:\n\n{render_procedure(procedure)}",
    ]
    if mode == "stateful":
        if confirmed_stage is None:
            raise ValueError("stateful prompt requires confirmed_stage")
        completed = completed_action_ids(procedure, confirmed_stage)
        sections.append("Confirmed completed actions:\n" + ("\n".join(completed) if completed else "none"))
        state_lines: list[str] = []
        for chain in procedure.chains:
            stage = int(confirmed_stage.get(chain.id, 0))
            if stage >= len(chain.actions):
                description = "confirmed completed"
            elif stage == 0:
                description = "not completed"
            else:
                description = f"confirmed through {chain.actions[stage - 1].id}"
            state_lines.append(f"- {chain.id}: {description}")
        sections.append("Procedure state:\n" + "\n".join(state_lines))
    sections.append(
        "Given the recent visual observation, infer the remaining actions required to complete the ORIGINAL task.\n"
        "Use only the canonical action IDs above. Do not rename, split, merge, or redefine actions.\n"
        "Do not treat occlusion or temporary invisibility as evidence that a confirmed completed action became incomplete.\n"
        "Return remaining actions under the exact heading 'The following actions are required:' and prefix every action with its canonical ID in square brackets.\n"
        "If no canonical actions remain, write 'The following actions are required: none.'\n"
        "Estimate progress with respect to the ORIGINAL task and end with <progress>NUMBER%</progress>."
    )
    return "\n\n".join(sections)


def extract_remaining_section(answer: str) -> str:
    text = str(answer or "")
    marker = re.search(r"the following actions are required\s*:\s*", text, flags=re.IGNORECASE)
    if marker:
        text = text[marker.end() :]
    return re.split(r"<progress>|\btherefore\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _suffix_stage(chain: Chain, remaining: set[str]) -> int | None:
    ids = [action.id for action in chain.actions]
    for stage in range(len(ids) + 1):
        if remaining == set(ids[stage:]):
            return stage
    return None


def parse_remaining_actions(
    answer: str,
    procedure: Procedure,
    *,
    allow_text_fallback: bool = True,
) -> ParseResult:
    section = extract_remaining_section(answer)
    known = set(procedure.action_ids)
    bracket_tokens = re.findall(r"\[\s*([A-Za-z][A-Za-z0-9_-]*)\s*\]", section)
    unknown = sorted({token for token in bracket_tokens if token not in known})
    ids = {token for token in bracket_tokens if token in known}
    source = "canonical_ids" if ids else "none"
    errors: list[str] = []
    explicit_none = bool(re.search(r"\b(?:none|no actions? remain(?:ing)?|nothing remains?)\b", section, re.I))
    if unknown:
        errors.append("unknown canonical IDs: " + ", ".join(unknown))
    if explicit_none and ids:
        errors.append("output contains both canonical IDs and an explicit none marker")

    if not ids and not explicit_none and allow_text_fallback:
        normalized = normalize_text(section)
        matched: set[str] = set()
        for chain in procedure.chains:
            for action in chain.actions:
                if any(
                    normalize_text(phrase) and normalize_text(phrase) in normalized
                    for phrase in (action.text,) + action.aliases
                ):
                    matched.add(action.id)
        if matched:
            ids, source = matched, "text_fallback"

    if not ids and explicit_none:
        source = "explicit_none"
    elif not ids and not explicit_none:
        errors.append("no canonical action IDs (or explicit none) found in remaining-actions section")

    observed: dict[str, int] = {}
    if not errors:
        for chain in procedure.chains:
            chain_ids = {action.id for action in chain.actions}
            stage = _suffix_stage(chain, ids & chain_ids)
            if stage is None:
                errors.append(f"chain {chain.id} violates suffix property: remaining={sorted(ids & chain_ids)}")
            else:
                observed[chain.id] = stage

    ordered = tuple(action_id for action_id in procedure.action_ids if action_id in ids)
    return ParseResult(ordered, not errors, observed if not errors else None, source, tuple(errors))


class StatefulProcedureTracker:
    """Monotonic per-chain tracker using temporally separated reasoning evidence."""

    def __init__(
        self,
        procedure: Procedure,
        *,
        fps: float,
        decision_interval_frames: int = 3,
        forward_votes: int = 3,
        forward_window: int = 4,
        forward_min_span_sec: float = 0.2,
        completion_votes: int = 4,
        completion_window: int = 5,
        completion_min_span_sec: float = 0.3,
        candidate_timeout_sec: float = 0.5,
        max_forward_jump: int = 1,
    ) -> None:
        if fps <= 0 or decision_interval_frames < 1:
            raise ValueError("fps and decision_interval_frames must be positive")
        if not 1 <= forward_votes <= forward_window:
            raise ValueError("forward_votes must be within forward_window")
        if not 1 <= completion_votes <= completion_window:
            raise ValueError("completion_votes must be within completion_window")
        if candidate_timeout_sec <= 0:
            raise ValueError("candidate_timeout_sec must be positive")
        if max_forward_jump != 1:
            raise ValueError("V1 supports max_forward_jump=1 only")
        self.procedure = procedure
        self.fps = float(fps)
        self.decision_interval_frames = int(decision_interval_frames)
        self.forward_votes, self.forward_window = int(forward_votes), int(forward_window)
        self.forward_min_span_sec = float(forward_min_span_sec)
        self.completion_votes, self.completion_window = int(completion_votes), int(completion_window)
        self.completion_min_span_sec = float(completion_min_span_sec)
        self.candidate_timeout_sec = float(candidate_timeout_sec)
        self.max_forward_jump = max_forward_jump
        self.confirmed_stage = {chain.id: 0 for chain in procedure.chains}
        self._history = {chain.id: deque() for chain in procedure.chains}
        self._last_decision_frame: int | None = None

    def _decision_due(self, frame_index: int) -> bool:
        return self._last_decision_frame is None or frame_index - self._last_decision_frame >= self.decision_interval_frames

    def _trim(self, chain_id: str, frame_index: int) -> None:
        history = self._history[chain_id]
        oldest = frame_index - int(round(self.candidate_timeout_sec * self.fps))
        while history and history[0][0] < oldest:
            history.popleft()
        max_window = max(self.forward_window, self.completion_window)
        while len(history) > max_window:
            history.popleft()

    def _support(self, chain: Chain, frame_index: int) -> tuple[int, dict[str, Any]]:
        self._trim(chain.id, frame_index)
        confirmed = self.confirmed_stage[chain.id]
        if confirmed >= len(chain.actions):
            return confirmed, {"votes": 0, "required": 0, "window": 0, "span_sec": 0.0, "min_span_sec": 0.0, "next_stage": confirmed}
        next_stage = confirmed + 1
        completing = next_stage == len(chain.actions)
        required = self.completion_votes if completing else self.forward_votes
        window = self.completion_window if completing else self.forward_window
        min_span = self.completion_min_span_sec if completing else self.forward_min_span_sec
        recent = list(self._history[chain.id])[-window:]
        supports = [(frame, stage) for frame, stage in recent if stage is not None and stage >= next_stage]
        span = (supports[-1][0] - supports[0][0]) / self.fps if len(supports) >= 2 else 0.0
        return (next_stage if supports else confirmed), {
            "votes": len(supports), "required": required, "window": window,
            "span_sec": round(span, 6), "min_span_sec": min_span, "next_stage": next_stage,
        }

    def update(self, frame_index: int, parsed: ParseResult) -> dict[str, Any]:
        decision = self._decision_due(frame_index)
        events: list[dict[str, Any]] = []
        if decision:
            self._last_decision_frame = frame_index
            for chain in self.procedure.chains:
                observed = parsed.observed_stage.get(chain.id) if parsed.parse_valid and parsed.observed_stage else None
                self._history[chain.id].append((frame_index, observed))

        candidates: dict[str, int] = {}
        supports: dict[str, dict[str, Any]] = {}
        for chain in self.procedure.chains:
            candidate, support = self._support(chain, frame_index)
            candidates[chain.id], supports[chain.id] = candidate, support
            old = self.confirmed_stage[chain.id]
            if not decision or candidate <= old:
                continue
            if support["votes"] < support["required"] or support["span_sec"] < support["min_span_sec"]:
                continue
            new = min(candidate, old + self.max_forward_jump)
            completed_action = chain.actions[new - 1].id
            reason = (
                f"{self.completion_votes}-of-{self.completion_window} chain-completion confirmation"
                if new == len(chain.actions)
                else f"{self.forward_votes}-of-{self.forward_window} forward confirmation"
            )
            self.confirmed_stage[chain.id] = new
            candidates[chain.id] = new
            events.append({
                "chain": chain.id, "from_stage": old, "to_stage": new,
                "completed_action": completed_action, "reason": reason, "frame_index": frame_index,
            })
            self._history[chain.id].clear()

        return {
            "decision_sample": decision,
            "candidate_stage": dict(candidates),
            "confirmed_stage": dict(self.confirmed_stage),
            "transition_support": supports,
            "transition_events": events,
            "transition_event": events[0] if len(events) == 1 else None,
        }
