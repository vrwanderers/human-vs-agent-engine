from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", value.lower())
        if len(token) > 1
    }


class MemoryKind(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class DecisionMode(StrEnum):
    HABITUAL = "habitual"
    DELIBERATIVE = "deliberative"


@dataclass
class CognitiveMemory:
    id: str
    turn: int
    kind: MemoryKind
    content: str
    action: str
    outcome_events: tuple[str, ...]
    score_delta: float
    importance: float
    emotional_valence: float
    surprise: float
    tags: tuple[str, ...] = ()
    last_access_turn: int = 0
    access_count: int = 0

    def public_view(self, retrieval_score: float | None = None) -> dict[str, Any]:
        view: dict[str, Any] = {
            "id": self.id,
            "turn": self.turn,
            "kind": self.kind.value,
            "content": self.content,
            "action": self.action,
            "outcome_events": list(self.outcome_events),
            "score_delta": round(self.score_delta, 3),
            "importance": round(self.importance, 3),
            "emotional_valence": round(self.emotional_valence, 3),
            "surprise": round(self.surprise, 3),
            "tags": list(self.tags),
        }
        if retrieval_score is not None:
            view["retrieval_score"] = round(retrieval_score, 3)
        return view


@dataclass(frozen=True)
class Reflection:
    id: str
    turn: int
    belief: str
    confidence: float
    evidence_memory_ids: tuple[str, ...]
    revision_of: str | None = None

    def public_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turn": self.turn,
            "belief": self.belief,
            "confidence": round(self.confidence, 3),
            "evidence_memory_ids": list(self.evidence_memory_ids),
            "revision_of": self.revision_of,
        }


@dataclass
class MemorySystem:
    """Four-part memory with evidence-backed reflection and salience retrieval.

    The implementation is deterministic so it can act as a testable baseline. An LLM may
    phrase a reflection later, but it cannot remove the evidence requirement.
    """

    episodic_limit: int = 64
    reflection_threshold: float = 1.55
    episodic: list[CognitiveMemory] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
    procedural: dict[str, list[float]] = field(default_factory=dict)
    _unreflected_importance: float = 0.0
    _next_memory_id: int = 1
    _next_reflection_id: int = 1

    def record(
        self,
        *,
        turn: int,
        content: str,
        action: str,
        outcome_events: list[str],
        score_delta: float,
        surprise: float,
        emotional_intensity: float,
        tags: tuple[str, ...] = (),
    ) -> CognitiveMemory:
        importance = _clamp(
            0.18
            + 0.28 * min(1.0, abs(score_delta))
            + 0.30 * surprise
            + 0.24 * emotional_intensity
        )
        valence = max(-1.0, min(1.0, score_delta - 0.25 * surprise))
        memory = CognitiveMemory(
            id=f"memory-{self._next_memory_id:04d}",
            turn=turn,
            kind=MemoryKind.EPISODIC,
            content=content,
            action=action,
            outcome_events=tuple(outcome_events),
            score_delta=score_delta,
            importance=importance,
            emotional_valence=valence,
            surprise=surprise,
            tags=tags,
        )
        self._next_memory_id += 1
        self.episodic.append(memory)
        self.episodic = self.episodic[-self.episodic_limit :]
        self.procedural.setdefault(action, []).append(score_delta - 0.15 * surprise)
        self.procedural[action] = self.procedural[action][-20:]
        self._unreflected_importance += importance
        return memory

    def retrieve(
        self,
        query: str,
        *,
        current_turn: int,
        mood_valence: float = 0.0,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, CognitiveMemory]] = []
        for memory in self.episodic:
            age = max(0, current_turn - memory.turn)
            recency = math.exp(-age / 8)
            memory_tokens = _tokens(" ".join((memory.content, memory.action, *memory.tags)))
            relevance = (
                len(query_tokens & memory_tokens) / max(1, len(query_tokens | memory_tokens))
            )
            mood_match = 1 - min(1.0, abs(mood_valence - memory.emotional_valence) / 2)
            score = 0.34 * recency + 0.34 * memory.importance + 0.24 * relevance + 0.08 * mood_match
            ranked.append((score, memory))
        selected = sorted(ranked, key=lambda item: (-item[0], item[1].id))[:limit]
        for _score, memory in selected:
            memory.last_access_turn = current_turn
            memory.access_count += 1
        return [memory.public_view(score) for score, memory in selected]

    def maybe_reflect(self, turn: int) -> Reflection | None:
        if self._unreflected_importance < self.reflection_threshold or len(self.episodic) < 2:
            return None
        evidence = sorted(self.episodic[-6:], key=lambda item: item.importance, reverse=True)[:3]
        action_counts = Counter(item.action for item in evidence)
        action = action_counts.most_common(1)[0][0]
        mean_delta = sum(item.score_delta for item in evidence) / len(evidence)
        direction = (
            "helps"
            if mean_delta > 0.08
            else "hurts"
            if mean_delta < -0.08
            else "is unreliable"
        )
        previous = self.reflections[-1] if self.reflections else None
        reflection = Reflection(
            id=f"reflection-{self._next_reflection_id:04d}",
            turn=turn,
            belief=(
                f"Under recent conditions, {action} {direction}; "
                "treat this as a revisable belief."
            ),
            confidence=_clamp(0.42 + 0.12 * len(evidence) + 0.18 * abs(mean_delta)),
            evidence_memory_ids=tuple(item.id for item in evidence),
            revision_of=previous.id if previous and action in previous.belief else None,
        )
        self._next_reflection_id += 1
        self.reflections.append(reflection)
        self.reflections = self.reflections[-12:]
        self._unreflected_importance = 0.0
        return reflection

    def procedural_values(self) -> dict[str, float]:
        return {
            action: sum(values) / len(values)
            for action, values in self.procedural.items()
            if values
        }

    def public_view(self) -> dict[str, Any]:
        return {
            "working_memory_policy": "current observation plus retrieved items",
            "episodic_count": len(self.episodic),
            "semantic_reflections": [item.public_view() for item in self.reflections[-4:]],
            "procedural_actions": sorted(self.procedural),
        }


@dataclass(frozen=True)
class AppraisalState:
    novelty: float
    goal_congruence: float
    controllability: float
    other_agency: float
    norm_compatibility: float
    social_threat: float
    coping: str
    reappraisal_target: str

    def public_view(self) -> dict[str, Any]:
        return {
            "novelty": round(self.novelty, 3),
            "goal_congruence": round(self.goal_congruence, 3),
            "controllability": round(self.controllability, 3),
            "other_agency": round(self.other_agency, 3),
            "norm_compatibility": round(self.norm_compatibility, 3),
            "social_threat": round(self.social_threat, 3),
            "coping": self.coping,
            "reappraisal_target": self.reappraisal_target,
        }


def appraise(
    *,
    score_delta: float,
    margin: float,
    surprise: float,
    mod_signals: dict[str, float],
    hostile_severity: float,
    uncertainty: float,
) -> AppraisalState:
    negative_mod = max(
        0.0,
        float(mod_signals.get("stress", 0.0)),
        float(mod_signals.get("frustration", 0.0)),
        float(mod_signals.get("anger", 0.0)),
        float(mod_signals.get("fear", 0.0)),
    )
    incongruence = _clamp(0.45 - 0.35 * score_delta - 0.12 * margin + 0.25 * negative_mod)
    controllability = _clamp(0.72 - 0.42 * uncertainty - 0.20 * surprise)
    social_threat = _clamp(0.72 * hostile_severity + 0.28 * negative_mod)
    if social_threat > 0.62 and controllability > 0.45:
        coping = "assert_boundary"
    elif social_threat > 0.55:
        coping = "protect_self"
    elif incongruence > 0.62 and controllability > 0.5:
        coping = "problem_solve"
    elif incongruence > 0.62:
        coping = "seek_information"
    else:
        coping = "maintain_course"
    return AppraisalState(
        novelty=_clamp(0.65 * surprise + 0.35 * uncertainty),
        goal_congruence=1 - incongruence,
        controllability=controllability,
        other_agency=_clamp(hostile_severity),
        norm_compatibility=1 - _clamp(0.8 * hostile_severity),
        social_threat=social_threat,
        coping=coping,
        reappraisal_target=(
            "separate criticism from identity"
            if social_threat > 0.5
            else "update plan from evidence"
        ),
    )


@dataclass
class SocialBelief:
    trust: float = 0.5
    respect: float = 0.5
    familiarity: float = 0.0
    perceived_hostility: float = 0.0
    perceived_sincerity: float = 0.5
    predicted_intent: str = "unknown"
    confidence: float = 0.0

    def update(self, *, hostile_severity: float, cooperative_signal: float, observed: bool) -> None:
        if not observed:
            return
        self.familiarity = _clamp(self.familiarity + 0.12)
        self.perceived_hostility = _clamp(
            0.62 * self.perceived_hostility + 0.38 * hostile_severity
        )
        self.trust = _clamp(
            0.78 * self.trust + 0.22 * (0.55 * cooperative_signal + 0.45 * (1 - hostile_severity))
        )
        self.respect = _clamp(0.85 * self.respect + 0.15 * (1 - 0.55 * hostile_severity))
        self.perceived_sincerity = _clamp(
            0.76 * self.perceived_sincerity + 0.24 * (1 - hostile_severity * 0.65)
        )
        self.predicted_intent = (
            "escalate" if self.perceived_hostility > 0.62 else "test_boundaries"
            if self.perceived_hostility > 0.35 else "cooperate_or_probe"
        )
        self.confidence = _clamp(self.familiarity * 0.9)

    def public_view(self) -> dict[str, Any]:
        return {
            "trust": round(self.trust, 3),
            "respect": round(self.respect, 3),
            "familiarity": round(self.familiarity, 3),
            "perceived_hostility": round(self.perceived_hostility, 3),
            "perceived_sincerity": round(self.perceived_sincerity, 3),
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PlanState:
    goal: str = "assess_situation"
    subgoal: str = "gather_evidence"
    commitment: float = 0.45
    age: int = 0
    revision: int = 0
    last_replan_reason: str = "initial"

    def update(
        self,
        *,
        desired_goal: str,
        surprise: float,
        stress: float,
        goal_congruence: float,
    ) -> bool:
        reason = ""
        if self.goal == "assess_situation" and desired_goal != self.goal:
            self.goal = desired_goal
            self.subgoal = "gather_evidence"
            self.age = 0
            self.last_replan_reason = "initial_commitment"
            return True
        if desired_goal != self.goal and self.age >= 1 and goal_congruence < 0.48:
            reason = "goal_incongruence"
        elif surprise > 0.72:
            reason = "prediction_failure"
        elif stress > 0.82:
            reason = "coping_overload"
        if reason:
            self.goal = desired_goal
            self.subgoal = "reduce_uncertainty" if surprise > 0.55 else "restore_control"
            self.commitment = _clamp(0.42 + 0.35 * goal_congruence)
            self.age = 0
            self.revision += 1
            self.last_replan_reason = reason
            return True
        self.age += 1
        self.commitment = _clamp(self.commitment + 0.04 * (goal_congruence - 0.45))
        self.last_replan_reason = "persisted"
        return False

    def public_view(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subgoal": self.subgoal,
            "commitment": round(self.commitment, 3),
            "age": self.age,
            "revision": self.revision,
            "last_replan_reason": self.last_replan_reason,
        }


def select_decision_mode(*, stress: float, uncertainty: float, stakes: float) -> DecisionMode:
    deliberation_need = 0.40 * uncertainty + 0.35 * stakes + 0.25 * (1 - stress)
    if stress > 0.82 or deliberation_need < 0.43:
        return DecisionMode.HABITUAL
    return DecisionMode.DELIBERATIVE
