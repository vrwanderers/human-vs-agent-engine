from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any

from hva_engine.memory_store import LongTermMemoryStore, MemoryDocument


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clean(value: Any, limit: int = 120) -> str:
    return " ".join(str(value).split())[:limit]


class SkillStage(StrEnum):
    NOVEL = "novel"
    GUIDED = "guided"
    PRACTICED = "practiced"
    AUTOMATIC = "automatic"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class SkillContext:
    descriptor: tuple[tuple[str, str], ...]
    signature: str

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SkillContext:
        descriptor = tuple(
            sorted(
                (cleaned_key, cleaned_value)
                for key, value in values.items()
                if (cleaned_key := _clean(key, 60).lower())
                and (cleaned_value := _clean(value, 120).lower())
            )
        )
        encoded = json.dumps(descriptor, ensure_ascii=False, separators=(",", ":"))
        return cls(descriptor, sha256(encoded.encode()).hexdigest()[:20])

    @property
    def terms(self) -> frozenset[str]:
        return frozenset(f"{key}={value}" for key, value in self.descriptor)

    def public_view(self) -> dict[str, Any]:
        return {"signature": self.signature, "descriptor": dict(self.descriptor)}


@dataclass
class ContextPractice:
    descriptor: tuple[tuple[str, str], ...]
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    surprises: int = 0
    last_used_turn: int = 0

    @property
    def terms(self) -> frozenset[str]:
        return frozenset(f"{key}={value}" for key, value in self.descriptor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": dict(self.descriptor),
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "surprises": self.surprises,
            "last_used_turn": self.last_used_turn,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContextPractice:
        descriptor = value.get("descriptor", {})
        return cls(
            descriptor=tuple(
                sorted(
                    (_clean(key, 60).lower(), _clean(item, 120).lower())
                    for key, item in descriptor.items()
                )
            )
            if isinstance(descriptor, dict)
            else (),
            attempts=max(0, int(value.get("attempts", 0))),
            successes=max(0, int(value.get("successes", 0))),
            failures=max(0, int(value.get("failures", 0))),
            surprises=max(0, int(value.get("surprises", 0))),
            last_used_turn=max(0, int(value.get("last_used_turn", 0))),
        )


@dataclass
class ProceduralSkill:
    skill_id: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    guided_attempts: int = 0
    automatic_attempts: int = 0
    surprise_count: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_used_turn: int = 0
    contexts: dict[str, ContextPractice] = field(default_factory=dict)

    @property
    def success_confidence(self) -> float:
        # Beta(1,1) posterior mean prevents one success from looking like mastery.
        return (self.successes + 1) / (self.attempts + 2)

    @property
    def surprise_rate(self) -> float:
        return self.surprise_count / max(1, self.attempts)

    @property
    def stage(self) -> SkillStage:
        if self.attempts == 0:
            return SkillStage.NOVEL
        if self.consecutive_failures >= 2 or (
            self.attempts >= 4 and self.surprise_rate > 0.55
        ):
            return SkillStage.DEGRADED
        if (
            self.attempts >= 5
            and self.success_confidence >= 0.72
            and self.consecutive_successes >= 3
            and self.surprise_rate <= 0.35
        ):
            return SkillStage.AUTOMATIC
        if self.attempts >= 3 and self.success_confidence >= 0.58:
            return SkillStage.PRACTICED
        return SkillStage.GUIDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "guided_attempts": self.guided_attempts,
            "automatic_attempts": self.automatic_attempts,
            "surprise_count": self.surprise_count,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "last_used_turn": self.last_used_turn,
            "contexts": {
                signature: context.to_dict()
                for signature, context in sorted(self.contexts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProceduralSkill:
        raw_contexts = value.get("contexts", {})
        contexts = (
            {
                str(signature): ContextPractice.from_dict(context)
                for signature, context in raw_contexts.items()
                if isinstance(context, dict)
            }
            if isinstance(raw_contexts, dict)
            else {}
        )
        return cls(
            skill_id=_clean(value.get("skill_id", "unknown"), 160),
            attempts=max(0, int(value.get("attempts", 0))),
            successes=max(0, int(value.get("successes", 0))),
            failures=max(0, int(value.get("failures", 0))),
            guided_attempts=max(0, int(value.get("guided_attempts", 0))),
            automatic_attempts=max(0, int(value.get("automatic_attempts", 0))),
            surprise_count=max(0, int(value.get("surprise_count", 0))),
            consecutive_successes=max(0, int(value.get("consecutive_successes", 0))),
            consecutive_failures=max(0, int(value.get("consecutive_failures", 0))),
            last_used_turn=max(0, int(value.get("last_used_turn", 0))),
            contexts=contexts,
        )


@dataclass(frozen=True)
class SkillReadiness:
    skill_id: str
    stage: SkillStage
    confidence: float
    context_familiarity: float
    context_similarity: float
    global_success_confidence: float
    attempts: int
    context_attempts: int
    automatic: bool
    guidance_required: bool
    reason: str
    context: SkillContext

    def public_view(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "stage": self.stage.value,
            "confidence": round(self.confidence, 3),
            "context_familiarity": round(self.context_familiarity, 3),
            "context_similarity": round(self.context_similarity, 3),
            "global_success_confidence": round(self.global_success_confidence, 3),
            "attempts": self.attempts,
            "context_attempts": self.context_attempts,
            "automatic": self.automatic,
            "guidance_required": self.guidance_required,
            "reason": self.reason,
            "context": self.context.public_view(),
        }


class SkillLearningSystem:
    """Persistent procedural learning with context-specific automaticity."""

    CATEGORY = "procedural_skill"

    def __init__(self, owner_id: str, store: LongTermMemoryStore) -> None:
        self.owner_id = owner_id
        self.store = store
        self._skills: dict[str, ProceduralSkill] = {}
        self._load()

    def _load(self) -> None:
        for document in self.store.list_by_category(
            self.owner_id, self.CATEGORY, limit=256
        ):
            raw = document.metadata.get("procedural_skill")
            if not isinstance(raw, dict):
                continue
            skill = ProceduralSkill.from_dict(raw)
            if skill.skill_id:
                self._skills[skill.skill_id] = skill

    @staticmethod
    def _document_id(skill_id: str) -> str:
        return f"skill-{sha256(skill_id.encode()).hexdigest()[:20]}"

    def readiness(
        self,
        skill_id: str,
        context_values: dict[str, Any],
        *,
        current_turn: int,
    ) -> SkillReadiness:
        skill_id = _clean(skill_id, 160)
        context = SkillContext.from_mapping(context_values)
        skill = self._skills.get(skill_id, ProceduralSkill(skill_id))
        exact = skill.contexts.get(context.signature)
        context_attempts = exact.attempts if exact else 0
        context_successes = exact.successes if exact else 0
        context_posterior = (context_successes + 1) / (context_attempts + 2)
        similarity = max(
            (
                len(context.terms & practiced.terms)
                / max(1, len(context.terms | practiced.terms))
                for practiced in skill.contexts.values()
            ),
            default=0.0,
        )
        familiarity = _clamp(context_attempts / 4)
        transfer = skill.success_confidence * similarity
        confidence = _clamp(
            0.46 * context_posterior
            + 0.34 * familiarity
            + 0.20 * transfer
        )
        disuse_gap = max(0, current_turn - skill.last_used_turn) if skill.attempts else 0
        if disuse_gap > 20:
            confidence *= math.exp(-(disuse_gap - 20) / 60)
        automatic = (
            skill.stage == SkillStage.AUTOMATIC
            and context_attempts >= 3
            and confidence >= 0.72
            and similarity >= 0.8
        )
        if skill.stage == SkillStage.DEGRADED:
            reason = "recent_failures_require_retraining"
        elif context_attempts == 0 and skill.attempts:
            reason = "skill_known_but_context_is_new"
        elif automatic:
            reason = "stable_success_in_familiar_context"
        elif skill.stage == SkillStage.NOVEL:
            reason = "no_experience"
        else:
            reason = "practice_not_yet_automatic"
        return SkillReadiness(
            skill_id=skill_id,
            stage=skill.stage,
            confidence=confidence,
            context_familiarity=familiarity,
            context_similarity=similarity,
            global_success_confidence=skill.success_confidence,
            attempts=skill.attempts,
            context_attempts=context_attempts,
            automatic=automatic,
            guidance_required=not automatic,
            reason=reason,
            context=context,
        )

    def record_execution(
        self,
        skill_id: str,
        context_values: dict[str, Any],
        *,
        turn: int,
        success: bool,
        surprise: float,
        guided: bool,
        automatic: bool = False,
    ) -> SkillReadiness:
        skill_id = _clean(skill_id, 160)
        context = SkillContext.from_mapping(context_values)
        skill = self._skills.setdefault(skill_id, ProceduralSkill(skill_id))
        practice = skill.contexts.setdefault(
            context.signature, ContextPractice(context.descriptor)
        )
        skill.attempts += 1
        practice.attempts += 1
        if success:
            skill.successes += 1
            practice.successes += 1
            skill.consecutive_successes += 1
            skill.consecutive_failures = 0
        else:
            skill.failures += 1
            practice.failures += 1
            skill.consecutive_failures += 1
            skill.consecutive_successes = 0
        if surprise >= 0.6:
            skill.surprise_count += 1
            practice.surprises += 1
        if guided:
            skill.guided_attempts += 1
        if automatic:
            skill.automatic_attempts += 1
        skill.last_used_turn = max(skill.last_used_turn, turn)
        practice.last_used_turn = max(practice.last_used_turn, turn)
        self._persist(skill, turn)
        return self.readiness(skill_id, context_values, current_turn=turn)

    def _persist(self, skill: ProceduralSkill, turn: int) -> None:
        stage = skill.stage
        self.store.upsert(
            MemoryDocument(
                owner_id=self.owner_id,
                id=self._document_id(skill.skill_id),
                turn=turn,
                kind="procedural",
                summary=(
                    f"Procedural skill {skill.skill_id}: {stage.value}, "
                    f"{skill.successes}/{skill.attempts} successful executions"
                ),
                content="Validated execution statistics; no hidden reasoning stored.",
                action=skill.skill_id,
                outcome_events=("skill_updated",),
                score_delta=skill.success_confidence - 0.5,
                importance=_clamp(0.45 + 0.08 * skill.attempts),
                emotional_valence=0.0,
                surprise=skill.surprise_rate,
                categories=(self.CATEGORY, "skill"),
                tags=(skill.skill_id, stage.value),
                metadata={"procedural_skill": skill.to_dict()},
            )
        )

    def public_view(self, *, current_turn: int = 0) -> dict[str, Any]:
        stages = {stage.value: 0 for stage in SkillStage}
        automatic: list[str] = []
        for skill in self._skills.values():
            stages[skill.stage.value] += 1
            if skill.stage == SkillStage.AUTOMATIC:
                automatic.append(skill.skill_id)
        return {
            "owner_scoped": True,
            "persistent_store": self.store.backend_name,
            "skill_count": len(self._skills),
            "stages": stages,
            "automatic_skills": sorted(automatic),
            "current_turn": current_turn,
            "automaticity_policy": (
                "five attempts, calibrated success, three consecutive successes, "
                "and familiar context"
            ),
            "new_context_requires_guidance": True,
            "failure_can_degrade_skill": True,
        }
