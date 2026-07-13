from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random
from typing import Any

from hva_engine.models import GameEvent


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _bounded_float(value: Any, default: float) -> float:
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return default


def _clean_term(value: Any, limit: int = 80) -> str:
    return " ".join(str(value).split())[:limit]


class StimulusModality(StrEnum):
    LANGUAGE = "language"
    VISION = "vision"
    AUDIO = "audio"
    TOUCH = "touch"
    INTEROCEPTION = "interoception"
    MEMORY = "memory"
    IMAGINATION = "imagination"
    WORLD_EVENT = "world_event"


class RealityStatus(StrEnum):
    """Epistemic status of a stimulus, not a claim about objective truth."""

    OBSERVED = "observed"
    REMEMBERED = "remembered"
    IMAGINED = "imagined"
    INFERRED = "inferred"
    CANONICAL = "canonical"


class StimulusPrivacy(StrEnum):
    PUBLIC = "public"
    AGENT_PRIVATE = "agent_private"


@dataclass(frozen=True)
class StimulusEvent:
    id: str
    source_id: str
    target_id: str
    modality: StimulusModality
    semantic_tags: tuple[str, ...]
    intensity: float
    valence: float
    urgency: float
    novelty: float
    uncertainty: float
    reality_status: RealityStatus
    privacy: StimulusPrivacy
    sequence: int
    evidence_ref: str
    causal_group: str

    def compact_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "modality": self.modality.value,
            "semantic_tags": list(self.semantic_tags),
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "urgency": round(self.urgency, 3),
            "novelty": round(self.novelty, 3),
            "uncertainty": round(self.uncertainty, 3),
            "reality_status": self.reality_status.value,
            "privacy": self.privacy.value,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class StimulusFrame:
    stimuli: tuple[StimulusEvent, ...] = ()
    modalities: tuple[StimulusModality, ...] = ()
    semantic_tags: tuple[str, ...] = ()
    intensity: float = 0.0
    valence: float = 0.0
    urgency: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    compound_gain: float = 0.0
    sequence_start: int = 0
    sequence_end: int = 0

    @property
    def has_input(self) -> bool:
        return bool(self.stimuli)

    def public_view(self) -> dict[str, Any]:
        return {
            "stimulus_count": len(self.stimuli),
            "modalities": [value.value for value in self.modalities],
            "semantic_tags": list(self.semantic_tags),
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "urgency": round(self.urgency, 3),
            "novelty": round(self.novelty, 3),
            "uncertainty": round(self.uncertainty, 3),
            "compound_gain": round(self.compound_gain, 3),
            "sequence_window": [self.sequence_start, self.sequence_end],
            "reality_statuses": sorted(
                {stimulus.reality_status.value for stimulus in self.stimuli}
            ),
            "evidence_refs": [stimulus.evidence_ref for stimulus in self.stimuli],
        }

    def private_view(self) -> dict[str, Any]:
        return {
            **self.public_view(),
            "stimuli": [stimulus.compact_view() for stimulus in self.stimuli],
        }


@dataclass(frozen=True)
class FastAppraisal:
    threat: float = 0.0
    social_threat: float = 0.0
    personal_relevance: float = 0.0
    memory_resonance: float = 0.0
    action_readiness: float = 0.0
    ambiguity: float = 0.0
    deliberation_pressure: float = 0.0
    trigger_tags: tuple[str, ...] = ()

    def public_view(self) -> dict[str, Any]:
        return {
            "threat": round(self.threat, 3),
            "social_threat": round(self.social_threat, 3),
            "personal_relevance": round(self.personal_relevance, 3),
            "memory_resonance": round(self.memory_resonance, 3),
            "action_readiness": round(self.action_readiness, 3),
            "ambiguity": round(self.ambiguity, 3),
            "deliberation_pressure": round(self.deliberation_pressure, 3),
        }

    def private_view(self) -> dict[str, Any]:
        return {**self.public_view(), "trigger_tags": list(self.trigger_tags)}


@dataclass(frozen=True)
class InvoluntaryCue:
    cue: str
    channel: str
    intensity: float
    latency_ms: int

    def public_view(self) -> dict[str, Any]:
        return {
            "cue": self.cue,
            "channel": self.channel,
            "intensity": round(self.intensity, 3),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class ReflexResponse:
    cues: tuple[InvoluntaryCue, ...] = ()
    masking_attempt: float = 0.0
    reflex_intensity: float = 0.0
    trigger_family: str = "none"
    habituation: float = 0.0
    suppressed_by_cooldown: tuple[str, ...] = ()

    def observable_view(self) -> dict[str, Any]:
        return {
            "cues": [cue.public_view() for cue in self.cues],
            "masking_attempt": round(self.masking_attempt, 3),
            "reflex_intensity": round(self.reflex_intensity, 3),
            "interpretation_policy": "ambiguous_nonverbal_signal_not_truth_or_lie_oracle",
        }

    def private_view(self) -> dict[str, Any]:
        return {
            **self.observable_view(),
            "trigger_family": self.trigger_family,
            "habituation": round(self.habituation, 3),
            "suppressed_by_cooldown": list(self.suppressed_by_cooldown),
        }


@dataclass(frozen=True)
class DeliberationDecision:
    should_deliberate: bool
    score: float
    routine_confidence: float
    reasons: tuple[str, ...]
    provider_available: bool
    automatic_action_indices: tuple[int, ...] = ()

    def public_view(self) -> dict[str, Any]:
        return {
            "should_deliberate": self.should_deliberate,
            "score": round(self.score, 3),
            "routine_confidence": round(self.routine_confidence, 3),
            "reasons": list(self.reasons),
            "provider_available": self.provider_available,
            "automatic_action_indices": list(self.automatic_action_indices),
        }


class RealityStatusGuard:
    """Keeps imagination, memory and inference out of canonical character facts."""

    FACT_WRITABLE = frozenset({RealityStatus.CANONICAL})

    @classmethod
    def can_write_canonical_fact(cls, status: RealityStatus | str) -> bool:
        return RealityStatus(status) in cls.FACT_WRITABLE

    @staticmethod
    def context_contract() -> dict[str, Any]:
        return {
            "canonical_fact_write_statuses": [RealityStatus.CANONICAL.value],
            "noncanonical_statuses": [
                RealityStatus.OBSERVED.value,
                RealityStatus.REMEMBERED.value,
                RealityStatus.IMAGINED.value,
                RealityStatus.INFERRED.value,
            ],
            "rule": (
                "Noncanonical stimuli may affect emotion and decisions but cannot become "
                "identity/history facts without authoritative evidence."
            ),
        }


class PerceptionAdapter:
    """Normalizes visible engine events into modality-independent stimuli."""

    _IGNORED_TYPES = frozenset(
        {
            "match_created",
            "match_finished",
            "agent_decision",
            "agent_influence_intent",
            "agent_reflex_diagnostic",
            "story_reveal_diagnostic",
        }
    )

    def from_game_events(
        self,
        events: Iterable[GameEvent],
        *,
        viewer_id: str,
        after_sequence: int = 0,
    ) -> list[StimulusEvent]:
        stimuli: list[StimulusEvent] = []
        for event in events:
            if event.seq <= after_sequence or event.type in self._IGNORED_TYPES:
                continue
            explicit = event.payload.get("stimulus", {})
            if not isinstance(explicit, dict):
                explicit = {}
            modality = self._modality(event, explicit)
            internal_for_viewer = modality in {
                StimulusModality.INTEROCEPTION,
                StimulusModality.MEMORY,
                StimulusModality.IMAGINATION,
            }
            target_id = _clean_term(
                explicit.get("target_id", event.payload.get("target_id", viewer_id)), 128
            )
            if target_id and target_id != viewer_id:
                continue
            if event.actor_id == viewer_id and not internal_for_viewer:
                # An agent does not re-perceive its own audit/action events as new input.
                continue
            status = self._reality_status(modality, event, explicit)
            tags = self._semantic_tags(event, explicit)
            intensity = _bounded_float(
                explicit.get(
                    "intensity",
                    event.payload.get("severity", event.payload.get("intensity", 0.45)),
                ),
                0.45,
            )
            valence = self._valence(event, explicit, intensity)
            urgency = _bounded_float(
                explicit.get("urgency", event.payload.get("urgency")),
                self._default_urgency(modality, event.type, intensity),
            )
            novelty = _bounded_float(
                explicit.get("novelty", event.payload.get("novelty")),
                0.62 if modality in {StimulusModality.IMAGINATION, StimulusModality.WORLD_EVENT}
                else 0.48,
            )
            uncertainty = _bounded_float(
                explicit.get("uncertainty", event.payload.get("uncertainty")),
                {
                    RealityStatus.CANONICAL: 0.05,
                    RealityStatus.OBSERVED: 0.22,
                    RealityStatus.REMEMBERED: 0.42,
                    RealityStatus.INFERRED: 0.68,
                    RealityStatus.IMAGINED: 0.82,
                }[status],
            )
            privacy = (
                StimulusPrivacy.AGENT_PRIVATE
                if internal_for_viewer or str(explicit.get("privacy", "")) == "agent_private"
                else StimulusPrivacy.PUBLIC
            )
            source_id = _clean_term(
                explicit.get(
                    "source_id", event.payload.get("source_id", event.actor_id or "world")
                ),
                128,
            )
            stimuli.append(
                StimulusEvent(
                    id=f"stimulus-{event.seq:06d}",
                    source_id=source_id or "world",
                    target_id=viewer_id,
                    modality=modality,
                    semantic_tags=tags,
                    intensity=intensity,
                    valence=valence,
                    urgency=urgency,
                    novelty=novelty,
                    uncertainty=uncertainty,
                    reality_status=status,
                    privacy=privacy,
                    sequence=event.seq,
                    evidence_ref=f"event:{event.seq}:{event.type}",
                    causal_group=_clean_term(
                        explicit.get("causal_group", event.payload.get("causal_group", event.seq)),
                        96,
                    ),
                )
            )
        return stimuli

    @staticmethod
    def _modality(event: GameEvent, explicit: dict[str, Any]) -> StimulusModality:
        raw = explicit.get("modality", event.payload.get("modality"))
        if raw is not None:
            try:
                return StimulusModality(str(raw))
            except ValueError:
                pass
        text = event.type.lower()
        payload = event.payload
        if any(token in text for token in ("imagination", "fantasy", "dream")):
            return StimulusModality.IMAGINATION
        if any(token in text for token in ("memory", "recall", "recollection")):
            return StimulusModality.MEMORY
        if any(token in text for token in ("heartbeat", "fatigue", "hunger", "pain_state")):
            return StimulusModality.INTEROCEPTION
        if any(token in text for token in ("touch", "collision", "impact", "damage", "hit")):
            return StimulusModality.TOUCH
        if any(token in text for token in ("audio", "sound", "voice", "heard", "music")):
            return StimulusModality.AUDIO
        if any(token in text for token in ("visual", "sight", "seen", "gesture", "cue")):
            return StimulusModality.VISION
        nested_action = payload.get("action_payload", {})
        has_language = any(key in payload for key in ("prompt", "utterance", "answer", "text"))
        has_language = has_language or (
            isinstance(nested_action, dict)
            and any(key in nested_action for key in ("prompt", "utterance", "answer", "text"))
        )
        if has_language or any(
            token in text for token in ("question", "response", "dialogue", "speech")
        ):
            return StimulusModality.LANGUAGE
        return StimulusModality.WORLD_EVENT

    @staticmethod
    def _reality_status(
        modality: StimulusModality, event: GameEvent, explicit: dict[str, Any]
    ) -> RealityStatus:
        raw = explicit.get("reality_status", event.payload.get("reality_status"))
        if raw is not None:
            try:
                return RealityStatus(str(raw))
            except ValueError:
                pass
        if modality == StimulusModality.IMAGINATION:
            return RealityStatus.IMAGINED
        if modality == StimulusModality.MEMORY:
            return RealityStatus.REMEMBERED
        if modality == StimulusModality.WORLD_EVENT:
            return RealityStatus.CANONICAL
        if event.type == "sensory_stimulus" or modality in {
            StimulusModality.LANGUAGE,
            StimulusModality.VISION,
            StimulusModality.AUDIO,
            StimulusModality.TOUCH,
            StimulusModality.INTEROCEPTION,
        }:
            return RealityStatus.OBSERVED
        return RealityStatus.CANONICAL

    @staticmethod
    def _semantic_tags(event: GameEvent, explicit: dict[str, Any]) -> tuple[str, ...]:
        values: list[Any] = [event.type]
        explicit_tags = explicit.get("semantic_tags", event.payload.get("semantic_tags", []))
        if isinstance(explicit_tags, (list, tuple, set)):
            values.extend(explicit_tags)
        nested = event.payload.get("action_payload", {})
        for source in (event.payload, nested if isinstance(nested, dict) else {}):
            for key in ("theme", "action_type", "strategy", "move", "event_class"):
                if source.get(key) is not None:
                    values.append(source[key])
        tags: list[str] = []
        for value in values:
            term = _clean_term(value).lower()
            if term and term not in tags:
                tags.append(term)
        return tuple(tags[:12])

    @staticmethod
    def _valence(event: GameEvent, explicit: dict[str, Any], intensity: float) -> float:
        raw = explicit.get("valence", event.payload.get("valence"))
        if raw is not None:
            try:
                return _clamp(float(raw), -1.0, 1.0)
            except (TypeError, ValueError):
                pass
        text = event.type.lower()
        if any(token in text for token in ("failure", "damage", "threat", "hostile")):
            return -intensity
        severity = _bounded_float(event.payload.get("severity"), 0.0)
        if severity > 0:
            return -severity
        if any(token in text for token in ("success", "repair", "support", "trust")):
            return intensity * 0.7
        return 0.0

    @staticmethod
    def _default_urgency(
        modality: StimulusModality, event_type: str, intensity: float
    ) -> float:
        if modality == StimulusModality.TOUCH:
            return _clamp(0.35 + 0.65 * intensity)
        if any(token in event_type.lower() for token in ("danger", "explosion", "attack")):
            return _clamp(0.45 + 0.55 * intensity)
        if modality == StimulusModality.LANGUAGE:
            return _clamp(0.18 + 0.42 * intensity)
        return _clamp(0.12 + 0.48 * intensity)


class TemporalBinder:
    """Fuses nearby inputs while limiting duplicate and multimodal amplification."""

    def __init__(self, sequence_window: int = 3) -> None:
        self.sequence_window = max(0, sequence_window)

    def bind(self, stimuli: Iterable[StimulusEvent]) -> StimulusFrame:
        ordered = sorted(stimuli, key=lambda item: (item.sequence, item.id))
        if not ordered:
            return StimulusFrame()
        newest = ordered[-1].sequence
        active = tuple(
            item for item in ordered if newest - item.sequence <= self.sequence_window
        )
        modalities = tuple(sorted({item.modality for item in active}, key=lambda item: item.value))
        tags: list[str] = []
        for item in active:
            for tag in item.semantic_tags:
                if tag not in tags:
                    tags.append(tag)
        base_intensity = max(item.intensity for item in active)
        unique_groups = len({item.causal_group for item in active})
        modality_gain = 0.08 * max(0, len(modalities) - 1)
        independent_gain = 0.035 * max(0, min(3, unique_groups - 1))
        compound_gain = _clamp(modality_gain + independent_gain, 0.0, 0.28)
        weights = [max(0.08, item.intensity) for item in active]
        weight_sum = sum(weights)
        valence = sum(
            weight * item.valence for weight, item in zip(weights, active, strict=True)
        ) / weight_sum
        # Max pooling avoids treating duplicated action/domain events as independent trauma.
        urgency = max(item.urgency for item in active)
        novelty = max(item.novelty for item in active)
        uncertainty = sum(
            weight * item.uncertainty for weight, item in zip(weights, active, strict=True)
        ) / weight_sum
        return StimulusFrame(
            stimuli=active,
            modalities=modalities,
            semantic_tags=tuple(tags[:16]),
            intensity=_clamp(base_intensity + compound_gain),
            valence=_clamp(valence, -1.0, 1.0),
            urgency=_clamp(urgency + 0.35 * compound_gain),
            novelty=_clamp(novelty + 0.25 * compound_gain),
            uncertainty=_clamp(uncertainty),
            compound_gain=compound_gain,
            sequence_start=active[0].sequence,
            sequence_end=active[-1].sequence,
        )


class FastAppraisalEngine:
    """Produces pre-deliberative relevance and readiness without selecting an action."""

    def appraise(
        self,
        frame: StimulusFrame,
        *,
        identity_themes: Iterable[str] = (),
        sensitive_topics: Iterable[str] = (),
        stress: float = 0.0,
        fear: float = 0.0,
        uncertainty: float = 0.0,
    ) -> FastAppraisal:
        if not frame.has_input:
            return FastAppraisal(
                ambiguity=_clamp(uncertainty),
                deliberation_pressure=_clamp(0.35 * uncertainty),
            )
        trigger_terms = {_clean_term(value).lower() for value in frame.semantic_tags if value}
        identity_terms = {_clean_term(value).lower() for value in identity_themes if value}
        sensitive_terms = {_clean_term(value).lower() for value in sensitive_topics if value}
        identity_hits = trigger_terms & identity_terms
        sensitive_hits = trigger_terms & sensitive_terms
        memory_resonance = _clamp(
            0.18 * len(identity_hits) + 0.28 * len(sensitive_hits)
        )
        personal_relevance = _clamp(
            0.34 * frame.intensity
            + 0.42 * memory_resonance
            + 0.12 * len(frame.modalities)
            + 0.12 * abs(frame.valence)
        )
        negative_arousal = frame.intensity * max(0.0, -frame.valence)
        threat = _clamp(
            0.38 * frame.urgency
            + 0.30 * negative_arousal
            + 0.18 * stress
            + 0.14 * fear
        )
        social = StimulusModality.LANGUAGE in frame.modalities
        social_threat = _clamp(
            (0.48 * frame.intensity + 0.30 * max(0.0, -frame.valence) + 0.22 * stress)
            if social
            else 0.12 * threat
        )
        ambiguity = _clamp(0.62 * frame.uncertainty + 0.38 * uncertainty)
        action_readiness = _clamp(
            0.38 * threat
            + 0.26 * frame.urgency
            + 0.18 * personal_relevance
            + 0.18 * max(stress, fear)
        )
        deliberation_pressure = _clamp(
            0.24 * frame.novelty
            + 0.22 * ambiguity
            + 0.20 * personal_relevance
            + 0.16 * social_threat
            + 0.12 * frame.urgency
            + 0.06 * min(1.0, len(frame.modalities) / 3)
        )
        return FastAppraisal(
            threat=threat,
            social_threat=social_threat,
            personal_relevance=personal_relevance,
            memory_resonance=memory_resonance,
            action_readiness=action_readiness,
            ambiguity=ambiguity,
            deliberation_pressure=deliberation_pressure,
            trigger_tags=tuple(sorted(identity_hits | sensitive_hits)),
        )


@dataclass
class ReflexController:
    cooldown_turns: int = 2
    _last_emitted: dict[str, int] = field(default_factory=dict)
    _trigger_counts: dict[str, int] = field(default_factory=dict)

    def respond(
        self,
        frame: StimulusFrame,
        appraisal: FastAppraisal,
        *,
        turn: int,
        conscientiousness: float,
        agreeableness: float,
        neuroticism: float,
        stress: float,
        rng: Random,
    ) -> ReflexResponse:
        if not frame.has_input or appraisal.action_readiness < 0.22:
            return ReflexResponse()
        trigger_family = self._trigger_family(frame, appraisal)
        trigger_count = self._trigger_counts.get(trigger_family, 0)
        habituation = _clamp(1 - math.exp(-trigger_count / 4.0), 0.0, 0.58)
        raw_intensity = _clamp(
            0.42 * appraisal.action_readiness
            + 0.22 * appraisal.personal_relevance
            + 0.18 * stress
            + 0.18 * neuroticism
        )
        reflex_intensity = _clamp(raw_intensity * (1 - 0.48 * habituation))
        masking_attempt = _clamp(
            0.42 * conscientiousness + 0.24 * agreeableness + 0.18 * (1 - stress)
        )
        leak_intensity = _clamp(reflex_intensity * (1 - 0.52 * masking_attempt))
        candidates = self._candidates(frame, appraisal)
        suppressed: list[str] = []
        available: list[tuple[str, str, float]] = []
        for cue, channel, cue_bias in candidates:
            if turn - self._last_emitted.get(cue, -10_000) <= self.cooldown_turns:
                suppressed.append(cue)
                continue
            jitter = rng.uniform(-0.06, 0.06)
            strength = _clamp(leak_intensity + cue_bias + jitter)
            if strength >= 0.2:
                available.append((cue, channel, strength))
        available.sort(key=lambda item: (-item[2], item[0]))
        cues: list[InvoluntaryCue] = []
        for index, (cue, channel, strength) in enumerate(available[:2]):
            latency = int(
                max(70, min(900, 620 - 390 * frame.urgency + index * 130 + rng.randint(-35, 35)))
            )
            cues.append(InvoluntaryCue(cue, channel, strength, latency))
            self._last_emitted[cue] = turn
        self._trigger_counts[trigger_family] = trigger_count + 1
        return ReflexResponse(
            cues=tuple(cues),
            masking_attempt=masking_attempt,
            reflex_intensity=reflex_intensity,
            trigger_family=trigger_family,
            habituation=habituation,
            suppressed_by_cooldown=tuple(sorted(suppressed)),
        )

    @staticmethod
    def _trigger_family(frame: StimulusFrame, appraisal: FastAppraisal) -> str:
        if StimulusModality.TOUCH in frame.modalities or appraisal.threat > 0.72:
            return "protective"
        if appraisal.memory_resonance > 0.25:
            return "conditioned_memory"
        if StimulusModality.LANGUAGE in frame.modalities:
            return "social_exposure"
        if StimulusModality.IMAGINATION in frame.modalities:
            return "imagined_scenario"
        return "orienting"

    @staticmethod
    def _candidates(
        frame: StimulusFrame, appraisal: FastAppraisal
    ) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        if StimulusModality.TOUCH in frame.modalities or appraisal.threat > 0.72:
            candidates.extend(
                [("protective_shift", "body", 0.16), ("startle", "posture", 0.12)]
            )
        if StimulusModality.LANGUAGE in frame.modalities:
            candidates.extend(
                [("speech_pause", "voice", 0.08), ("gaze_break", "gaze", 0.02)]
            )
        if StimulusModality.AUDIO in frame.modalities:
            candidates.append(("orient_toward_sound", "head", 0.06))
        if StimulusModality.VISION in frame.modalities:
            candidates.append(("visual_fixation", "gaze", 0.04))
        if appraisal.memory_resonance > 0.18:
            candidates.extend(
                [("breath_catch", "breath", 0.10), ("hand_hesitation", "gesture", 0.05)]
            )
        if StimulusModality.IMAGINATION in frame.modalities:
            candidates.append(("momentary_withdrawal", "posture", 0.04))
        if not candidates:
            candidates.append(("posture_tightening", "posture", 0.0))
        return candidates


class DeliberationGate:
    """Escalates novel/conflicted situations while letting learned routines stay local."""

    def evaluate(
        self,
        *,
        provider_available: bool,
        frame: StimulusFrame,
        appraisal: FastAppraisal,
        decision_mode: str,
        legal_action_types: Iterable[str],
        procedural_values: dict[str, float] | None,
        previous_action: str | None,
        text_interaction: bool,
        plan_revised: bool,
        skill_candidates: list[dict[str, Any]] | None = None,
    ) -> DeliberationDecision:
        legal = tuple(legal_action_types)
        procedural_values = procedural_values or {}
        candidates = skill_candidates or []
        automatic_indices = tuple(
            int(candidate["action_index"])
            for candidate in candidates
            if candidate.get("automatic") is True
            and isinstance(candidate.get("action_index"), int)
        )
        preferred_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("preferred_by_local_policy") is True
        ]
        control_candidates = preferred_candidates or candidates
        if candidates:
            known = [
                str(candidate.get("skill_id", ""))
                for candidate in control_candidates
                if int(candidate.get("attempts", 0)) > 0
            ]
            best_confidence = max(
                (
                    float(candidate.get("confidence", 0.0))
                    for candidate in control_candidates
                ),
                default=0.0,
            )
            preferred_automatic = any(
                candidate.get("automatic") is True
                for candidate in control_candidates
            )
            routine_confidence = _clamp(
                best_confidence if preferred_automatic else 0.58 * best_confidence
            )
        else:
            known = [action for action in legal if action in procedural_values]
            routine_confidence = _clamp(
                (len(known) / max(1, len(legal))) * 0.62
                + (0.22 if previous_action in legal else 0.0)
                + (0.16 if len(legal) == 1 else 0.0)
            )
            preferred_automatic = bool(known)
        reasons: list[str] = []
        if not provider_available:
            return DeliberationDecision(
                False,
                0.0,
                routine_confidence,
                ("provider_unavailable",),
                False,
                automatic_indices,
            )
        score = appraisal.deliberation_pressure
        if text_interaction and StimulusModality.LANGUAGE in frame.modalities:
            score += 0.18
            reasons.append("open_ended_social_language")
        if plan_revised:
            score += 0.12
            reasons.append("plan_revised")
        if appraisal.ambiguity >= 0.58:
            reasons.append("high_ambiguity")
        if frame.novelty >= 0.68:
            reasons.append("novel_stimulus")
        if appraisal.personal_relevance >= 0.62:
            reasons.append("personally_salient")
        if len(frame.modalities) >= 2:
            reasons.append("multimodal_compound")
        if routine_confidence >= 0.55:
            score -= 0.20 * routine_confidence
            reasons.append("known_routine")
        if decision_mode == "habitual":
            score -= 0.08
            reasons.append("habitual_mode")
        # An initial authoritative world snapshot is itself sufficient reason to form a plan,
        # even when no discrete sensory event preceded the first turn.
        unfamiliar_situation = not known and previous_action is None
        guided_practice_required = bool(candidates) and not preferred_automatic
        if not known and legal:
            score += 0.10
            reasons.append("no_procedural_skill")
        if unfamiliar_situation:
            score += 0.08
            reasons.append("unfamiliar_situation")
        if guided_practice_required:
            score += 0.10
            reasons.append("skill_not_yet_automatic")
        score = _clamp(score)
        force_language = (
            text_interaction
            and StimulusModality.LANGUAGE in frame.modalities
            and appraisal.social_threat >= 0.48
        )
        should_deliberate = (
            force_language
            or unfamiliar_situation
            or guided_practice_required
            or score >= 0.52
        )
        if should_deliberate:
            reasons.append("threshold_exceeded" if not force_language else "social_reply_required")
        else:
            reasons.append("handled_by_reflex_or_routine")
        return DeliberationDecision(
            should_deliberate,
            score,
            routine_confidence,
            tuple(dict.fromkeys(reasons)),
            True,
            automatic_indices,
        )


@dataclass
class StimulusPipeline:
    adapter: PerceptionAdapter = field(default_factory=PerceptionAdapter)
    binder: TemporalBinder = field(default_factory=TemporalBinder)
    appraisal_engine: FastAppraisalEngine = field(default_factory=FastAppraisalEngine)
    reflex_controller: ReflexController = field(default_factory=ReflexController)
    deliberation_gate: DeliberationGate = field(default_factory=DeliberationGate)

    def perceive(
        self,
        events: Iterable[GameEvent],
        *,
        viewer_id: str,
        after_sequence: int,
        identity_themes: Iterable[str],
        sensitive_topics: Iterable[str],
        stress: float,
        fear: float,
        uncertainty: float,
    ) -> tuple[StimulusFrame, FastAppraisal]:
        stimuli = self.adapter.from_game_events(
            events, viewer_id=viewer_id, after_sequence=after_sequence
        )
        frame = self.binder.bind(stimuli)
        appraisal = self.appraisal_engine.appraise(
            frame,
            identity_themes=identity_themes,
            sensitive_topics=sensitive_topics,
            stress=stress,
            fear=fear,
            uncertainty=uncertainty,
        )
        return frame, appraisal
