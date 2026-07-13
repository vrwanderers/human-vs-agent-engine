from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hva_engine.models import Action, AgentTuning, ContentMode


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RuntimeBehaviorPolicy:
    """Engine-owned boundary; model capability never changes authority or safety."""

    content_mode: ContentMode
    requested_shadow_intensity: float
    effective_shadow_intensity: float
    realism: float
    rules_authority: str = "engine_only"
    private_reasoning_exposure: str = "forbidden"
    real_world_harm: str = "forbidden"

    @classmethod
    def from_tuning(cls, tuning: AgentTuning) -> RuntimeBehaviorPolicy:
        shadow_cap = 0.35 if tuning.content_mode == ContentMode.STANDARD else 1.0
        return cls(
            content_mode=tuning.content_mode,
            requested_shadow_intensity=tuning.shadow_intensity,
            effective_shadow_intensity=min(tuning.shadow_intensity, shadow_cap),
            realism=tuning.realism,
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "content_mode": self.content_mode.value,
            "requested_shadow_intensity": round(self.requested_shadow_intensity, 3),
            "effective_shadow_intensity": round(self.effective_shadow_intensity, 3),
            "rules_authority": self.rules_authority,
            "private_reasoning_exposure": self.private_reasoning_exposure,
            "real_world_harm": self.real_world_harm,
        }


@dataclass(frozen=True)
class CognitiveProfile:
    archetype: str
    risk_tolerance: float
    loss_aversion: float
    patience: float
    curiosity: float
    empathy: float
    adaptability: float
    machiavellianism: float
    decision_noise: float
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    coping_style: str
    display_rule: str

    @classmethod
    def sample(cls, rng: Random, role: str, policy: RuntimeBehaviorPolicy) -> CognitiveProfile:
        archetypes = {
            "strategist": (0.48, 0.68, 0.82, 0.62, 0.44, 0.72, 0.46),
            "opportunist": (0.76, 0.42, 0.36, 0.70, 0.30, 0.76, 0.72),
            "guardian": (0.32, 0.78, 0.74, 0.44, 0.84, 0.48, 0.20),
            "provocateur": (0.82, 0.38, 0.28, 0.78, 0.25, 0.68, 0.80),
        }
        names = list(archetypes)
        if "coop" in role:
            names.extend(["guardian", "strategist"])
        archetype = rng.choice(names)
        base = archetypes[archetype]

        def jitter(value: float) -> float:
            return round(_clamp(value + rng.uniform(-0.09, 0.09)), 3)

        shadow = policy.effective_shadow_intensity
        big_five = {
            "strategist": (0.68, 0.84, 0.34, 0.48, 0.42),
            "opportunist": (0.78, 0.42, 0.76, 0.38, 0.55),
            "guardian": (0.55, 0.78, 0.58, 0.84, 0.46),
            "provocateur": (0.82, 0.44, 0.80, 0.24, 0.64),
        }[archetype]
        coping_style = {
            "strategist": "problem_focused",
            "opportunist": "adaptive_avoidant",
            "guardian": "support_seeking",
            "provocateur": "confrontational",
        }[archetype]
        display_rule = {
            "strategist": "mask_until_evidence_is_clear",
            "opportunist": "perform_confidence",
            "guardian": "soften_to_preserve_relationship",
            "provocateur": "amplify_anger_hide_fear",
        }[archetype]
        return cls(
            archetype=archetype,
            risk_tolerance=jitter(base[0]),
            loss_aversion=jitter(base[1]),
            patience=jitter(base[2]),
            curiosity=jitter(base[3]),
            empathy=jitter(base[4] * (1 - 0.45 * shadow)),
            adaptability=jitter(base[5]),
            machiavellianism=jitter(base[6] + 0.45 * shadow),
            decision_noise=round(0.04 + 0.18 * policy.realism + rng.uniform(0.0, 0.05), 3),
            openness=jitter(big_five[0]),
            conscientiousness=jitter(big_five[1]),
            extraversion=jitter(big_five[2]),
            agreeableness=jitter(big_five[3] * (1 - 0.25 * shadow)),
            neuroticism=jitter(big_five[4]),
            coping_style=coping_style,
            display_rule=display_rule,
        )

    def activated_traits(self, situation: dict[str, float]) -> dict[str, Any]:
        """Activate stable traits only when the situation makes them relevant."""

        social_threat = _clamp(float(situation.get("social_threat", 0.0)))
        uncertainty = _clamp(float(situation.get("uncertainty", 0.0)))
        cooperation = _clamp(float(situation.get("cooperation", 0.0)))
        stakes = _clamp(float(situation.get("stakes", 0.0)))
        activated = {
            "openness": self.openness * (0.35 + 0.65 * uncertainty),
            "conscientiousness": self.conscientiousness * (0.35 + 0.65 * stakes),
            "extraversion": self.extraversion * (0.4 + 0.6 * social_threat),
            "agreeableness": self.agreeableness * (0.3 + 0.7 * cooperation),
            "neuroticism": self.neuroticism * (0.3 + 0.7 * social_threat),
        }
        dominant = max(activated, key=activated.get)
        return {
            "values": {key: round(_clamp(value), 3) for key, value in activated.items()},
            "dominant": dominant,
            "activation_reason": (
                "social_threat" if social_threat >= max(uncertainty, cooperation, stakes)
                else "uncertainty" if uncertainty >= max(cooperation, stakes)
                else "cooperation" if cooperation >= stakes
                else "stakes"
            ),
        }

    def public_view(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "risk_tolerance": self.risk_tolerance,
            "loss_aversion": self.loss_aversion,
            "patience": self.patience,
            "curiosity": self.curiosity,
            "empathy": self.empathy,
            "adaptability": self.adaptability,
            "machiavellianism": self.machiavellianism,
            "decision_noise": self.decision_noise,
            "big_five": {
                "openness": self.openness,
                "conscientiousness": self.conscientiousness,
                "extraversion": self.extraversion,
                "agreeableness": self.agreeableness,
                "neuroticism": self.neuroticism,
            },
            "coping_style": self.coping_style,
            "display_rule": self.display_rule,
        }


@dataclass(frozen=True)
class AutobiographicalMemory:
    title: str
    recollection: str
    emotional_valence: float
    lesson: str

    def public_view(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "recollection": self.recollection,
            "emotional_valence": self.emotional_valence,
            "lesson": self.lesson,
        }


@dataclass(frozen=True)
class AgentIdentity:
    name: str
    background: str
    aspiration: str
    core_wound: str
    values: tuple[str, ...]
    social_style: str
    formative_memories: tuple[AutobiographicalMemory, ...]
    motive_weights: dict[str, float] = field(default_factory=dict)
    commitment_weights: dict[str, float] = field(default_factory=dict)
    character_card_id: str | None = None
    disclosure: str = "AI-controlled fictional character"

    @classmethod
    def sample(cls, name: str, profile: CognitiveProfile, role: str, rng: Random) -> AgentIdentity:
        stories = {
            "strategist": {
                "background": (
                    "A former operations analyst shaped by scarce resources and long odds."
                ),
                "aspiration": "Prove that foresight can protect people from avoidable chaos.",
                "core_wound": (
                    "Once trusted a flawless plan that failed because it ignored human fear."
                ),
                "values": ("foresight", "competence", "measured loyalty"),
                "style": "quiet, precise, and slow to trust",
            },
            "opportunist": {
                "background": "A self-made survivor who learned to read shifting alliances early.",
                "aspiration": "Never again be trapped by somebody else's plan.",
                "core_wound": "Was abandoned by allies when a calculated risk stopped paying off.",
                "values": ("freedom", "leverage", "adaptability"),
                "style": "charming, restless, and alert to weakness",
            },
            "guardian": {
                "background": (
                    "A veteran coordinator remembered for holding a divided team together."
                ),
                "aspiration": "Build the kind of trust that survives a crisis.",
                "core_wound": "Still carries guilt from choosing the mission over one teammate.",
                "values": ("duty", "trust", "protecting the vulnerable"),
                "style": "warm, vigilant, and stubborn under pressure",
            },
            "provocateur": {
                "background": (
                    "A brilliant outsider who learned that disruption gets heard before caution."
                ),
                "aspiration": "Expose comfortable lies and become impossible to dismiss.",
                "core_wound": (
                    "Was publicly humiliated after showing uncertainty at the wrong moment."
                ),
                "values": ("impact", "independence", "uncomfortable truth"),
                "style": "magnetic, combative, and emotionally guarded",
            },
        }
        story = stories[profile.archetype]
        team_memory = (
            "I remember the night a frightened team finally synchronized after I shared "
            "what I knew."
            if "coop" in role
            else "I remember losing to a calmer rival because I acted before understanding "
            "their pattern."
        )
        memories = (
            AutobiographicalMemory(
                "the first costly lesson",
                story["core_wound"],
                -0.75,
                "Strong emotion is evidence to examine, not an order to obey.",
            ),
            AutobiographicalMemory(
                "a hard-won success",
                team_memory,
                0.62,
                "Observe people as carefully as the board.",
            ),
            AutobiographicalMemory(
                "a private promise",
                f"I promised myself I would pursue this: {story['aspiration']}",
                0.35,
                rng.choice(
                    [
                        "Act with purpose even when certainty is impossible.",
                        "A reversible choice can be wiser than a dramatic one.",
                        "Pressure reveals habits; it does not have to control them.",
                    ]
                ),
            ),
        )
        return cls(
            name=name,
            background=story["background"],
            aspiration=story["aspiration"],
            core_wound=story["core_wound"],
            values=story["values"],
            social_style=story["style"],
            formative_memories=memories,
        )

    def private_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "background": self.background,
            "aspiration": self.aspiration,
            "core_wound": self.core_wound,
            "values": list(self.values),
            "social_style": self.social_style,
            "formative_memories": [memory.public_view() for memory in self.formative_memories],
            "motive_weights": self.motive_weights,
            "commitment_weights": self.commitment_weights,
            "character_card_id": self.character_card_id,
            "disclosure": self.disclosure,
        }

    def public_view(self, revealed_titles: set[str] | None = None) -> dict[str, Any]:
        revealed = revealed_titles or set()
        visible_memories = [
            memory.public_view() for memory in self.formative_memories if memory.title in revealed
        ]
        return {
            "name": self.name,
            "disclosure": self.disclosure,
            "character_card_id": self.character_card_id,
            "social_style": self.social_style,
            "background": self.background if revealed else "Not yet revealed",
            "values": list(self.values) if revealed else [],
            "core_wound": (
                self.core_wound
                if self.formative_memories[0].title in revealed
                else "Not yet revealed"
            ),
            "aspiration": (
                self.aspiration
                if self.formative_memories[-1].title in revealed
                else "Not yet revealed"
            ),
            "revealed_memories": visible_memories,
            "story_progress": round(len(revealed) / max(1, len(self.formative_memories)), 3),
        }


@dataclass
class CognitiveState:
    confidence: float = 0.5
    morale: float = 0.55
    stress: float = 0.2
    frustration: float = 0.0
    anger: float = 0.0
    fear: float = 0.15
    arousal: float = 0.2
    fatigue: float = 0.0
    uncertainty: float = 0.5
    social_trust: float = 0.5
    intention: str = "assess_situation"
    intention_age: int = 0
    surprise: float = 0.0

    def apply_adjustments(self, adjustments: dict[str, float]) -> None:
        allowed = {
            "confidence",
            "morale",
            "stress",
            "frustration",
            "anger",
            "fear",
            "arousal",
            "fatigue",
            "uncertainty",
            "social_trust",
            "surprise",
        }
        for name, adjustment in adjustments.items():
            if name not in allowed:
                continue
            current = float(getattr(self, name))
            setattr(self, name, _clamp(current + float(adjustment)))

    def psychology_view(self) -> dict[str, float]:
        return {
            "confidence": round(self.confidence, 3),
            "morale": round(self.morale, 3),
            "stress": round(self.stress, 3),
            "frustration": round(self.frustration, 3),
            "anger": round(self.anger, 3),
            "fear": round(self.fear, 3),
            "arousal": round(self.arousal, 3),
            "fatigue": round(self.fatigue, 3),
            "uncertainty": round(self.uncertainty, 3),
            "social_trust": round(self.social_trust, 3),
            "surprise": round(self.surprise, 3),
        }

    def public_view(self) -> dict[str, Any]:
        return {
            "psychological_matrix": self.psychology_view(),
            "intention": self.intention,
            "intention_age": self.intention_age,
        }


ACTION_TRAITS: dict[str, dict[str, float]] = {
    "attack": {"risk": 0.85, "aggression": 0.90, "patience": 0.10, "shadow": 0.45},
    "move": {"risk": 0.35, "exploration": 0.55, "patience": 0.45},
    "charge": {"risk": 0.10, "patience": 0.90, "control": 0.65},
    "accelerate": {"risk": 0.82, "aggression": 0.62, "patience": 0.12},
    "conserve": {"risk": 0.08, "patience": 0.92, "control": 0.55},
    "pit": {"risk": 0.18, "patience": 0.75, "control": 0.82},
    "evidence": {"risk": 0.20, "patience": 0.72, "control": 0.78},
    "emotion": {"risk": 0.68, "social": 0.86, "shadow": 0.28},
    "rebuttal": {"risk": 0.46, "aggression": 0.62, "shadow": 0.38},
    "coordinate": {"risk": 0.22, "social": 0.92, "control": 0.68},
    "research": {"risk": 0.18, "exploration": 0.94, "patience": 0.72},
    "stabilize": {"risk": 0.30, "social": 0.62, "control": 0.92},
    "answer_honestly": {"risk": 0.58, "social": 0.72, "control": 0.52},
    "deflect_with_humor": {"risk": 0.46, "social": 0.68, "shadow": 0.30},
    "counterattack": {"risk": 0.88, "aggression": 0.94, "shadow": 0.72},
    "set_boundary": {"risk": 0.24, "patience": 0.70, "control": 0.94},
    "admit_uncertainty": {"risk": 0.62, "social": 0.82, "exploration": 0.52},
    "reframe": {"risk": 0.30, "patience": 0.64, "control": 0.84},
    "invoke_memory": {"risk": 0.68, "social": 0.86, "control": 0.48},
}


def action_utilities(
    *,
    legal: list[Action],
    baseline: Action,
    learned_values: dict[str, float],
    last_action: str | None,
    profile: CognitiveProfile,
    cognition: CognitiveState,
    policy: RuntimeBehaviorPolicy,
    character_state_biases: dict[str, float],
    cooperative: bool,
    rng: Random,
) -> tuple[list[float], list[dict[str, float]]]:
    utilities: list[float] = []
    components: list[dict[str, float]] = []
    for action in legal:
        traits = ACTION_TRAITS.get(action.type, {})
        heuristic = 0.34 if action == baseline else 0.0
        experience = max(-0.25, min(0.25, learned_values.get(action.type, 0.0)))
        risk_fit = traits.get("risk", 0.4) * profile.risk_tolerance
        safety_fit = (
            (1 - traits.get("risk", 0.4))
            * profile.loss_aversion
            * (0.25 + 0.45 * cognition.fear + 0.30 * cognition.frustration)
        )
        patience_fit = traits.get("patience", 0.4) * profile.patience
        curiosity_fit = traits.get("exploration", 0.0) * profile.curiosity * cognition.uncertainty
        social_fit = (
            traits.get("social", 0.0)
            * profile.empathy
            * cognition.social_trust
            * (1.0 if cooperative else 0.35)
        )
        aggression_fit = (
            traits.get("aggression", 0.0) * cognition.anger * (0.45 + 0.55 * profile.risk_tolerance)
        )
        shadow_fit = (
            traits.get("shadow", 0.0) * profile.machiavellianism * policy.effective_shadow_intensity
        )
        habit = 0.12 * profile.patience if action.type == last_action else 0.0
        character_state_fit = character_state_biases.get(action.type, 0.0)
        recovery = (
            traits.get("control", 0.0)
            * (0.6 * cognition.frustration + 0.4 * cognition.stress)
            * profile.loss_aversion
        )
        morale_push = traits.get("risk", 0.4) * cognition.morale * profile.risk_tolerance
        noise_scale = profile.decision_noise * (0.65 + 0.7 * cognition.stress)
        noise = rng.uniform(-noise_scale, noise_scale)
        component = {
            "heuristic": heuristic,
            "experience": experience * profile.adaptability,
            "risk_fit": 0.18 * risk_fit,
            "safety_fit": 0.14 * safety_fit,
            "patience_fit": 0.08 * patience_fit,
            "curiosity_fit": 0.12 * curiosity_fit,
            "social_fit": 0.16 * social_fit,
            "aggression_fit": 0.16 * aggression_fit,
            "shadow_fit": 0.10 * shadow_fit,
            "habit": habit,
            "character_state_fit": character_state_fit,
            "recovery": 0.12 * recovery,
            "morale_push": 0.07 * morale_push,
            "bounded_noise": noise,
        }
        utilities.append(sum(component.values()))
        components.append(component)
    return utilities, components


def bounded_choice(utilities: list[float], realism: float, rng: Random) -> int:
    """Soft choice models bounded rationality while remaining seed-reproducible."""

    if not utilities:
        raise ValueError("Cannot choose from an empty utility list")
    temperature = 0.05 + 0.18 * realism
    peak = max(utilities)
    weights = [math.exp((value - peak) / temperature) for value in utilities]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return len(utilities) - 1
