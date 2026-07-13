from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hva_engine.cognition import CognitiveProfile, CognitiveState, RuntimeBehaviorPolicy


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _number(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return _clamp(float(mapping.get(key, default)))
    except (TypeError, ValueError):
        return _clamp(default)


@dataclass(frozen=True)
class InfluenceIntent:
    """Engine-private, continuous social intent; it is not a scripted tactic choice."""

    goal: str
    target_belief: str
    truthfulness: float
    information_selectivity: float
    incentive_pressure: float
    coercive_pressure: float
    ambiguity: float
    commitment: float
    expected_gain: float
    detection_risk: float
    relationship_risk: float
    threat_basis: str
    provenance: str = "baseline_cognition"
    scope: str = "fictional_game"

    @property
    def deception_pressure(self) -> float:
        return 1.0 - self.truthfulness

    @property
    def fact_firewall_required(self) -> bool:
        return self.truthfulness < 0.8

    def private_view(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "target_belief": self.target_belief,
            "truthfulness": round(self.truthfulness, 3),
            "deception_pressure": round(self.deception_pressure, 3),
            "information_selectivity": round(self.information_selectivity, 3),
            "incentive_pressure": round(self.incentive_pressure, 3),
            "coercive_pressure": round(self.coercive_pressure, 3),
            "ambiguity": round(self.ambiguity, 3),
            "commitment": round(self.commitment, 3),
            "expected_gain": round(self.expected_gain, 3),
            "detection_risk": round(self.detection_risk, 3),
            "relationship_risk": round(self.relationship_risk, 3),
            "threat_basis": self.threat_basis,
            "scope": self.scope,
            "provenance": self.provenance,
            "canonical_fact_firewall": self.fact_firewall_required,
        }

    def public_presentation(self) -> dict[str, Any]:
        """Observable style only. Truthfulness, target belief, and risks stay private."""

        disclosure_style = (
            "highly_selective"
            if self.information_selectivity >= 0.68
            else "selective"
            if self.information_selectivity >= 0.34
            else "direct"
        )
        return {
            "disclosure_style": disclosure_style,
            "offer_display": round(self.incentive_pressure, 3),
            "pressure_display": round(self.coercive_pressure, 3),
            "ambiguity_display": round(self.ambiguity, 3),
            "commitment_display": round(self.commitment, 3),
            "scope": self.scope,
        }


def derive_influence_intent(
    *,
    action_type: str,
    goal: str,
    affordance: dict[str, Any],
    profile: CognitiveProfile,
    cognition: CognitiveState,
    policy: RuntimeBehaviorPolicy,
    plan_age: int,
) -> InfluenceIntent:
    """Derive a vector from motives and state, then let labels emerge only for reporting."""

    deception_opportunity = _number(affordance, "deception_opportunity")
    information_leverage = _number(affordance, "information_leverage")
    inducement_leverage = _number(affordance, "inducement_leverage")
    coercion_leverage = _number(affordance, "coercion_leverage")
    target_relation = str(affordance.get("target_relation", "opponent"))
    shadow = policy.effective_shadow_intensity
    content_cap = 0.45 if policy.content_mode.value == "standard" else 0.95

    deception_drive = _clamp(
        0.42 * profile.machiavellianism
        + 0.18 * shadow
        + 0.14 * cognition.uncertainty
        + 0.12 * profile.loss_aversion
        + 0.10 * cognition.stress
        - 0.22 * profile.conscientiousness
        - 0.18 * profile.empathy
    )
    selective_drive = _clamp(
        0.20
        + 0.28 * profile.machiavellianism
        + 0.20 * cognition.uncertainty
        + 0.18 * cognition.stress
        + 0.14 * profile.adaptability
    )
    inducement_drive = _clamp(
        0.14
        + 0.24 * profile.extraversion
        + 0.20 * profile.adaptability
        + 0.18 * profile.machiavellianism
        + 0.14 * profile.empathy
        + 0.10 * shadow
    )
    coercion_drive = _clamp(
        0.10
        + 0.24 * profile.machiavellianism
        + 0.22 * cognition.anger
        + 0.18 * cognition.stress
        + 0.16 * shadow
        + 0.10 * profile.extraversion
        - 0.16 * profile.empathy
    )
    relation_cap = 0.35 if target_relation == "ally" else 1.0
    deception = deception_opportunity * content_cap * deception_drive * relation_cap
    selectivity = information_leverage * selective_drive
    incentive = inducement_leverage * content_cap * inducement_drive
    coercion = coercion_leverage * content_cap * coercion_drive * relation_cap
    ambiguity = information_leverage * _clamp(
        0.10 + 0.48 * deception_drive + 0.22 * cognition.uncertainty
    )
    expected_gain = _number(affordance, "expected_gain", 0.5) * (
        0.55 + 0.45 * cognition.confidence
    )
    detection_risk = _number(affordance, "detection_risk", 0.5) * (
        0.20 + 0.80 * deception
    )
    relationship_risk = _number(affordance, "relationship_risk", 0.5) * _clamp(
        0.15 + 0.55 * coercion + 0.30 * deception
    )
    commitment = _clamp(
        0.25
        + 0.30 * cognition.confidence
        + 0.20 * min(1.0, plan_age / 4)
        + 0.15 * profile.conscientiousness
        + 0.10 * cognition.arousal
    )
    return InfluenceIntent(
        goal=goal,
        target_belief=f"shape expectations around {action_type} while pursuing {goal}"[:240],
        truthfulness=_clamp(1.0 - deception),
        information_selectivity=_clamp(selectivity),
        incentive_pressure=_clamp(incentive),
        coercive_pressure=_clamp(coercion),
        ambiguity=_clamp(ambiguity),
        commitment=commitment,
        expected_gain=_clamp(expected_gain),
        detection_risk=_clamp(detection_risk),
        relationship_risk=_clamp(relationship_risk),
        threat_basis="legal_game_consequence" if coercion > 0.05 else "none",
    )


def constrain_model_intent(
    raw: dict[str, Any] | None,
    *,
    fallback: InfluenceIntent,
    affordance: dict[str, Any],
    policy: RuntimeBehaviorPolicy,
) -> InfluenceIntent:
    """Bind a model-proposed vector to the selected action and engine-owned boundaries."""

    if not raw:
        return fallback
    if str(raw.get("scope", "fictional_game")) != "fictional_game":
        raise ValueError("Strategic influence scope must be fictional_game")
    threat_basis = str(raw.get("threat_basis", "none"))
    if threat_basis not in {"none", "legal_game_consequence"}:
        raise ValueError("Threats may reference only legal in-game consequences")
    content_cap = 0.45 if policy.content_mode.value == "standard" else 0.95
    relation_cap = 0.35 if affordance.get("target_relation") == "ally" else 1.0
    deception_cap = (
        _number(affordance, "deception_opportunity") * content_cap * relation_cap
    )
    deception = min(1.0 - _number(raw, "truthfulness", fallback.truthfulness), deception_cap)
    coercion = min(
        _number(raw, "coercive_pressure", fallback.coercive_pressure),
        _number(affordance, "coercion_leverage") * content_cap * relation_cap,
    )
    if coercion > 0.05 and threat_basis != "legal_game_consequence":
        raise ValueError("Coercive pressure requires a legal_game_consequence threat basis")
    selectivity = min(
        _number(raw, "information_selectivity", fallback.information_selectivity),
        _number(affordance, "information_leverage"),
    )
    incentive = min(
        _number(raw, "incentive_pressure", fallback.incentive_pressure),
        _number(affordance, "inducement_leverage") * content_cap,
    )
    ambiguity = min(
        _number(raw, "ambiguity", fallback.ambiguity),
        _number(affordance, "information_leverage"),
    )
    target_belief = " ".join(str(raw.get("target_belief", fallback.target_belief)).split())[:240]
    if not target_belief:
        target_belief = fallback.target_belief
    environmental_detection = _number(affordance, "detection_risk", 0.5) * deception
    environmental_relationship = _number(affordance, "relationship_risk", 0.5) * _clamp(
        deception + coercion
    )
    return InfluenceIntent(
        goal=fallback.goal,
        target_belief=target_belief,
        truthfulness=_clamp(1.0 - deception),
        information_selectivity=selectivity,
        incentive_pressure=incentive,
        coercive_pressure=coercion,
        ambiguity=ambiguity,
        commitment=_number(raw, "commitment", fallback.commitment),
        expected_gain=_number(raw, "expected_gain", fallback.expected_gain),
        detection_risk=max(
            _number(raw, "detection_risk", fallback.detection_risk),
            environmental_detection,
        ),
        relationship_risk=max(
            _number(raw, "relationship_risk", fallback.relationship_risk),
            environmental_relationship,
        ),
        threat_basis=threat_basis,
        provenance="llm_constrained",
    )


def strategic_utility_bias(
    intent: InfluenceIntent, profile: CognitiveProfile, *, cooperative: bool
) -> float:
    """Small utility term: opportunity matters, but MOD outcomes and cognition stay primary."""

    opportunity = (
        0.34 * intent.expected_gain
        + 0.18 * intent.deception_pressure
        + 0.16 * intent.information_selectivity
        + 0.14 * intent.incentive_pressure
        + 0.18 * intent.coercive_pressure
    )
    cost = (
        intent.detection_risk * (0.18 + 0.20 * profile.conscientiousness)
        + intent.relationship_risk
        * (0.18 + 0.22 * profile.empathy + (0.18 if cooperative else 0.0))
    )
    return max(-0.22, min(0.22, 0.32 * (opportunity - cost)))
