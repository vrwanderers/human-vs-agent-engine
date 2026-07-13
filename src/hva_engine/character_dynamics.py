from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hva_engine.cognition import ACTION_TRAITS, AgentIdentity, CognitiveProfile, CognitiveState
from hva_engine.human_cognition import AppraisalState
from hva_engine.models import Action


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class MotiveState:
    name: str
    strength: float
    satisfaction: float = 0.5
    frustration: float = 0.0

    def pressure(self) -> float:
        return _clamp(self.strength * (0.55 + 0.45 * self.frustration))

    def public_view(self) -> dict[str, Any]:
        return {
            "strength": round(self.strength, 3),
            "satisfaction": round(self.satisfaction, 3),
            "frustration": round(self.frustration, 3),
            "pressure": round(self.pressure(), 3),
        }


@dataclass
class NarrativeDynamics:
    """Slow character state inspired by human-authored narrative decision points."""

    motives: dict[str, MotiveState]
    commitments: dict[str, float]
    secret_pressure: float = 0.2
    identity_dissonance: float = 0.0
    resentment: float = 0.0
    shame: float = 0.0
    moral_injury: float = 0.0
    hope: float = 0.5
    attachment: float = 0.45
    impulse_pressure: float = 0.2
    social_susceptibility: float = 0.35
    self_licensing: float = 0.15
    arc_stage: str = "guarded"
    consequence_trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_identity(
        cls, identity: AgentIdentity, profile: CognitiveProfile, cooperative: bool
    ) -> NarrativeDynamics:
        value_text = " ".join(identity.values).lower()

        def value_signal(*terms: str) -> float:
            return 0.18 if any(term in value_text for term in terms) else 0.0

        motives = {
            "self_preservation": MotiveState(
                "self_preservation", _clamp(0.42 + 0.38 * profile.loss_aversion)
            ),
            "truth": MotiveState(
                "truth", _clamp(0.38 + 0.30 * profile.curiosity + value_signal("truth"))
            ),
            "belonging": MotiveState(
                "belonging", _clamp(0.30 + 0.34 * profile.agreeableness)
            ),
            "autonomy": MotiveState(
                "autonomy", _clamp(0.38 + 0.34 * profile.openness + value_signal("freedom"))
            ),
            "status": MotiveState(
                "status", _clamp(0.24 + 0.40 * profile.extraversion)
            ),
            "duty": MotiveState(
                "duty",
                _clamp(
                    0.34
                    + 0.40 * profile.conscientiousness
                    + value_signal("duty", "competence")
                ),
            ),
            "redemption": MotiveState(
                "redemption", _clamp(0.35 + 0.25 * profile.neuroticism)
            ),
            "care": MotiveState(
                "care",
                _clamp(
                    0.32
                    + 0.42 * profile.empathy
                    + value_signal("protect", "trust", "loyalty")
                ),
            ),
        }
        if cooperative:
            motives["belonging"].strength = _clamp(motives["belonging"].strength + 0.15)
            motives["care"].strength = _clamp(motives["care"].strength + 0.12)
        return cls(
            motives=motives,
            commitments={
                "aspiration": 0.72,
                "core_values": _clamp(0.55 + 0.30 * profile.conscientiousness),
                "relationship": _clamp(0.35 + 0.30 * profile.agreeableness),
            },
            secret_pressure=_clamp(0.20 + 0.25 * profile.neuroticism),
            attachment=_clamp(0.35 + 0.35 * profile.agreeableness),
            impulse_pressure=_clamp(
                0.18
                + 0.32 * profile.neuroticism
                + 0.18 * profile.extraversion
                - 0.20 * profile.conscientiousness
            ),
            social_susceptibility=_clamp(
                0.16 + 0.42 * profile.agreeableness + 0.18 * profile.neuroticism
            ),
            self_licensing=_clamp(
                0.12
                + 0.28 * profile.loss_aversion
                + 0.16 * profile.extraversion
                - 0.20 * profile.conscientiousness
            ),
        )

    def update_before_decision(
        self,
        appraisal: AppraisalState,
        cognition: CognitiveState,
        social_trust: float,
    ) -> None:
        threat = appraisal.social_threat
        incongruence = 1 - appraisal.goal_congruence
        self.motives["self_preservation"].frustration = _clamp(
            0.72 * self.motives["self_preservation"].frustration + 0.28 * threat
        )
        self.motives["truth"].frustration = _clamp(
            0.78 * self.motives["truth"].frustration + 0.22 * cognition.uncertainty
        )
        self.motives["belonging"].frustration = _clamp(
            0.80 * self.motives["belonging"].frustration + 0.20 * (1 - social_trust)
        )
        self.motives["duty"].frustration = _clamp(
            0.80 * self.motives["duty"].frustration + 0.20 * incongruence
        )
        self.secret_pressure = _clamp(
            0.82 * self.secret_pressure
            + 0.10 * threat
            + 0.08 * max(self.shame, self.identity_dissonance)
        )
        self.resentment = _clamp(0.92 * self.resentment + 0.11 * threat * (1 - social_trust))
        self.impulse_pressure = _clamp(
            0.84 * self.impulse_pressure
            + 0.10 * cognition.stress
            + 0.08 * self.resentment
            + 0.06 * incongruence
        )
        self.social_susceptibility = _clamp(
            0.90 * self.social_susceptibility
            + 0.07 * threat
            + 0.06 * self.motives["belonging"].frustration
            - 0.05 * social_trust
        )
        self.self_licensing = _clamp(
            0.91 * self.self_licensing
            + 0.08 * self.resentment
            + 0.06 * self.motives["status"].frustration
        )
        self.hope = _clamp(
            0.90 * self.hope
            + 0.08 * appraisal.controllability
            + 0.04 * cognition.morale
            - 0.07 * incongruence
        )
        self._update_arc()

    def action_biases(self, legal: list[Action]) -> dict[str, float]:
        biases: dict[str, float] = {}
        pressures = {name: motive.pressure() for name, motive in self.motives.items()}
        for action in legal:
            traits = ACTION_TRAITS.get(action.type, {})
            truth_fit = (
                1.0
                if action.type in {"evidence", "research", "answer_honestly", "admit_uncertainty"}
                else traits.get("exploration", 0.0)
            )
            care_fit = traits.get("social", 0.0)
            duty_fit = traits.get("control", 0.0) + 0.35 * traits.get("patience", 0.0)
            autonomy_fit = traits.get("risk", 0.0) + 0.30 * traits.get("aggression", 0.0)
            safety_fit = 1 - traits.get("risk", 0.4)
            redemption_fit = (
                1.0
                if action.type in {"answer_honestly", "invoke_memory", "coordinate", "stabilize"}
                else 0.0
            )
            status_fit = traits.get("aggression", 0.0) + 0.35 * traits.get("social", 0.0)
            motive_fit = (
                0.08 * pressures["truth"] * truth_fit
                + 0.08 * pressures["care"] * care_fit
                + 0.07 * pressures["duty"] * duty_fit
                + 0.06 * pressures["autonomy"] * autonomy_fit
                + 0.07 * pressures["self_preservation"] * safety_fit
                + 0.07 * pressures["redemption"] * redemption_fit
                + 0.04 * pressures["status"] * status_fit
            )
            legacy = 0.0
            distortion = (
                0.06 * self.impulse_pressure * traits.get("aggression", 0.0)
                + 0.05 * self.impulse_pressure * traits.get("risk", 0.0)
                + 0.05 * self.social_susceptibility * traits.get("social", 0.0)
                + 0.05 * self.self_licensing * status_fit
            )
            if action.type == "counterattack":
                legacy += 0.13 * self.resentment - 0.10 * self.moral_injury
            if action.type in {"answer_honestly", "admit_uncertainty", "invoke_memory"}:
                legacy += 0.10 * self.shame + 0.08 * self.identity_dissonance
            if action.type == "set_boundary":
                legacy += 0.10 * self.secret_pressure + 0.08 * self.resentment
            biases[action.type] = motive_fit + legacy + distortion
        return biases

    def record_consequence(
        self,
        *,
        action_type: str,
        score_delta: float,
        surprise: float,
        cognition: CognitiveState,
        profile: CognitiveProfile,
    ) -> None:
        traits = ACTION_TRAITS.get(action_type, {})
        aggressive = traits.get("aggression", 0.0)
        social = traits.get("social", 0.0)
        honest = action_type in {"answer_honestly", "admit_uncertainty", "invoke_memory"}
        value_conflict = aggressive * profile.empathy * max(0.0, -score_delta)
        avoidance = (
            action_type in {"deflect_with_humor", "set_boundary", "conserve"}
            and self.motives["truth"].pressure() > 0.55
        )
        self.identity_dissonance = _clamp(
            0.84 * self.identity_dissonance
            + 0.24 * value_conflict
            + (0.09 if avoidance else 0.0)
            - (0.12 if honest else 0.0)
        )
        self.moral_injury = _clamp(
            0.92 * self.moral_injury + 0.18 * value_conflict + 0.08 * surprise * aggressive
        )
        self.shame = _clamp(
            0.86 * self.shame
            + 0.20 * max(0.0, -score_delta) * profile.conscientiousness
            + 0.10 * self.identity_dissonance
            - (0.10 if honest else 0.0)
        )
        self.hope = _clamp(
            0.88 * self.hope
            + 0.16 * max(0.0, score_delta)
            + 0.07 * social
            - 0.10 * max(0.0, -score_delta)
        )
        self.self_licensing = _clamp(
            0.88 * self.self_licensing
            + 0.14 * max(0.0, -score_delta)
            + 0.08 * self.resentment
            - (0.12 if honest else 0.0)
        )
        self.impulse_pressure = _clamp(
            0.86 * self.impulse_pressure
            + 0.10 * surprise
            + 0.08 * max(0.0, -score_delta)
            - 0.06 * max(0.0, score_delta)
        )
        for motive in self.motives.values():
            motive.satisfaction = _clamp(
                0.86 * motive.satisfaction + 0.14 * (0.5 + max(-0.5, min(0.5, score_delta)))
            )
        self.consequence_trace.append(
            {
                "action": action_type,
                "score_delta": round(score_delta, 3),
                "dissonance": round(self.identity_dissonance, 3),
                "resentment": round(self.resentment, 3),
                "shame": round(self.shame, 3),
                "moral_injury": round(self.moral_injury, 3),
                "hope": round(self.hope, 3),
                "impulse_pressure": round(self.impulse_pressure, 3),
                "self_licensing": round(self.self_licensing, 3),
            }
        )
        self.consequence_trace = self.consequence_trace[-16:]
        self._update_arc()

    def active_conflict(self) -> dict[str, Any]:
        approach_names = {"truth", "autonomy", "status", "duty", "redemption", "care"}
        approach = max(
            (motive for name, motive in self.motives.items() if name in approach_names),
            key=lambda item: item.pressure(),
        )
        avoidance = self.motives["self_preservation"]
        intensity = _clamp(min(approach.pressure(), avoidance.pressure()) * 1.35)
        return {
            "approach_motive": approach.name,
            "avoidance_motive": avoidance.name,
            "intensity": round(intensity, 3),
        }

    def _update_arc(self) -> None:
        if self.moral_injury > 0.62 and self.hope < 0.38:
            self.arc_stage = "breaking"
        elif self.resentment > 0.65 and self.identity_dissonance < 0.42:
            self.arc_stage = "hardened"
        elif self.identity_dissonance > 0.58:
            self.arc_stage = "rationalizing" if self.secret_pressure > 0.55 else "conflicted"
        elif self.shame > 0.42 and self.hope > 0.52:
            self.arc_stage = "repairing"
        elif self.hope > 0.72 and self.identity_dissonance < 0.22:
            self.arc_stage = "opening"
        else:
            self.arc_stage = "guarded"

    def public_view(self) -> dict[str, Any]:
        return {
            "motives": {name: motive.public_view() for name, motive in self.motives.items()},
            "commitments": {key: round(value, 3) for key, value in self.commitments.items()},
            "active_conflict": self.active_conflict(),
            "secret_pressure": round(self.secret_pressure, 3),
            "identity_dissonance": round(self.identity_dissonance, 3),
            "resentment": round(self.resentment, 3),
            "shame": round(self.shame, 3),
            "moral_injury": round(self.moral_injury, 3),
            "hope": round(self.hope, 3),
            "attachment": round(self.attachment, 3),
            "impulse_pressure": round(self.impulse_pressure, 3),
            "social_susceptibility": round(self.social_susceptibility, 3),
            "self_licensing": round(self.self_licensing, 3),
            "arc_stage": self.arc_stage,
            "consequence_count": len(self.consequence_trace),
        }
