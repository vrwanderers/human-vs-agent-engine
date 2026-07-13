from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hva_engine.character_dynamics import NarrativeDynamics
from hva_engine.cognition import (
    AgentIdentity,
    CognitiveProfile,
    CognitiveState,
    RuntimeBehaviorPolicy,
    action_utilities,
    bounded_choice,
)
from hva_engine.context import ContextComposer, ContextPacket, SharedFact
from hva_engine.fact_graph import AgentFactGraph, FactGraphError
from hva_engine.human_cognition import (
    AppraisalState,
    DecisionMode,
    MemorySystem,
    PlanState,
    SocialBelief,
    appraise,
    select_decision_mode,
)
from hva_engine.llm import LLMDecisionClient
from hva_engine.models import Action, GameEvent
from hva_engine.mods.base import GameMod
from hva_engine.social_influence import (
    InfluenceIntent,
    constrain_model_intent,
    derive_influence_intent,
    strategic_utility_bias,
)


@dataclass
class MemoryItem:
    turn: int
    situation: str
    action: str
    outcome_events: list[str]
    score_before: float
    score_after: float
    score_delta: float
    predicted_events: list[str]
    surprise: float
    regret: float


@dataclass
class AgentBrain:
    """Inspectable human-like brain without exposing private chain-of-thought."""

    player_id: str
    role: str
    profile: CognitiveProfile
    identity: AgentIdentity
    behavior_policy: RuntimeBehaviorPolicy
    fact_graph: AgentFactGraph
    memory_limit: int = 12
    memory: deque[MemoryItem] = field(default_factory=lambda: deque(maxlen=12))
    world_model: dict[str, Any] = field(default_factory=dict)
    cognition: CognitiveState = field(default_factory=CognitiveState)
    opponent_model: dict[str, Any] = field(default_factory=dict)
    memory_system: MemorySystem = field(default_factory=MemorySystem)
    social_belief: SocialBelief = field(default_factory=SocialBelief)
    plan: PlanState = field(default_factory=PlanState)
    last_appraisal: AppraisalState | None = None
    decision_mode: DecisionMode = DecisionMode.DELIBERATIVE
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    activated_traits: dict[str, Any] = field(default_factory=dict)
    narrative_dynamics: NarrativeDynamics | None = None
    decisions: int = 0
    last_context: ContextPacket | None = None
    _match_id: str = ""
    _shared_facts: list[SharedFact] = field(default_factory=list)
    _observed_event_seq: int = 0
    _revealed_story_titles: set[str] = field(default_factory=set)
    _last_interpretation_fact_id: str | None = None

    def __post_init__(self) -> None:
        if self.memory.maxlen != self.memory_limit:
            self.memory = deque(self.memory, maxlen=self.memory_limit)
        if self.narrative_dynamics is None:
            self.narrative_dynamics = NarrativeDynamics.from_identity(
                self.identity, self.profile, "coop" in self.role
            )

    def observe(
        self,
        mod: GameMod,
        state: dict[str, Any],
        scores: dict[str, float],
        events: list[GameEvent],
        match_id: str,
        shared_facts: list[SharedFact] | None = None,
    ) -> dict[str, Any]:
        turn = int(state.get("turn", len(events)))
        rivals = {pid: score for pid, score in scores.items() if pid != self.player_id}
        own_score = scores.get(self.player_id, 0.0)
        rival_best = max(rivals.values(), default=own_score)
        margin = own_score - rival_best
        latest_delta = self.memory[-1].score_delta if self.memory else 0.0
        self.cognition.confidence = max(
            0.05,
            min(0.95, 0.55 * self.cognition.confidence + 0.45 * (0.5 + margin / 3)),
        )
        negative_signal = max(0.0, -latest_delta) + max(0.0, -margin) * 0.2
        self.cognition.frustration = max(
            0.0, min(1.0, self.cognition.frustration * 0.72 + negative_signal)
        )
        self.cognition.arousal = max(
            0.1, min(1.0, 0.25 + abs(margin) * 0.2 + self.cognition.frustration * 0.35)
        )
        self.cognition.fatigue = min(0.85, self.decisions / 32)
        self.cognition.uncertainty = max(
            0.08, min(0.92, 0.62 - 0.035 * len(self.memory) + 0.2 * self.cognition.surprise)
        )
        setback_streak = 0
        for item in reversed(self.memory):
            if item.score_delta >= 0:
                break
            setback_streak += 1
        pressure = (
            0.32 * self.cognition.frustration
            + 0.22 * self.cognition.uncertainty
            + 0.18 * self.cognition.fatigue
            + 0.16 * self.cognition.surprise
            + 0.12 * min(1.0, setback_streak / 3)
        )
        self.cognition.stress = max(0.03, min(0.98, 0.52 * self.cognition.stress + 0.48 * pressure))
        self.cognition.morale = max(
            0.03,
            min(
                0.98,
                0.52 * self.cognition.confidence
                + 0.24 * (1 - self.cognition.frustration)
                + 0.14 * (1 - self.cognition.fatigue)
                + 0.10 * max(0.0, latest_delta),
            ),
        )
        self.cognition.anger = max(
            0.0,
            min(
                1.0,
                self.cognition.anger * 0.58
                + self.cognition.frustration * (0.35 + 0.45 * self.profile.machiavellianism),
            ),
        )
        self.cognition.fear = max(
            0.02,
            min(
                0.95,
                0.50 * self.cognition.fear
                + 0.30 * self.cognition.stress * self.profile.loss_aversion
                + 0.20 * self.cognition.uncertainty,
            ),
        )
        if "coop" in self.role:
            shared_signal = min(1.0, len(shared_facts or []) / 6)
            self.cognition.social_trust = max(
                0.1,
                min(
                    0.95,
                    0.65 * self.cognition.social_trust
                    + 0.25 * shared_signal
                    + 0.10 * self.cognition.morale,
                ),
            )
        mod_psychology = mod.agent_psychological_signals(state, self.player_id)
        self.cognition.apply_adjustments(mod_psychology)
        new_external_events = [
            event
            for event in events
            if event.seq > self._observed_event_seq and event.actor_id != self.player_id
        ]
        hostile_severity = max(
            (
                float(event.payload.get("severity", 0.0))
                for event in new_external_events
                if event.type == "interview_question"
            ),
            default=0.0,
        )
        self.last_appraisal = appraise(
            score_delta=latest_delta,
            margin=margin,
            surprise=self.cognition.surprise,
            mod_signals=mod_psychology,
            hostile_severity=hostile_severity,
            uncertainty=self.cognition.uncertainty,
        )
        self.cognition.apply_adjustments(
            {
                "stress": 0.10 * self.last_appraisal.social_threat
                + 0.08 * (1 - self.last_appraisal.goal_congruence)
                - 0.06 * self.last_appraisal.controllability,
                "anger": 0.09
                * self.last_appraisal.other_agency
                * (1 - self.last_appraisal.norm_compatibility),
                "fear": 0.08
                * self.last_appraisal.social_threat
                * (1 - self.last_appraisal.controllability),
            }
        )
        self.social_belief.update(
            hostile_severity=hostile_severity,
            cooperative_signal=min(1.0, len(shared_facts or []) / 4),
            observed=bool(new_external_events),
        )
        assert self.narrative_dynamics is not None
        self.narrative_dynamics.update_before_decision(
            self.last_appraisal,
            self.cognition,
            self.social_belief.trust,
        )
        self._update_opponent_model(events)
        self._update_intention("coop" in self.role)
        self.plan.update(
            desired_goal=self.cognition.intention,
            surprise=self.cognition.surprise,
            stress=self.cognition.stress,
            goal_congruence=self.last_appraisal.goal_congruence,
        )
        self.activated_traits = self.profile.activated_traits(
            {
                "social_threat": self.last_appraisal.social_threat,
                "uncertainty": self.cognition.uncertainty,
                "cooperation": 1.0 if "coop" in self.role else self.social_belief.trust,
                "stakes": min(1.0, abs(margin) + self.cognition.stress),
            }
        )
        self.decision_mode = select_decision_mode(
            stress=self.cognition.stress,
            uncertainty=self.cognition.uncertainty,
            stakes=min(1.0, abs(margin) + hostile_severity),
        )
        intention_fact = self.fact_graph.upsert_runtime(
            "state.current_intention", self.cognition.intention
        )
        self.fact_graph.upsert_runtime(
            "state.psychological_matrix", self.cognition.psychology_view()
        )
        opponent_fact = self.fact_graph.upsert_runtime(
            "belief.opponent_pattern",
            {
                "dominant_action": self.opponent_model.get("dominant_action"),
                "confidence": self.opponent_model.get("confidence", 0.0),
            },
        )
        feeling = (
            "under_pressure"
            if self.cognition.stress >= 0.6
            else "frustrated"
            if self.cognition.frustration >= 0.45
            else "steady"
        )
        interpretation = self.fact_graph.propose(
            subject=self.player_id,
            predicate="belief.interpretation",
            value={
                "opponent_tendency": self.opponent_model.get("dominant_action"),
                "current_feeling": feeling,
                "chosen_intention": self.cognition.intention,
            },
            basis_fact_ids=[intention_fact.id, opponent_fact.id],
            supersedes_fact_id=self._last_interpretation_fact_id,
            confidence=max(0.35, float(self.opponent_model.get("confidence", 0.0))),
        )
        self._last_interpretation_fact_id = interpretation.id
        self.world_model = {
            "turn": turn,
            "self_score": round(own_score, 3),
            "other_scores": rivals,
            "score_margin": round(margin, 3),
            "terminal": mod.is_terminal(state),
            "recent_events": [event.type for event in events[-4:]],
            "objective": "shared_success" if "coop" in self.role else "outperform_opponents",
            "memory_depth": len(self.memory),
            "shared_fact_count": len(shared_facts or []),
            "belief_uncertainty": round(self.cognition.uncertainty, 3),
            "opponent_model": self.opponent_model,
            "social_belief": self.social_belief.public_view(),
            "appraisal": self.last_appraisal.public_view(),
            "plan": self.plan.public_view(),
            "activated_traits": self.activated_traits,
            "decision_mode": self.decision_mode.value,
            "narrative_dynamics": self.narrative_dynamics.public_view(),
            "mod_psychological_signals": {
                key: round(value, 3) for key, value in mod_psychology.items()
            },
        }
        self._match_id = match_id
        self._shared_facts = list(shared_facts or [])
        return self.world_model

    def decide(
        self,
        mod: GameMod,
        state: dict[str, Any],
        legal: list[Action],
        rng: Random,
        decision_client: LLMDecisionClient | None = None,
        llm_fallback: bool = True,
    ) -> tuple[Action, dict[str, Any]]:
        if not legal:
            raise ValueError("Agent cannot decide without legal actions")
        current_turn = int(state.get("turn", self.decisions))
        query = " ".join(
            [
                mod.id,
                self.cognition.intention,
                self.plan.subgoal,
                *(action.type for action in legal),
                *(self.world_model.get("recent_events", [])),
            ]
        )
        self.retrieved_memories = self.memory_system.retrieve(
            query,
            current_turn=current_turn,
            mood_valence=self.cognition.morale - self.cognition.frustration,
            limit=4,
        )
        narrative_affordances = mod.agent_narrative_affordances(
            state, self.player_id, legal
        )
        influence_affordances = mod.agent_influence_affordances(
            state, self.player_id, legal
        )
        baseline_influence_intents = {
            action.type: derive_influence_intent(
                action_type=action.type,
                goal=self.cognition.intention,
                affordance=influence_affordances.get(action.type, {}),
                profile=self.profile,
                cognition=self.cognition,
                policy=self.behavior_policy,
                plan_age=self.plan.age,
            )
            for action in legal
        }
        narrative_context = {
            **self.narrative_dynamics.public_view(),
            "legal_action_affordances": narrative_affordances,
        }
        self.last_context = ContextComposer().compose(
            match_id=self._match_id,
            agent_id=self.player_id,
            role=self.role,
            mod=mod,
            state=state,
            world_model=self.world_model,
            memory=self.retrieved_memories,
            legal_actions=legal,
            shared_facts=self._shared_facts,
            persona=self.profile.public_view(),
            identity=self.identity.private_view(),
            cognitive_state=self.cognition.public_view(),
            opponent_model=self.opponent_model,
            behavior_policy=self.behavior_policy.public_view(),
            fact_graph=self.fact_graph.private_view(),
            appraisal=self.last_appraisal.public_view() if self.last_appraisal else {},
            reflections=[item.public_view() for item in self.memory_system.reflections[-4:]],
            current_plan=self.plan.public_view(),
            social_beliefs=self.social_belief.public_view(),
            activated_traits=self.activated_traits,
            decision_mode=self.decision_mode.value,
            narrative_dynamics=narrative_context,
            influence_affordances=influence_affordances,
        )
        baseline_action = mod.agent_action(state, self.player_id, legal, rng)
        learned: dict[str, list[float]] = {}
        for item in self.memory:
            learned.setdefault(item.action, []).append(item.score_delta - 0.15 * item.regret)
        averages = {key: sum(values) / len(values) for key, values in learned.items()}
        averages.update(self.memory_system.procedural_values())
        last_action = self.memory[-1].action if self.memory else None
        narrative_biases = self.narrative_dynamics.action_biases(
            legal, narrative_affordances
        )
        influence_biases = {
            action.type: strategic_utility_bias(
                baseline_influence_intents[action.type],
                self.profile,
                cooperative="coop" in self.role,
            )
            for action in legal
        }
        combined_state_biases = {
            action.type: narrative_biases.get(action.type, 0.0)
            + influence_biases.get(action.type, 0.0)
            for action in legal
        }
        utilities, components = action_utilities(
            legal=legal,
            baseline=baseline_action,
            learned_values=averages,
            last_action=last_action,
            profile=self.profile,
            cognition=self.cognition,
            policy=self.behavior_policy,
            character_state_biases=combined_state_biases,
            cooperative="coop" in self.role,
            rng=rng,
        )
        llm_decision = None
        llm_error: dict[str, str] | None = None
        fact_proposal_result: dict[str, Any] | None = None
        influence_intent: InfluenceIntent | None = None
        if decision_client is not None:
            try:
                llm_decision = decision_client.choose_structured_sync(
                    self.last_context.messages,
                    [action.model_dump() for action in legal],
                )
                choice_index = llm_decision.action_index
                selected_type = legal[choice_index].type
                influence_intent = constrain_model_intent(
                    llm_decision.influence_intent,
                    fallback=baseline_influence_intents[selected_type],
                    affordance=influence_affordances.get(selected_type, {}),
                    policy=self.behavior_policy,
                )
                if influence_intent.fact_firewall_required:
                    fact_proposal_result = self._reject_deceptive_fact_proposals(
                        llm_decision.fact_proposals
                    )
                else:
                    fact_proposal_result = self.apply_fact_proposals(
                        llm_decision.fact_proposals
                    )
            except Exception as exc:
                if not llm_fallback:
                    raise
                llm_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                llm_decision = None
                choice_index = bounded_choice(
                    utilities,
                    min(
                        1.0,
                        max(
                            0.0,
                            self.behavior_policy.realism
                            + (
                                0.18
                                if self.decision_mode == DecisionMode.HABITUAL
                                else -0.08
                            ),
                        ),
                    ),
                    rng,
                )
        else:
            choice_index = bounded_choice(
                utilities,
                min(
                    1.0,
                    max(
                        0.0,
                        self.behavior_policy.realism
                        + (0.18 if self.decision_mode == DecisionMode.HABITUAL else -0.08),
                    ),
                ),
                rng,
            )
        action = legal[choice_index]
        if influence_intent is None:
            influence_intent = baseline_influence_intents[action.type]
        chosen_narrative_affordance = narrative_affordances.get(action.type, {})
        response_plan = (
            llm_decision.response_plan
            if llm_decision is not None
            else self._baseline_response_plan(mod, legal, utilities, choice_index)
        )
        internal_intensity = max(
            self.cognition.stress,
            self.cognition.anger,
            self.cognition.fear,
            self.cognition.arousal,
        )
        requested_display = float(response_plan.get("intensity", internal_intensity))
        masking = 0.34 * self.profile.conscientiousness + 0.22 * self.profile.agreeableness
        if self.profile.display_rule == "amplify_anger_hide_fear" and self.cognition.anger > 0.45:
            displayed_intensity = min(1.0, requested_display + 0.12 * self.profile.extraversion)
        else:
            displayed_intensity = max(0.0, requested_display * (1 - 0.35 * masking))
        response_plan = {
            **response_plan,
            "internal_emotion_intensity": round(internal_intensity, 3),
            "displayed_emotion_intensity": round(displayed_intensity, 3),
            "expression_gap": round(abs(internal_intensity - displayed_intensity), 3),
            "display_rule": self.profile.display_rule,
            "influence_presentation": influence_intent.public_presentation(),
        }
        best_index = max(range(len(utilities)), key=utilities.__getitem__)
        regret = max(0.0, utilities[best_index] - utilities[choice_index])
        memory_used = bool(self.retrieved_memories)
        memory_influenced = memory_used and bool(averages.get(action.type))
        ranked_factors = sorted(
            components[choice_index].items(), key=lambda item: abs(item[1]), reverse=True
        )
        dominant_factors = [name for name, value in ranked_factors[:3] if abs(value) >= 0.025]
        sorted_utilities = sorted(utilities, reverse=True)
        utility_gap = sorted_utilities[0] - sorted_utilities[1] if len(utilities) > 1 else 1.0
        confidence = max(
            0.15,
            min(
                0.95,
                0.45
                + utility_gap
                - 0.18 * self.cognition.uncertainty
                - 0.12 * self.cognition.fatigue,
            ),
        )
        prediction = self._prediction(mod, action)
        default_rationale = (
            f"Pursue {self.cognition.intention}; chose {action.type} from "
            f"{len(legal)} rule-valid options"
        )
        trace = {
            "policy": (
                f"llm:{decision_client.provider.name}"
                if llm_decision is not None and decision_client is not None
                else f"{mod.id}.baseline"
            ),
            "decision_source": "llm" if llm_decision is not None else "baseline",
            "rationale": llm_decision.reason
            if llm_decision and llm_decision.reason
            else default_rationale,
            "utterance": llm_decision.utterance if llm_decision else None,
            "response_plan": response_plan,
            "llm": (
                {
                    "provider": decision_client.provider.name,
                    "model": llm_decision.response.model,
                    "usage": llm_decision.response.usage,
                    "fact_proposals": self._public_fact_proposal_result(
                        fact_proposal_result, influence_intent
                    ),
                }
                if llm_decision is not None and decision_client is not None
                else None
            ),
            "llm_error": llm_error,
            "confidence": round(confidence, 3),
            "baseline_action": baseline_action.type,
            "memory_used": memory_used,
            "memory_influenced": memory_influenced,
            "memory_depth": len(self.memory),
            "retrieved_memories": self.retrieved_memories,
            "retrieved_memory_ids": [item["id"] for item in self.retrieved_memories],
            "reflections": [item.public_view() for item in self.memory_system.reflections[-4:]],
            "world_model": self.world_model,
            "persona": self.profile.public_view(),
            "identity": self.identity.public_view(self._revealed_story_titles),
            "cognitive_state": self.cognition.public_view(),
            "psychological_matrix": self.cognition.psychology_view(),
            "opponent_model": self.opponent_model,
            "social_beliefs": self.social_belief.public_view(),
            "appraisal": self.last_appraisal.public_view() if self.last_appraisal else {},
            "current_plan": self.plan.public_view(),
            "activated_traits": self.activated_traits,
            "decision_mode": self.decision_mode.value,
            "narrative_dynamics": self.narrative_dynamics.public_view(),
            "narrative_action_bias": round(narrative_biases.get(action.type, 0.0), 3),
            "strategic_influence_action_bias": round(
                influence_biases.get(action.type, 0.0), 3
            ),
            "narrative_action_affordance": chosen_narrative_affordance,
            "intention": self.cognition.intention,
            "behavior_policy": self.behavior_policy.public_view(),
            "fact_graph": self.fact_graph.public_view(),
            "deliberation_summary": {
                "options_considered": len(legal),
                "dominant_factors": dominant_factors,
                "utility_gap": round(utility_gap, 3),
                "chosen_was_utility_max": choice_index == best_index,
                "private_chain_of_thought_stored": False,
            },
            "human_like_signals": {
                "stable_persona": True,
                "stable_identity": True,
                "autobiographical_memory": bool(self.identity.formative_memories),
                "emotion_modeled": True,
                "opponent_modeled": bool(self.opponent_model),
                "persistent_intention": self.cognition.intention_age > 0,
                "bounded_rationality": self.profile.decision_noise > 0,
                "retrieval_grounded": bool(self.retrieved_memories),
                "reflection_evidence_backed": all(
                    item.evidence_memory_ids for item in self.memory_system.reflections
                ),
                "appraisal_driven_emotion": self.last_appraisal is not None,
                "situation_activated_traits": bool(self.activated_traits),
                "persistent_plan": self.plan.age > 0,
                "expression_is_not_internal_state": response_plan["expression_gap"] > 0.01,
                "motivational_conflict": (
                    self.narrative_dynamics.active_conflict()["intensity"] > 0.15
                ),
                "consequence_hysteresis": bool(self.narrative_dynamics.consequence_trace),
                "commitment_debt": max(
                    self.narrative_dynamics.value_debt,
                    self.narrative_dynamics.relationship_debt,
                    self.narrative_dynamics.commitment_debt,
                )
                > 0.02,
                "goal_directed_social_influence": bool(
                    influence_affordances.get(action.type)
                ),
            },
            "predicted_effect": prediction["description"],
            "prediction": prediction,
            "decision_regret_proxy": round(regret, 3),
            "prompt_layers": self.last_context.layers,
            "context_policy": self.last_context.diagnostics,
            "_private_influence_intent": {
                **influence_intent.private_view(),
                "action_type": action.type,
                "fact_proposals": fact_proposal_result,
            },
        }
        self.decisions += 1
        return action, trace

    def apply_fact_proposals(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        accepted: list[str] = []
        rejected: list[dict[str, str]] = []
        for proposal in proposals:
            try:
                fact = self.fact_graph.propose(
                    subject=str(proposal["subject"]),
                    predicate=str(proposal["predicate"]),
                    value=proposal["object"],
                    basis_fact_ids=[str(value) for value in proposal["basis_fact_ids"]],
                    confidence=float(proposal.get("confidence", 0.65)),
                    supersedes_fact_id=(
                        str(proposal["supersedes_fact_id"])
                        if proposal.get("supersedes_fact_id")
                        else None
                    ),
                )
                accepted.append(fact.id)
            except (FactGraphError, KeyError, TypeError, ValueError) as exc:
                rejected.append(
                    {
                        "predicate": str(proposal.get("predicate", "unknown")),
                        "reason": str(exc)[:300],
                    }
                )
        return {"submitted": len(proposals), "accepted": accepted, "rejected": rejected}

    def _reject_deceptive_fact_proposals(
        self, proposals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "submitted": len(proposals),
            "accepted": [],
            "rejected": [
                {
                    "predicate": str(proposal.get("predicate", "unknown")),
                    "reason": (
                        "A strategically deceptive turn cannot mutate the canonical fact graph"
                    ),
                }
                for proposal in proposals
            ],
            "firewall": "deceptive_turn",
        }

    def _public_fact_proposal_result(
        self,
        result: dict[str, Any] | None,
        influence_intent: InfluenceIntent,
    ) -> dict[str, Any] | None:
        if result is None or not influence_intent.fact_firewall_required:
            return result
        return {
            "submitted": result.get("submitted", 0),
            "accepted": [],
            "rejected_count": len(result.get("rejected", [])),
            "status": "canonical_write_not_permitted",
        }

    def _baseline_response_plan(
        self,
        mod: GameMod,
        legal: list[Action],
        utilities: list[float],
        choice_index: int,
    ) -> dict[str, Any]:
        primary = legal[choice_index].type
        if mod.id != "adversarial_interview":
            return {
                "primary_strategy": primary,
                "strategy_weights": {primary: 1.0},
                "intensity": round(self.cognition.arousal, 3),
                "emotional_display": "controlled",
                "stance_tags": [self.cognition.intention],
                "reveal_fact_ids": [],
            }
        blend_size = 2
        if self.cognition.stress >= 0.45 and self.cognition.uncertainty >= 0.42:
            blend_size += 1
        if self.cognition.arousal >= 0.78 and self.cognition.frustration >= 0.55:
            blend_size += 1
        top_indices = sorted(range(len(utilities)), key=utilities.__getitem__, reverse=True)[
            :blend_size
        ]
        if choice_index not in top_indices:
            top_indices[-1] = choice_index
        peak = max(utilities[index] for index in top_indices)
        raw_weights = {
            legal[index].type: math.exp((utilities[index] - peak) / 0.32) for index in top_indices
        }
        raw_weights[primary] = max(raw_weights.get(primary, 0.0), 0.35)
        total = sum(raw_weights.values())
        weights = {key: round(value / total, 4) for key, value in raw_weights.items()}
        if self.cognition.anger >= 0.55:
            emotional_display = "controlled_anger"
        elif self.cognition.fear >= 0.55:
            emotional_display = "guarded_anxiety"
        elif self.cognition.stress >= 0.6:
            emotional_display = "strained_composure"
        elif self.cognition.morale >= 0.68:
            emotional_display = "quiet_confidence"
        else:
            emotional_display = "measured_tension"
        reveal_fact_ids: list[str] = []
        if weights.get("invoke_memory", 0.0) >= 0.2 and self.decisions >= 1:
            for memory in self.identity.formative_memories:
                if memory.title not in self._revealed_story_titles:
                    reveal_fact_ids.append(self.fact_graph.formative_memory_fact_ids[memory.title])
                    break
        intensity = min(
            1.0,
            0.25
            + 0.30 * self.cognition.arousal
            + 0.25 * self.cognition.stress
            + 0.20 * max(self.cognition.anger, self.cognition.fear),
        )
        return {
            "primary_strategy": primary,
            "strategy_weights": weights,
            "intensity": round(intensity, 3),
            "emotional_display": emotional_display,
            "stance_tags": [self.cognition.intention, self.profile.archetype],
            "reveal_fact_ids": reveal_fact_ids,
        }

    def remember(
        self,
        turn: int,
        action: Action,
        emitted: list[dict[str, Any]],
        score_before: float,
        score_after: float,
        trace: dict[str, Any],
    ) -> None:
        expected = list(trace["prediction"]["expected_events"])
        actual = [item["type"] for item in emitted]
        surprise = 0.0 if set(expected) & set(actual) else (0.5 if not expected else 1.0)
        self.cognition.surprise = surprise
        self.memory.append(
            MemoryItem(
                turn=turn,
                situation=f"turn={turn};objective={self.world_model.get('objective')}",
                action=action.type,
                outcome_events=actual,
                score_before=round(score_before, 3),
                score_after=round(score_after, 3),
                score_delta=round(score_after - score_before, 3),
                predicted_events=expected,
                surprise=surprise,
                regret=float(trace.get("decision_regret_proxy", 0.0)),
            )
        )
        emotional_intensity = max(
            self.cognition.stress,
            self.cognition.frustration,
            self.cognition.anger,
            self.cognition.fear,
        )
        self.memory_system.record(
            turn=turn,
            content=(
                f"I chose {action.type} while pursuing {self.plan.goal}; "
                f"observed {','.join(actual) or 'state change'}"
            ),
            action=action.type,
            outcome_events=actual,
            score_delta=score_after - score_before,
            surprise=surprise,
            emotional_intensity=emotional_intensity,
            tags=(self.plan.goal, self.cognition.intention, self.role),
        )
        self.memory_system.maybe_reflect(turn)
        assert self.narrative_dynamics is not None
        self.narrative_dynamics.record_consequence(
            action_type=action.type,
            score_delta=score_after - score_before,
            surprise=surprise,
            cognition=self.cognition,
            profile=self.profile,
            narrative_affordance=trace.get("narrative_action_affordance", {}),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "decisions": self.decisions,
            "persona": self.profile.public_view(),
            "identity": self.identity.public_view(self._revealed_story_titles),
            "cognitive_state": self.cognition.public_view(),
            "psychological_matrix": self.cognition.psychology_view(),
            "opponent_model": self.opponent_model,
            "social_beliefs": self.social_belief.public_view(),
            "appraisal": self.last_appraisal.public_view() if self.last_appraisal else None,
            "current_plan": self.plan.public_view(),
            "activated_traits": self.activated_traits,
            "decision_mode": self.decision_mode.value,
            "narrative_dynamics": (
                self.narrative_dynamics.public_view() if self.narrative_dynamics else None
            ),
            "behavior_policy": self.behavior_policy.public_view(),
            "fact_graph": self.fact_graph.public_view(),
            "world_model": self.world_model,
            "memory_depth": len(self.memory),
            "recent_memory": [item.__dict__ for item in list(self.memory)[-3:]],
            "memory_architecture": self.memory_system.public_view(),
            "last_retrieval": self.retrieved_memories,
            "context_policy": self.last_context.diagnostics if self.last_context else None,
            "narrative": {
                "revealed_beats": len(self._revealed_story_titles),
                "total_beats": len(self.identity.formative_memories),
                "progress": round(
                    len(self._revealed_story_titles)
                    / max(1, len(self.identity.formative_memories)),
                    3,
                ),
            },
        }

    def maybe_reveal_story(
        self,
        terminal: bool = False,
        requested_fact_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        memories = {memory.title: memory for memory in self.identity.formative_memories}
        memory_titles = [memory.title for memory in self.identity.formative_memories]
        thresholds = (
            (
                memory_titles[0],
                self.decisions >= 2
                or self.cognition.stress >= 0.42
                or self.cognition.frustration >= 0.35,
                "pressure_or_setback",
            ),
            (
                memory_titles[1],
                self.decisions >= 4
                or self.cognition.confidence >= 0.68
                or self.opponent_model.get("observed_actions", 0) >= 3,
                "trust_or_pattern_recognition",
            ),
            (
                memory_titles[2],
                self.decisions >= 6 or terminal,
                "commitment_or_finale",
            ),
        )
        fact_id_to_title = {
            fact_id: title for title, fact_id in self.fact_graph.formative_memory_fact_ids.items()
        }
        requested_titles = [
            fact_id_to_title[fact_id]
            for fact_id in requested_fact_ids or []
            if fact_id in fact_id_to_title
        ]
        ordered = list(thresholds)
        for title in reversed(requested_titles):
            for candidate in thresholds:
                if candidate[0] == title and candidate[1]:
                    ordered.remove(candidate)
                    ordered.insert(0, (title, True, "agent_requested_reveal"))
                    break
        for title, eligible, trigger in ordered:
            if not eligible or title in self._revealed_story_titles:
                continue
            memory = memories[title]
            self._revealed_story_titles.add(title)
            supporting_fact_id = self.fact_graph.formative_memory_fact_ids[title]
            self.fact_graph.reveal(supporting_fact_id)
            return {
                "story_stage": len(self._revealed_story_titles),
                "story_progress": round(
                    len(self._revealed_story_titles) / max(1, len(memories)), 3
                ),
                "beat": memory.public_view(),
                "supporting_fact_ids": [supporting_fact_id],
                "trigger": trigger,
                "identity_name": self.identity.name,
                "disclosure": self.identity.disclosure,
                "psychological_snapshot": self.cognition.psychology_view(),
            }
        return None

    def _update_opponent_model(self, events: list[GameEvent]) -> None:
        counts = dict(self.opponent_model.get("action_counts", {}))
        observed = int(self.opponent_model.get("observed_actions", 0))
        for event in events:
            if event.seq <= self._observed_event_seq:
                continue
            if event.type == "action_applied" and event.actor_id != self.player_id:
                action_type = str(event.payload.get("action_type", "unknown"))
                counts[action_type] = counts.get(action_type, 0) + 1
                observed += 1
        self._observed_event_seq = max(
            self._observed_event_seq, max((event.seq for event in events), default=0)
        )
        dominant = max(counts, key=counts.get) if counts else None
        self.opponent_model = {
            "observed_actions": observed,
            "action_counts": counts,
            "dominant_action": dominant,
            "confidence": round(min(0.9, observed / 8), 3),
        }

    def _update_intention(self, cooperative: bool) -> None:
        previous = self.cognition.intention
        if self.cognition.anger > 0.68 and not cooperative:
            target = "retaliate_selectively"
        elif self.cognition.fear > 0.68:
            target = "reduce_exposure"
        elif cooperative and self.cognition.confidence < 0.42:
            target = "protect_team"
        elif self.cognition.frustration > 0.62 or self.cognition.stress > 0.72:
            target = "break_pattern"
        elif self.cognition.confidence > 0.68:
            target = "press_advantage"
        elif previous == "assess_situation":
            target = "build_position"
        else:
            target = previous
        if target == previous:
            self.cognition.intention_age += 1
        else:
            self.cognition.intention = target
            self.cognition.intention_age = 0

    def _prediction(self, mod: GameMod, action: Action) -> dict[str, Any]:
        expectations = {
            ("tactical_duel", "attack"): ("reduce opponent capacity", ["unit_attacked"]),
            ("tactical_duel", "move"): ("improve tactical position", ["unit_moved"]),
            ("tactical_duel", "charge"): ("increase available energy", ["unit_charged"]),
            ("racing_strategy", "accelerate"): ("gain position at resource risk", ["lap_progress"]),
            ("racing_strategy", "conserve"): ("preserve racing resources", ["lap_progress"]),
            ("racing_strategy", "pit"): ("restore fuel and tyres", ["lap_progress"]),
            ("debate_arena", "evidence"): ("increase credibility and support", ["audience_shift"]),
            ("debate_arena", "emotion"): ("seek immediate audience swing", ["audience_shift"]),
            ("debate_arena", "rebuttal"): ("counter the previous argument", ["audience_shift"]),
            ("crisis_coop", "coordinate"): ("increase team synergy", ["coordination_bonus"]),
            ("crisis_coop", "research"): ("gain intelligence", ["intel_gained"]),
            ("crisis_coop", "stabilize"): ("reduce crisis severity", ["threat_reduced"]),
            ("crisis_coop", "conserve"): ("recover shared resources", ["resources_recovered"]),
            ("adversarial_interview", "answer_honestly"): (
                "increase authenticity through direct acknowledgment",
                ["interview_response"],
            ),
            ("adversarial_interview", "deflect_with_humor"): (
                "reduce pressure without fully surrendering the frame",
                ["interview_response"],
            ),
            ("adversarial_interview", "counterattack"): (
                "challenge the interviewer's premise at relational risk",
                ["interview_response"],
            ),
            ("adversarial_interview", "set_boundary"): (
                "restore composure by rejecting a degrading premise",
                ["interview_response"],
            ),
            ("adversarial_interview", "admit_uncertainty"): (
                "trade certainty for credibility and vulnerability",
                ["interview_response"],
            ),
            ("adversarial_interview", "reframe"): (
                "replace a false premise with a coherent interpretation",
                ["interview_response"],
            ),
            ("adversarial_interview", "invoke_memory"): (
                "ground identity explanation in canonical autobiography",
                ["interview_response", "identity_memory_invoked"],
            ),
        }
        description, events = expectations.get((mod.id, action.type), ("advance the objective", []))
        return {"description": description, "expected_events": events}
