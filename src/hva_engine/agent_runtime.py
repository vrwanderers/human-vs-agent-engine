from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from random import Random
from time import perf_counter
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
from hva_engine.relationship_memory import RelationshipMemory, RelationshipProfile
from hva_engine.skill_learning import SkillLearningSystem
from hva_engine.social_influence import (
    InfluenceIntent,
    constrain_model_intent,
    derive_influence_intent,
    strategic_utility_bias,
)
from hva_engine.speech_style import SpeechStyleRealizer
from hva_engine.stimulus import (
    DeliberationDecision,
    FastAppraisal,
    RealityStatusGuard,
    ReflexResponse,
    StimulusFrame,
    StimulusPipeline,
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
    expires_turn: int


@dataclass
class AgentBrain:
    """Inspectable human-like brain without exposing private chain-of-thought."""

    player_id: str
    role: str
    profile: CognitiveProfile
    identity: AgentIdentity
    behavior_policy: RuntimeBehaviorPolicy
    fact_graph: AgentFactGraph
    participant_directory: dict[str, dict[str, str]] = field(default_factory=dict)
    memory_limit: int = 12
    short_term_ttl_turns: int = 6
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
    last_decision_tendencies: dict[str, Any] = field(default_factory=dict)
    last_outcome_reappraisal: dict[str, Any] = field(default_factory=dict)
    last_story_reveal_diagnostic: dict[str, Any] = field(default_factory=dict)
    stimulus_pipeline: StimulusPipeline = field(default_factory=StimulusPipeline)
    last_stimulus_frame: StimulusFrame = field(default_factory=StimulusFrame)
    last_fast_appraisal: FastAppraisal = field(default_factory=FastAppraisal)
    last_reflex_response: ReflexResponse = field(default_factory=ReflexResponse)
    last_deliberation_gate: DeliberationDecision | None = None
    skill_learning: SkillLearningSystem | None = None
    last_skill_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_selected_skill: dict[str, Any] = field(default_factory=dict)
    last_skill_update: dict[str, Any] = field(default_factory=dict)
    last_plan_revised: bool = False
    narrative_dynamics: NarrativeDynamics | None = None
    relationship_memory: RelationshipMemory | None = None
    relationship_profiles: dict[str, RelationshipProfile] = field(default_factory=dict)
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
        if self.relationship_memory is None:
            self.relationship_memory = RelationshipMemory(
                self.memory_system.owner_id, self.memory_system.store
            )
        if self.skill_learning is None:
            self.skill_learning = SkillLearningSystem(
                self.memory_system.owner_id, self.memory_system.store
            )
        self.memory_system.seed_identity_memories(
            self.identity,
            formative_fact_ids=self.fact_graph.formative_memory_fact_ids,
            lived_fact_ids=self.fact_graph.lived_memory_fact_ids,
        )

    def _forget_expired_short_term(self, current_turn: int) -> None:
        while self.memory and self.memory[0].expires_turn < current_turn:
            self.memory.popleft()

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
        self._forget_expired_short_term(turn)
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
        memory_query_cues = self._event_memory_cues(new_external_events)
        hostile_severity = max(
            (
                float(event.payload.get("severity", 0.0))
                for event in new_external_events
                if event.type == "interview_question"
            ),
            default=0.0,
        )
        self._ensure_relationship_profiles()
        identity_themes = {
            theme
            for memory in (*self.identity.formative_memories, *self.identity.lived_memories)
            for theme in memory.themes
        }
        identity_themes.update(self.identity.values)
        sensitive_topics = {
            point.topic
            for profile in self.relationship_profiles.values()
            for point in profile.sensitive_points.values()
        }
        self.last_stimulus_frame, self.last_fast_appraisal = self.stimulus_pipeline.perceive(
            events,
            viewer_id=self.player_id,
            after_sequence=self._observed_event_seq,
            identity_themes=identity_themes,
            sensitive_topics=sensitive_topics,
            stress=self.cognition.stress,
            fear=self.cognition.fear,
            uncertainty=self.cognition.uncertainty,
        )
        self.cognition.apply_adjustments(
            {
                "stress": 0.07 * self.last_fast_appraisal.threat
                + 0.05 * self.last_fast_appraisal.social_threat,
                "fear": 0.08 * self.last_fast_appraisal.threat,
                "arousal": 0.10 * self.last_fast_appraisal.action_readiness,
                "uncertainty": 0.05 * self.last_fast_appraisal.ambiguity,
            }
        )
        active_relationship_actor = next(
            (
                event.actor_id
                for event in reversed(new_external_events)
                if event.actor_id in self.relationship_profiles
            ),
            next(iter(self.relationship_profiles), None),
        )
        if active_relationship_actor is not None and self.decisions == 0:
            self._sync_social_belief(
                self.relationship_profiles[active_relationship_actor], preserve_intent=True
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
        self._update_opponent_model(events)
        self._remember_relationship_events(
            new_external_events,
            current_turn=turn,
            cooperative="coop" in self.role,
        )
        if active_relationship_actor is not None:
            self._sync_social_belief(self.relationship_profiles[active_relationship_actor])
        assert self.narrative_dynamics is not None
        self.narrative_dynamics.update_before_decision(
            self.last_appraisal,
            self.cognition,
            self.social_belief.trust,
        )
        self._update_intention("coop" in self.role)
        self.last_plan_revised = self.plan.update(
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
        if self.relationship_profiles:
            self.fact_graph.upsert_runtime(
                "belief.relationship_impression",
                {"profiles": self._relationship_fact_summaries()},
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
            subject="self",
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
            "memory_query_cues": memory_query_cues,
            "objective": "shared_success" if "coop" in self.role else "outperform_opponents",
            "memory_depth": len(self.memory),
            "shared_fact_count": len(shared_facts or []),
            "belief_uncertainty": round(self.cognition.uncertainty, 3),
            "opponent_model": self.opponent_model,
            "social_belief": self.social_belief.public_view(),
            "relationship_memory": self._relationship_views(),
            "appraisal": self.last_appraisal.public_view(),
            "stimulus_frame": self.last_stimulus_frame.public_view(),
            "fast_appraisal": self.last_fast_appraisal.public_view(),
            "reality_boundary": RealityStatusGuard.context_contract(),
            "plan": self._plan_context(),
            "activated_traits": self.activated_traits,
            "decision_mode": self.decision_mode.value,
            "narrative_dynamics": self.narrative_dynamics.public_view(),
            "mod_psychological_signals": {
                key: round(value, 3) for key, value in mod_psychology.items()
            },
            "environment": mod.agent_world_model(state, self.player_id),
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
                *(self.world_model.get("memory_query_cues", [])),
                *(
                    profile.attitude
                    for profile in self.relationship_profiles.values()
                ),
                *(
                    point.topic
                    for profile in self.relationship_profiles.values()
                    for point in profile.sensitive_points.values()
                ),
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
        self.last_decision_tendencies = self._decision_tendencies(
            legal, utilities, components
        )
        assert self.skill_learning is not None
        self.last_skill_candidates = []
        for action_index, candidate_action in enumerate(legal):
            readiness = self.skill_learning.readiness(
                mod.agent_skill_id(candidate_action),
                mod.agent_skill_context(state, self.player_id, candidate_action),
                current_turn=current_turn,
            )
            self.last_skill_candidates.append(
                {
                    "action_index": action_index,
                    "action_type": candidate_action.type,
                    "preferred_by_local_policy": candidate_action == baseline_action,
                    **readiness.public_view(),
                }
            )
        reflex_seed = int(
            sha256(
                (
                    f"{self._match_id}:{self.player_id}:{self.decisions}:"
                    f"{self.last_stimulus_frame.sequence_end}"
                ).encode()
            ).hexdigest()[:16],
            16,
        )
        self.last_reflex_response = self.stimulus_pipeline.reflex_controller.respond(
            self.last_stimulus_frame,
            self.last_fast_appraisal,
            turn=current_turn,
            conscientiousness=self.profile.conscientiousness,
            agreeableness=self.profile.agreeableness,
            neuroticism=self.profile.neuroticism,
            stress=self.cognition.stress,
            rng=Random(reflex_seed),
        )
        self.last_deliberation_gate = self.stimulus_pipeline.deliberation_gate.evaluate(
            provider_available=decision_client is not None,
            frame=self.last_stimulus_frame,
            appraisal=self.last_fast_appraisal,
            decision_mode=self.decision_mode.value,
            legal_action_types=(action.type for action in legal),
            procedural_values=averages,
            previous_action=last_action,
            text_interaction=(
                "text_state" in mod.capabilities
                or "text" in mod.tags
                or "interview" in mod.tags
                or "debate" in mod.tags
            ),
            plan_revised=self.last_plan_revised,
            skill_candidates=self.last_skill_candidates,
        )
        skill_context = [
            {
                key: value
                for key, value in candidate.items()
                if key
                in {
                    "action_index",
                    "skill_id",
                    "stage",
                    "confidence",
                    "context_attempts",
                    "automatic",
                }
            }
            for candidate in self.last_skill_candidates
        ]
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
            current_plan=self._plan_context(),
            social_beliefs=self.social_belief.public_view(),
            relationship_profiles=self._relationship_views(),
            activated_traits=self.activated_traits,
            decision_mode=self.decision_mode.value,
            narrative_dynamics=narrative_context,
            influence_affordances=influence_affordances,
            decision_tendencies=self.last_decision_tendencies,
            implicit_control={
                "stimulus_frame": self.last_stimulus_frame.public_view(),
                "fast_appraisal": self.last_fast_appraisal.private_view(),
                "involuntary_response": self.last_reflex_response.observable_view(),
                "deliberation_gate": self.last_deliberation_gate.public_view(),
                "procedural_skill_learning": {
                    "candidate_actions": skill_context,
                    "policy": self.skill_learning.public_view(
                        current_turn=current_turn
                    )["automaticity_policy"],
                    "new_context_requires_guidance": True,
                },
                "reality_boundary": RealityStatusGuard.context_contract(),
            },
        )
        llm_decision = None
        llm_error: dict[str, str] | None = None
        fact_proposal_result: dict[str, Any] | None = None
        influence_intent: InfluenceIntent | None = None
        if decision_client is not None and self.last_deliberation_gate.should_deliberate:
            llm_started = perf_counter()
            try:
                llm_decision = decision_client.choose_structured_sync(
                    self.last_context.messages,
                    [action.model_dump() for action in legal],
                    context_metadata=self.last_context.provider_metadata(),
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
                llm_error = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "provider": decision_client.provider.name,
                    "latency_ms": round((perf_counter() - llm_started) * 1_000, 3),
                    "context_id": self.last_context.context_id,
                    "context_sha256": self.last_context.content_sha256,
                    "status": "fallback",
                }
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
            automatic_indices = self.last_deliberation_gate.automatic_action_indices
            eligible_indices = (
                automatic_indices
                if decision_client is not None and automatic_indices
                else tuple(range(len(legal)))
            )
            eligible_utilities = [utilities[index] for index in eligible_indices]
            eligible_choice = bounded_choice(
                eligible_utilities,
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
            choice_index = eligible_indices[eligible_choice]
        action = legal[choice_index]
        self.last_selected_skill = dict(self.last_skill_candidates[choice_index])
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
            "involuntary_response": self.last_reflex_response.observable_view(),
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
        baseline_utterance = None
        speech_style_diagnostic: dict[str, Any] = {
            "applied": False,
            "source": "provider" if llm_decision is not None else "no_semantic_utterance",
        }
        provider_utterance = llm_decision.utterance if llm_decision is not None else None
        if (
            provider_utterance
            and self.identity.speech_style.get("constraint_source") != "sampled_archetype"
        ):
            provider_utterance, speech_style_diagnostic = SpeechStyleRealizer().realize(
                provider_utterance,
                self.identity.speech_style,
                mature_fiction=self.behavior_policy.content_mode.value == "mature_fiction",
            )
            speech_style_diagnostic["source"] = "provider_then_style_enforcer"
        if llm_decision is None:
            semantic_utterance = mod.agent_utterance(state, self.player_id, action)
            if semantic_utterance:
                baseline_utterance, speech_style_diagnostic = SpeechStyleRealizer().realize(
                    semantic_utterance,
                    self.identity.speech_style,
                    mature_fiction=self.behavior_policy.content_mode.value
                    == "mature_fiction",
                )
                speech_style_diagnostic["source"] = "engine_baseline_realizer"
        trace = {
            "policy": (
                f"llm:{decision_client.provider.name}"
                if llm_decision is not None and decision_client is not None
                else f"{mod.id}.baseline"
            ),
            "decision_source": "llm" if llm_decision is not None else "baseline",
            "decision_path": (
                "llm_deliberation"
                if llm_decision is not None
                else "provider_fallback"
                if llm_error is not None
                else "reflex_or_routine"
                if decision_client is not None
                else "local_baseline"
            ),
            "rationale": llm_decision.reason
            if llm_decision and llm_decision.reason
            else default_rationale,
            "utterance": provider_utterance if llm_decision else baseline_utterance,
            "speech_style": {
                "profile": self.identity.speech_style,
                "diagnostic": speech_style_diagnostic,
                "content_and_action_independent": True,
            },
            "response_plan": response_plan,
            "llm": (
                {
                    "provider": decision_client.provider.name,
                    "model": llm_decision.response.model,
                    "usage": llm_decision.response.usage,
                    "telemetry": llm_decision.response.telemetry,
                    "context_id": self.last_context.context_id,
                    "context_sha256": self.last_context.content_sha256,
                    "context_owner_agent_id": self.last_context.owner_agent_id,
                    "fact_proposals": self._public_fact_proposal_result(
                        fact_proposal_result, influence_intent
                    ),
                }
                if llm_decision is not None and decision_client is not None
                else None
            ),
            "llm_error": llm_error,
            "llm_skip": (
                {
                    **self.last_deliberation_gate.public_view(),
                    "provider": decision_client.provider.name,
                }
                if decision_client is not None
                and not self.last_deliberation_gate.should_deliberate
                else None
            ),
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
            "relationship_memory": self._relationship_views(),
            "appraisal": self.last_appraisal.public_view() if self.last_appraisal else {},
            "stimulus_frame": self.last_stimulus_frame.public_view(),
            "fast_appraisal": self.last_fast_appraisal.public_view(),
            "involuntary_response": self.last_reflex_response.observable_view(),
            "deliberation_gate": self.last_deliberation_gate.public_view(),
            "skill_learning": self.skill_learning.public_view(current_turn=current_turn),
            "skill_candidates": self.last_skill_candidates,
            "selected_skill": self.last_selected_skill,
            "current_plan": self._plan_context(),
            "decision_tendencies": self.last_decision_tendencies,
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
                "relationship_specific_memory": bool(self.relationship_profiles),
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
                "multimodal_stimulus_grounding": self.last_stimulus_frame.has_input,
                "pre_deliberative_appraisal": self.last_stimulus_frame.has_input,
                "involuntary_expression": bool(self.last_reflex_response.cues),
                "provider_call_gated": decision_client is not None,
                "dynamic_skill_automaticity": True,
                "selected_skill_automatic": bool(
                    self.last_selected_skill.get("automatic")
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
            "_private_reflex": {
                "stimulus_frame": self.last_stimulus_frame.private_view(),
                "fast_appraisal": self.last_fast_appraisal.private_view(),
                "response": self.last_reflex_response.private_view(),
                "reality_boundary": RealityStatusGuard.context_contract(),
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
        self._forget_expired_short_term(turn)
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
                expires_turn=turn + self.short_term_ttl_turns,
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
        selected_skill = trace.get("selected_skill", {})
        if isinstance(selected_skill, dict) and selected_skill.get("skill_id"):
            score_delta = score_after - score_before
            skill_success = surprise < 0.6 and score_delta >= -0.05
            skill_readiness = self.skill_learning.record_execution(
                str(selected_skill["skill_id"]),
                dict(selected_skill.get("context", {}).get("descriptor", {})),
                turn=turn,
                success=skill_success,
                surprise=surprise,
                guided=trace.get("decision_path") == "llm_deliberation",
                automatic=(
                    trace.get("decision_path") == "reflex_or_routine"
                    and bool(selected_skill.get("automatic"))
                ),
            )
            self.last_skill_update = {
                **skill_readiness.public_view(),
                "success": skill_success,
                "surprise": round(surprise, 3),
                "control_source": (
                    "guided_llm"
                    if trace.get("decision_path") == "llm_deliberation"
                    else "automatic_skill"
                    if trace.get("decision_path") == "reflex_or_routine"
                    and selected_skill.get("automatic")
                    else "local_practice"
                ),
            }
        else:
            self.last_skill_update = {}
        assert self.narrative_dynamics is not None
        self.narrative_dynamics.record_consequence(
            action_type=action.type,
            score_delta=score_after - score_before,
            surprise=surprise,
            cognition=self.cognition,
            profile=self.profile,
            narrative_affordance=trace.get("narrative_action_affordance", {}),
        )
        self._integrate_outcome(
            action_type=action.type,
            score_delta=score_after - score_before,
            response_plan=trace.get("response_plan", {}),
        )

    def _integrate_outcome(
        self,
        *,
        action_type: str,
        score_delta: float,
        response_plan: dict[str, Any],
    ) -> None:
        before = self.cognition.psychology_view()
        raw_weights = response_plan.get("strategy_weights", {})
        weights = raw_weights if isinstance(raw_weights, dict) else {}

        def weight(name: str) -> float:
            try:
                return max(0.0, min(1.0, float(weights.get(name, 0.0))))
            except (TypeError, ValueError):
                return 0.0

        direct_disclosure = min(
            1.0,
            weight("answer_honestly")
            + weight("admit_uncertainty")
            + 0.65 * weight("invoke_memory"),
        )
        boundary = weight("set_boundary")
        counterattack = weight("counterattack")
        outcome = max(-1.0, min(1.0, score_delta))
        relief = max(0.0, outcome)
        strain = max(0.0, -outcome)
        adjustments = {
            "confidence": (
                0.10 * outcome
                + 0.025 * boundary
                - 0.02 * weight("admit_uncertainty")
            ),
            "morale": 0.08 * outcome + 0.02 * direct_disclosure,
            "stress": (
                -0.10 * relief
                + 0.12 * strain
                - 0.06 * direct_disclosure
                + 0.025 * counterattack
            ),
            "frustration": -0.08 * relief + 0.10 * strain,
            "anger": (
                -0.08 * weight("answer_honestly")
                - 0.05 * weight("admit_uncertainty")
                - 0.04 * weight("invoke_memory")
                + 0.055 * counterattack
                + 0.08 * strain
            ),
            "fear": -0.05 * boundary - 0.025 * direct_disclosure + 0.06 * strain,
            "arousal": -0.04 * direct_disclosure + 0.05 * counterattack,
            "uncertainty": -0.05 * relief + 0.025 * weight("admit_uncertainty"),
        }
        self.cognition.apply_adjustments(adjustments)
        self.last_outcome_reappraisal = {
            "action_type": action_type,
            "score_delta": round(score_delta, 3),
            "drivers": {
                "direct_disclosure": round(direct_disclosure, 3),
                "boundary": round(boundary, 3),
                "counterattack": round(counterattack, 3),
                "relief": round(relief, 3),
                "strain": round(strain, 3),
            },
            "adjustments": {
                key: round(value, 3) for key, value in adjustments.items()
            },
            "before": before,
            "after": self.cognition.psychology_view(),
        }

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
            "relationship_memory": self._relationship_views(),
            "appraisal": self.last_appraisal.public_view() if self.last_appraisal else None,
            "stimulus_frame": self.last_stimulus_frame.private_view(),
            "fast_appraisal": self.last_fast_appraisal.private_view(),
            "involuntary_response": self.last_reflex_response.private_view(),
            "deliberation_gate": (
                self.last_deliberation_gate.public_view()
                if self.last_deliberation_gate
                else None
            ),
            "current_plan": self._plan_context(),
            "decision_tendencies": self.last_decision_tendencies,
            "last_outcome_reappraisal": self.last_outcome_reappraisal,
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
            "skill_learning": (
                self.skill_learning.public_view(
                    current_turn=int(self.world_model.get("turn", self.decisions))
                )
                if self.skill_learning
                else None
            ),
            "last_skill_candidates": self.last_skill_candidates,
            "last_skill_update": self.last_skill_update,
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

    def public_summary(self) -> dict[str, Any]:
        """Only information that a spectator or opponent may observe."""

        return {
            "role": self.role,
            "decisions": self.decisions,
            "identity": self.identity.public_view(self._revealed_story_titles),
            "fact_graph": self.fact_graph.public_view(),
            "skill_learning": (
                self.skill_learning.public_view(current_turn=self.decisions)
                if self.skill_learning
                else None
            ),
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
                self.cognition.stress >= 0.42
                or self.cognition.frustration >= 0.35,
                "pressure_or_setback",
            ),
            (
                memory_titles[1],
                bool(self._revealed_story_titles)
                and (
                    self.cognition.confidence >= 0.68
                    or self.opponent_model.get("observed_actions", 0) >= 3
                ),
                "trust_or_pattern_recognition",
            ),
            (
                memory_titles[2],
                terminal and len(self._revealed_story_titles) >= 2,
                "commitment_or_finale",
            ),
        )
        fact_id_to_title = {
            fact_id: title for title, fact_id in self.fact_graph.formative_memory_fact_ids.items()
        }
        eligibility = {
            title: {"eligible": eligible, "threshold": trigger}
            for title, eligible, trigger in thresholds
        }
        requested_statuses: list[dict[str, Any]] = []
        for fact_id in requested_fact_ids or []:
            title = fact_id_to_title.get(fact_id)
            if title is None:
                requested_statuses.append(
                    {
                        "fact_id": fact_id,
                        "status": "rejected_unknown_or_non_formative_fact",
                    }
                )
                continue
            if title in self._revealed_story_titles:
                status = "already_revealed"
            elif eligibility[title]["eligible"]:
                status = "eligible"
            else:
                status = "deferred_by_story_pacing"
            requested_statuses.append(
                {
                    "fact_id": fact_id,
                    "title": title,
                    "status": status,
                    "eligibility_threshold": eligibility[title]["threshold"],
                    "decision_count": self.decisions,
                    "terminal": terminal,
                }
            )
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
            reveal = {
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
            self.last_story_reveal_diagnostic = {
                "outcome": "revealed",
                "revealed_fact_id": supporting_fact_id,
                "revealed_title": title,
                "trigger": trigger,
                "requested": requested_statuses,
            }
            return reveal
        self.last_story_reveal_diagnostic = {
            "outcome": (
                "deferred_or_rejected"
                if requested_statuses
                else "no_eligible_unrevealed_story"
            ),
            "requested": requested_statuses,
            "decision_count": self.decisions,
            "terminal": terminal,
        }
        return None

    def _update_opponent_model(self, events: list[GameEvent]) -> None:
        counts = dict(self.opponent_model.get("action_counts", {}))
        observed = int(self.opponent_model.get("observed_actions", 0))
        by_actor = {
            actor_id: dict(value)
            for actor_id, value in dict(self.opponent_model.get("by_actor", {})).items()
        }
        if not counts:
            for actor_id, profile in self.relationship_profiles.items():
                actor_counts = dict(profile.action_counts)
                if actor_counts:
                    by_actor[actor_id] = {
                        "observed_actions": sum(actor_counts.values()),
                        "action_counts": actor_counts,
                        "dominant_action": max(actor_counts, key=actor_counts.get),
                    }
                for action, count in actor_counts.items():
                    counts[action] = counts.get(action, 0) + count
                    observed += count
        for event in events:
            if event.seq <= self._observed_event_seq:
                continue
            if event.type == "action_applied" and event.actor_id != self.player_id:
                action_type = str(event.payload.get("action_type", "unknown"))
                counts[action_type] = counts.get(action_type, 0) + 1
                observed += 1
                actor_id = str(event.actor_id)
                actor_model = by_actor.setdefault(
                    actor_id, {"observed_actions": 0, "action_counts": {}}
                )
                actor_counts = dict(actor_model.get("action_counts", {}))
                actor_counts[action_type] = actor_counts.get(action_type, 0) + 1
                actor_model["action_counts"] = actor_counts
                actor_model["observed_actions"] = int(
                    actor_model.get("observed_actions", 0)
                ) + 1
                actor_model["dominant_action"] = max(
                    actor_counts, key=actor_counts.get
                )
                actor_model["confidence"] = round(
                    min(0.95, int(actor_model["observed_actions"]) / 8), 3
                )
        self._observed_event_seq = max(
            self._observed_event_seq, max((event.seq for event in events), default=0)
        )
        dominant = max(counts, key=counts.get) if counts else None
        self.opponent_model = {
            "observed_actions": observed,
            "action_counts": counts,
            "dominant_action": dominant,
            "confidence": round(min(0.9, observed / 8), 3),
            "by_actor": by_actor,
        }

    def _ensure_relationship_profiles(self) -> None:
        assert self.relationship_memory is not None
        for actor_id, participant in self.participant_directory.items():
            if actor_id == self.player_id or actor_id in self.relationship_profiles:
                continue
            self.relationship_profiles[actor_id] = self.relationship_memory.load_or_create(
                participant["memory_key"],
                participant["display_name"],
                participant["kind"],
            )

    def _sync_social_belief(
        self, profile: RelationshipProfile, *, preserve_intent: bool = False
    ) -> None:
        self.social_belief.trust = profile.trust
        self.social_belief.respect = profile.respect
        self.social_belief.familiarity = profile.familiarity
        self.social_belief.perceived_hostility = profile.hostility
        self.social_belief.confidence = min(0.98, profile.familiarity * 0.9)
        if preserve_intent and profile.interaction_count == 0:
            return
        self.social_belief.predicted_intent = {
            "hostile": "escalate",
            "guarded": "test_boundaries",
            "warm": "cooperate",
            "cautiously_positive": "cooperate_or_probe",
        }.get(profile.attitude, "cooperate_or_probe")

    def _relationship_event_content(
        self, event: GameEvent, profile: RelationshipProfile
    ) -> tuple[str, tuple[str, ...], float]:
        payload = event.payload
        if event.type == "interview_question":
            theme = str(payload.get("theme", "unknown"))[:80]
            prompt = " ".join(str(payload.get("prompt", "")).split())[:280]
            severity = max(0.0, min(1.0, float(payload.get("severity", 0.0))))
            return (
                f"{profile.target_label} asked a {theme} question with severity "
                f"{severity:.2f}: {prompt}",
                (theme, "question_style"),
                severity,
            )
        if event.type == "interview_response":
            theme = str(payload.get("theme", "unknown"))[:80]
            strategy = str(payload.get("strategy", "respond"))[:80]
            intensity = max(0.0, min(1.0, float(payload.get("intensity", 0.5))))
            return (
                f"{profile.target_label} responded to {theme} with {strategy} at "
                f"intensity {intensity:.2f}.",
                (theme, strategy, "observed_reaction"),
                intensity,
            )
        if event.type == "story_reveal":
            beat = payload.get("beat", {})
            title = str(beat.get("title", "background")) if isinstance(beat, dict) else "background"
            return (
                f"{profile.target_label} publicly disclosed autobiographical background: {title}.",
                (title[:100], "reported_background"),
                0.75,
            )
        action = str(payload.get("action_type", "unknown"))[:80]
        return (
            f"Observed {profile.target_label} use {action} in the current interaction.",
            (action, "observed_behavior"),
            0.35,
        )

    @staticmethod
    def _event_memory_cues(events: list[GameEvent]) -> list[str]:
        cues: list[str] = []
        for event in events[-6:]:
            for key in (
                "theme",
                "prompt",
                "action_type",
                "strategy",
                "move",
                "headline",
                "title",
                "summary",
                "category",
                "policy_id",
                "incident_id",
            ):
                value = event.payload.get(key)
                if value is not None:
                    cue = " ".join(str(value).split())[:240]
                    if cue and cue not in cues:
                        cues.append(cue)
        return cues[-10:]

    def _remember_relationship_events(
        self,
        events: list[GameEvent],
        *,
        current_turn: int,
        cooperative: bool,
    ) -> None:
        significant = {
            "action_applied",
            "interview_question",
            "interview_response",
            "story_reveal",
        }
        dirty: set[str] = set()
        for event in events:
            if event.type not in significant or event.actor_id not in self.relationship_profiles:
                continue
            actor_id = str(event.actor_id)
            profile = self.relationship_profiles[actor_id]
            content, event_tags, intensity = self._relationship_event_content(event, profile)
            memory = self.memory_system.record(
                turn=current_turn,
                content=content,
                action="",
                outcome_events=[event.type],
                score_delta=-0.2 * intensity if not cooperative else 0.1 * (1 - intensity),
                surprise=max(self.cognition.surprise, 0.25 * intensity),
                emotional_intensity=max(
                    intensity,
                    self.cognition.stress,
                    self.cognition.anger,
                    self.cognition.fear,
                ),
                tags=("relationship", profile.target_label, *event_tags),
                extra_categories=("relationship", "person_model"),
                force_long_term=True,
                track_procedural=False,
                metadata={
                    "source": "observed_interaction",
                    "target_token": profile.target_token,
                    "source_event_seq": event.seq,
                    "epistemic_status": "observed",
                },
            )
            profile.observe(
                event,
                evidence_memory_id=memory.id,
                cooperative=cooperative,
            )
            dirty.add(actor_id)
        assert self.relationship_memory is not None
        for actor_id in dirty:
            self.relationship_memory.persist(self.relationship_profiles[actor_id])

    def _relationship_views(self) -> list[dict[str, Any]]:
        return [
            {
                "actor_id": actor_id,
                **profile.private_view(),
            }
            for actor_id, profile in self.relationship_profiles.items()
        ]

    def _relationship_fact_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "target_actor_id": actor_id,
                "target_label": profile.target_label,
                "attitude": profile.attitude,
                "trust": round(profile.trust, 3),
                "hostility": round(profile.hostility, 3),
                "impressions": [
                    {
                        "statement": item.statement[:180],
                        "confidence": round(item.confidence, 3),
                        "epistemic_status": item.epistemic_status,
                    }
                    for item in list(profile.impressions.values())[-3:]
                ],
                "sensitive_topics": [
                    {
                        "topic": item.topic,
                        "kind": item.kind,
                        "confidence": round(item.confidence, 3),
                    }
                    for item in list(profile.sensitive_points.values())[-3:]
                ],
                "warning": "revisable belief, not canonical target fact",
            }
            for actor_id, profile in self.relationship_profiles.items()
        ]

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

    def _plan_context(self) -> dict[str, Any]:
        strategic_goal = self.plan.goal
        tactical_intention = self.cognition.intention
        return {
            **self.plan.public_view(),
            "strategic_goal": strategic_goal,
            "current_tactical_intention": tactical_intention,
            "alignment": (
                "aligned"
                if strategic_goal == tactical_intention
                else "temporary_tactical_deviation"
            ),
            "semantics": (
                "The strategic goal persists across turns; the tactical intention may change "
                "with coping pressure without automatically replacing that goal."
            ),
        }

    def _decision_tendencies(
        self,
        legal: list[Action],
        utilities: list[float],
        components: list[dict[str, float]],
    ) -> dict[str, Any]:
        peak = max(utilities)
        weights = [math.exp((value - peak) / 0.35) for value in utilities]
        total = sum(weights)
        ranks = {
            index: rank
            for rank, index in enumerate(
                sorted(range(len(utilities)), key=utilities.__getitem__, reverse=True),
                start=1,
            )
        }
        actions: list[dict[str, Any]] = []
        for index, action in enumerate(legal):
            drivers = sorted(
                components[index].items(), key=lambda item: abs(item[1]), reverse=True
            )
            actions.append(
                {
                    "action_index": index,
                    "action_type": action.type,
                    "attraction": round(weights[index] / total, 3),
                    "rank": ranks[index],
                    "relative_to_peak": round(utilities[index] - peak, 3),
                    "dominant_drivers": [
                        {"name": name, "effect": round(value, 3)}
                        for name, value in drivers[:3]
                        if abs(value) >= 0.02
                    ],
                }
            )
        return {
            "semantics": "fallible_motivational_attraction_not_action_command",
            "decision_mode": self.decision_mode.value,
            "strategic_goal": self.plan.goal,
            "tactical_intention": self.cognition.intention,
            "actions": actions,
        }

    def _prediction(self, mod: GameMod, action: Action) -> dict[str, Any]:
        expectations = {
            ("agent_town", "move_to"): (
                "reach a chosen place while spending some energy",
                ["town_moved"],
            ),
            ("agent_town", "work"): (
                "advance work experience and town progress",
                ["town_worked"],
            ),
            ("agent_town", "socialize"): (
                "change a specific relationship through conversation",
                ["town_conversation"],
            ),
            ("agent_town", "rest"): ("recover energy", ["town_rested"]),
            ("agent_town", "explore"): (
                "discover a small environmental detail",
                ["town_explored"],
            ),
            ("agent_town", "wait"): ("observe the immediate area", ["town_waited"]),
            ("agent_town", "respond_incident"): (
                "reduce an observed incident through direct effort",
                ["town_incident_response"],
            ),
            ("agent_town", "seek_shelter"): (
                "move away from a known environmental threat",
                ["town_sheltered"],
            ),
            ("agent_town", "check_bulletin"): (
                "update knowledge from discoverable public records",
                ["town_news_checked"],
            ),
            ("agent_town", "support_neighbor"): (
                "help a nearby resident regulate distress",
                ["town_neighbor_supported"],
            ),
            ("agent_town", "check_phone"): (
                "observe a ranked social feed without treating posts as facts",
                ["town_social_feed_checked"],
            ),
            ("agent_town", "publish_post"): (
                "publish a sourced report to a configured social platform",
                ["town_social_posted"],
            ),
            ("agent_town", "reshare_post"): (
                "relay a post while preserving its provenance chain",
                ["town_social_reshared"],
            ),
            ("agent_town", "comment_post"): (
                "discuss or question a public post",
                ["town_social_commented"],
            ),
            ("agent_town", "verify_claim"): (
                "compare a social claim with canonical evidence",
                ["town_claim_verified"],
            ),
            ("agent_town", "investigate_claim"): (
                "acquire new observed evidence before judging a social claim",
                ["town_claim_investigated"],
            ),
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
