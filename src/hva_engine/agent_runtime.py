from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import Random
from typing import Any

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
from hva_engine.llm import LLMDecisionClient
from hva_engine.models import Action, GameEvent
from hva_engine.mods.base import GameMod


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
        self._update_opponent_model(events)
        self._update_intention("coop" in self.role)
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
        self.last_context = ContextComposer().compose(
            match_id=self._match_id,
            agent_id=self.player_id,
            role=self.role,
            mod=mod,
            state=state,
            world_model=self.world_model,
            memory=[item.__dict__ for item in self.memory],
            legal_actions=legal,
            shared_facts=self._shared_facts,
            persona=self.profile.public_view(),
            identity=self.identity.private_view(),
            cognitive_state=self.cognition.public_view(),
            opponent_model=self.opponent_model,
            behavior_policy=self.behavior_policy.public_view(),
            fact_graph=self.fact_graph.private_view(),
        )
        baseline_action = mod.agent_action(state, self.player_id, legal, rng)
        learned: dict[str, list[float]] = {}
        for item in self.memory:
            learned.setdefault(item.action, []).append(item.score_delta - 0.15 * item.regret)
        averages = {key: sum(values) / len(values) for key, values in learned.items()}
        last_action = self.memory[-1].action if self.memory else None
        utilities, components = action_utilities(
            legal=legal,
            baseline=baseline_action,
            learned_values=averages,
            last_action=last_action,
            profile=self.profile,
            cognition=self.cognition,
            policy=self.behavior_policy,
            identity_biases=self.identity.action_biases,
            cooperative="coop" in self.role,
            rng=rng,
        )
        llm_decision = None
        llm_error: dict[str, str] | None = None
        fact_proposal_result: dict[str, Any] | None = None
        if decision_client is not None:
            try:
                llm_decision = decision_client.choose_structured_sync(
                    self.last_context.messages,
                    [action.model_dump() for action in legal],
                )
                choice_index = llm_decision.action_index
                fact_proposal_result = self.apply_fact_proposals(llm_decision.fact_proposals)
            except Exception as exc:
                if not llm_fallback:
                    raise
                llm_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                choice_index = bounded_choice(utilities, self.behavior_policy.realism, rng)
        else:
            choice_index = bounded_choice(utilities, self.behavior_policy.realism, rng)
        action = legal[choice_index]
        best_index = max(range(len(utilities)), key=utilities.__getitem__)
        regret = max(0.0, utilities[best_index] - utilities[choice_index])
        memory_used = len(self.memory) >= 2
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
            "llm": (
                {
                    "provider": decision_client.provider.name,
                    "model": llm_decision.response.model,
                    "usage": llm_decision.response.usage,
                    "fact_proposals": fact_proposal_result,
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
            "world_model": self.world_model,
            "persona": self.profile.public_view(),
            "identity": self.identity.public_view(self._revealed_story_titles),
            "cognitive_state": self.cognition.public_view(),
            "psychological_matrix": self.cognition.psychology_view(),
            "opponent_model": self.opponent_model,
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
            },
            "predicted_effect": prediction["description"],
            "prediction": prediction,
            "decision_regret_proxy": round(regret, 3),
            "prompt_layers": self.last_context.layers,
            "context_policy": self.last_context.diagnostics,
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

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "decisions": self.decisions,
            "persona": self.profile.public_view(),
            "identity": self.identity.public_view(self._revealed_story_titles),
            "cognitive_state": self.cognition.public_view(),
            "psychological_matrix": self.cognition.psychology_view(),
            "opponent_model": self.opponent_model,
            "behavior_policy": self.behavior_policy.public_view(),
            "fact_graph": self.fact_graph.public_view(),
            "world_model": self.world_model,
            "memory_depth": len(self.memory),
            "recent_memory": [item.__dict__ for item in list(self.memory)[-3:]],
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

    def maybe_reveal_story(self, terminal: bool = False) -> dict[str, Any] | None:
        memories = {memory.title: memory for memory in self.identity.formative_memories}
        thresholds = (
            (
                "the first costly lesson",
                self.decisions >= 2
                or self.cognition.stress >= 0.42
                or self.cognition.frustration >= 0.35,
                "pressure_or_setback",
            ),
            (
                "a hard-won success",
                self.decisions >= 4
                or self.cognition.confidence >= 0.68
                or self.opponent_model.get("observed_actions", 0) >= 3,
                "trust_or_pattern_recognition",
            ),
            (
                "a private promise",
                self.decisions >= 6 or terminal,
                "commitment_or_finale",
            ),
        )
        for title, eligible, trigger in thresholds:
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
