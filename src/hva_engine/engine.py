from __future__ import annotations

import os
from dataclasses import dataclass, field
from random import Random
from typing import Any
from uuid import uuid4

from hva_engine.agent_runtime import AgentBrain
from hva_engine.cognition import AgentIdentity, CognitiveProfile, RuntimeBehaviorPolicy
from hva_engine.context import SharedBlackboard
from hva_engine.evaluation import MatchEvaluator
from hva_engine.fact_graph import AgentFactGraph
from hva_engine.fact_store import FactStore, InMemoryFactStore, build_fact_store_from_env
from hva_engine.llm import LLMDecisionClient, OpenAICompatibleProvider
from hva_engine.models import (
    Action,
    ActorKind,
    AgentTuning,
    GameEvent,
    MatchMode,
    MatchStatus,
    MatchView,
    Player,
)
from hva_engine.mods import (
    AdversarialInterview,
    CrisisCoop,
    DebateArena,
    RacingStrategy,
    TacticalDuel,
)
from hva_engine.mods.base import GameMod


class EngineError(ValueError):
    pass


@dataclass
class Match:
    id: str
    mod: GameMod
    mode: MatchMode
    players: list[Player]
    human_player_id: str | None
    state: dict[str, Any]
    rng: Random
    events: list[GameEvent] = field(default_factory=list)
    agent_brains: dict[str, AgentBrain] = field(default_factory=dict)
    blackboard: SharedBlackboard | None = None

    @property
    def status(self) -> MatchStatus:
        return MatchStatus.FINISHED if self.mod.is_terminal(self.state) else MatchStatus.ACTIVE

    def add_event(self, event_type: str, actor_id: str | None = None, **payload: Any) -> GameEvent:
        event = GameEvent(
            seq=len(self.events) + 1, type=event_type, actor_id=actor_id, payload=payload
        )
        self.events.append(event)
        return event


class GameEngine:
    def __init__(
        self,
        fact_store: FactStore | None = None,
        llm_decision_client: LLMDecisionClient | None = None,
        llm_mod_ids: set[str] | None = None,
        llm_fallback: bool = True,
    ) -> None:
        self.mods: dict[str, GameMod] = {}
        self.matches: dict[str, Match] = {}
        self.evaluator = MatchEvaluator()
        self.fact_store = fact_store or InMemoryFactStore()
        self.llm_decision_client = llm_decision_client
        self.llm_mod_ids = llm_mod_ids or set()
        self.llm_fallback = llm_fallback
        self.agent_runtime = "llm" if llm_decision_client is not None else "baseline"

    def register(self, mod: GameMod) -> None:
        if mod.id in self.mods:
            raise EngineError(f"MOD already registered: {mod.id}")
        self.mods[mod.id] = mod

    def create_match(
        self,
        mod_id: str,
        human_name: str = "Human",
        seed: int | None = None,
        mode: MatchMode | str | None = None,
        reverse_seats: bool = False,
        agent_tuning: AgentTuning | None = None,
    ) -> MatchView:
        if mod_id not in self.mods:
            raise EngineError(f"Unknown MOD: {mod_id}")
        match_id = uuid4().hex
        rng = Random(seed if seed is not None else int(match_id[:8], 16))
        mod = self.mods[mod_id]
        selected_mode = (
            MatchMode(mode)
            if mode
            else (
                MatchMode.HUMAN_VS_AGENT
                if MatchMode.HUMAN_VS_AGENT.value in mod.supported_modes
                else MatchMode(mod.supported_modes[0])
            )
        )
        if selected_mode.value not in mod.supported_modes:
            supported = ", ".join(mod.supported_modes)
            raise EngineError(
                f"MOD {mod_id} does not support {selected_mode.value}; supported: {supported}"
            )
        players = self._make_players(selected_mode, human_name)
        if reverse_seats:
            players.reverse()
        human = next((p for p in players if p.kind == ActorKind.HUMAN), None)
        state = mod.initial_state(players, rng)
        tuning = agent_tuning or AgentTuning()
        behavior_policy = RuntimeBehaviorPolicy.from_tuning(tuning)
        brains: dict[str, AgentBrain] = {}
        for player in players:
            if player.kind != ActorKind.AGENT:
                continue
            role = "coop" if "coop" in selected_mode.value else "opponent"
            profile = CognitiveProfile.sample(rng, role, behavior_policy)
            identity = AgentIdentity.sample(player.name, profile, role, rng)
            fact_graph = AgentFactGraph.from_identity(player.id, identity, self.fact_store)
            brains[player.id] = AgentBrain(
                player.id,
                role,
                profile,
                identity,
                behavior_policy,
                fact_graph,
            )
        match = Match(
            id=match_id,
            mod=mod,
            mode=selected_mode,
            players=players,
            human_player_id=human.id if human else None,
            state=state,
            rng=rng,
            agent_brains=brains,
            blackboard=(
                SharedBlackboard({player.id for player in players})
                if "coop" in selected_mode.value
                else None
            ),
        )
        match.add_event("match_created", mod_id=mod_id, mode=selected_mode.value, seed=seed)
        self.matches[match_id] = match
        self._run_agents(match)
        return self.view(match_id)

    def _make_players(self, mode: MatchMode, human_name: str) -> list[Player]:
        if mode in {MatchMode.HUMAN_VS_AGENT, MatchMode.HUMAN_AGENT_COOP}:
            return [
                Player(id=f"human-{uuid4().hex[:8]}", name=human_name, kind=ActorKind.HUMAN),
                Player(id=f"agent-{uuid4().hex[:8]}", name="Astra", kind=ActorKind.AGENT),
            ]
        return [
            Player(id=f"agent-{uuid4().hex[:8]}", name="Astra", kind=ActorKind.AGENT),
            Player(id=f"agent-{uuid4().hex[:8]}", name="Nova", kind=ActorKind.AGENT),
        ]

    def submit(self, match_id: str, actor_id: str, action: Action) -> MatchView:
        match = self.get(match_id)
        if match.status == MatchStatus.FINISHED:
            raise EngineError("Match is already finished")
        if actor_id != match.mod.current_player_id(match.state):
            raise EngineError("It is not this actor's turn")
        legal = match.mod.legal_actions(match.state, actor_id)
        canonical = next((candidate for candidate in legal if candidate == action), None)
        if canonical is None:
            allowed = [candidate.model_dump() for candidate in legal]
            raise EngineError(f"Illegal action. Allowed actions: {allowed}")
        self._apply(match, actor_id, canonical, source="player")
        self._run_agents(match)
        return self.view(match_id)

    def _apply(
        self, match: Match, actor_id: str, action: Action, source: str
    ) -> list[dict[str, Any]]:
        previous_state = match.state
        scores_before = match.mod.scores(previous_state)
        new_state, emitted = match.mod.apply_action(previous_state, actor_id, action, match.rng)
        match.state = new_state
        scores_after = match.mod.scores(match.state)
        changed_keys = sorted(
            key
            for key in set(previous_state) | set(new_state)
            if previous_state.get(key) != new_state.get(key)
        )
        match.add_event(
            "action_applied",
            actor_id,
            action_type=action.type,
            action_payload=action.payload,
            source=source,
            scores_before=scores_before,
            scores_after=scores_after,
            changed_state_keys=changed_keys,
            leaders_after=self._leaders(scores_after),
        )
        for item in emitted:
            details = dict(item)
            event_type = details.pop("type")
            match.add_event(event_type, actor_id, **details)
        if match.mod.is_terminal(match.state):
            match.add_event("match_finished", scores=match.mod.scores(match.state))
        if match.blackboard is not None:
            outcomes = ",".join(item["type"] for item in emitted) or "state_updated"
            match.blackboard.publish(
                actor_id,
                f"Actor {actor_id} used {action.type}; observed outcomes: {outcomes}",
                ("game_fact", action.type),
            )
        return emitted

    def _leaders(self, scores: dict[str, float]) -> list[str]:
        if not scores:
            return []
        best = max(scores.values())
        return sorted(player_id for player_id, value in scores.items() if value == best)

    def _run_agents(self, match: Match) -> None:
        player_map = {player.id: player for player in match.players}
        guard = 0
        while match.status == MatchStatus.ACTIVE:
            actor_id = match.mod.current_player_id(match.state)
            if actor_id is None or player_map[actor_id].kind != ActorKind.AGENT:
                break
            legal = match.mod.legal_actions(match.state, actor_id)
            if not legal:
                raise EngineError("MOD returned no legal action for an active agent")
            brain = match.agent_brains[actor_id]
            shared_facts = match.blackboard.view_for(actor_id) if match.blackboard else []
            brain.observe(
                match.mod,
                match.state,
                match.mod.scores(match.state),
                match.events,
                match.id,
                shared_facts,
            )
            use_llm = self.llm_decision_client if match.mod.id in self.llm_mod_ids else None
            action, trace = brain.decide(
                match.mod,
                match.state,
                legal,
                match.rng,
                decision_client=use_llm,
                llm_fallback=self.llm_fallback,
            )
            if action not in legal:
                raise EngineError("Agent policy returned an illegal action")
            score_before = match.mod.scores(match.state).get(actor_id, 0.0)
            decision_event = match.add_event(
                "agent_decision", actor_id, action_type=action.type, **trace
            )
            applied_action = action
            if trace.get("utterance"):
                applied_action = action.model_copy(
                    update={
                        "payload": {
                            **action.payload,
                            "utterance": str(trace["utterance"]),
                        }
                    }
                )
            emitted = self._apply(
                match,
                actor_id,
                applied_action,
                source="llm_agent" if trace["decision_source"] == "llm" else "baseline_agent",
            )
            expected = set(trace["prediction"]["expected_events"])
            actual = {item["type"] for item in emitted}
            decision_event.payload["prediction_verified"] = bool(expected) and bool(
                expected & actual
            )
            brain.remember(
                int(match.state.get("turn", len(match.events))),
                action,
                emitted,
                score_before,
                match.mod.scores(match.state).get(actor_id, 0.0),
                trace,
            )
            if reveal := brain.maybe_reveal_story(match.status == MatchStatus.FINISHED):
                match.add_event("story_reveal", actor_id, **reveal)
            guard += 1
            if guard > 100:
                raise EngineError("Agent turn loop exceeded safety limit")

    def get(self, match_id: str) -> Match:
        try:
            return self.matches[match_id]
        except KeyError as exc:
            raise EngineError(f"Unknown match: {match_id}") from exc

    def view(self, match_id: str) -> MatchView:
        match = self.get(match_id)
        current = match.mod.current_player_id(match.state)
        legal = match.mod.legal_actions(match.state, current) if current else []
        return MatchView(
            id=match.id,
            mod_id=match.mod.id,
            mode=match.mode,
            status=match.status,
            players=match.players,
            human_player_id=match.human_player_id,
            current_player_id=current,
            state=match.mod.public_state(match.state),
            legal_actions=legal,
            scores=match.mod.scores(match.state),
            events=match.events,
            agent_summaries={pid: brain.summary() for pid, brain in match.agent_brains.items()},
        )

    def evaluation(self, match_id: str) -> dict[str, Any]:
        match = self.get(match_id)
        return self.evaluator.evaluate(
            match.mod,
            match.players,
            match.events,
            match.mod.scores(match.state),
            match.status == MatchStatus.FINISHED,
            match.mode,
        )

    def evaluation_summary(self) -> dict[str, Any]:
        rows = [self.evaluation(match_id) for match_id in self.matches]
        if not rows:
            return {"matches": 0, "overall": None, "groups": {}}

        def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
            dimensions = items[0]["dimensions"]
            composite_values = [item["composite_score"] for item in items]
            return {
                "matches": len(items),
                "composite_score": round(sum(composite_values) / len(composite_values), 3),
                "composite_sd": round(
                    (
                        sum(
                            (value - sum(composite_values) / len(composite_values)) ** 2
                            for value in composite_values
                        )
                        / len(composite_values)
                    )
                    ** 0.5,
                    3,
                ),
                "dimensions": {
                    key: (
                        round(sum(values) / len(values), 3)
                        if (
                            values := [
                                item["dimensions"][key]
                                for item in items
                                if item["dimensions"][key] is not None
                            ]
                        )
                        else None
                    )
                    for key in dimensions
                },
            }

        grouped: dict[str, list[dict[str, Any]]] = {}
        for match_id, match in self.matches.items():
            key = f"{match.mod.id}:{match.mode.value}"
            grouped.setdefault(key, []).append(self.evaluation(match_id))
        return {
            "matches": len(rows),
            "overall": aggregate(rows),
            "groups": {key: aggregate(items) for key, items in grouped.items()},
        }

    def context_preview(self, match_id: str, agent_id: str) -> dict[str, Any]:
        match = self.get(match_id)
        try:
            brain = match.agent_brains[agent_id]
        except KeyError as exc:
            raise EngineError("Unknown agent in this match") from exc
        if brain.last_context is None:
            return {"owner_agent_id": agent_id, "status": "no_decision_yet"}
        packet = brain.last_context
        return {
            "owner_agent_id": packet.owner_agent_id,
            "layers": packet.layers,
            "diagnostics": packet.diagnostics,
            "messages": [message.__dict__ for message in packet.messages],
        }

    def public_fact_graph(self, match_id: str, agent_id: str) -> dict[str, Any]:
        match = self.get(match_id)
        try:
            brain = match.agent_brains[agent_id]
        except KeyError as exc:
            raise EngineError("Unknown agent in this match") from exc
        return brain.fact_graph.public_view()


def build_default_engine() -> GameEngine:
    runtime = os.environ.get("HVA_AGENT_RUNTIME", "baseline").strip().lower()
    decision_client: LLMDecisionClient | None = None
    llm_mod_ids: set[str] = set()
    if runtime in {"llm", "hybrid"}:
        provider = OpenAICompatibleProvider.from_env()
        decision_client = LLMDecisionClient(
            provider,
            temperature=float(os.environ.get("HVA_LLM_TEMPERATURE", "0.75")),
            max_tokens=int(os.environ.get("HVA_LLM_MAX_TOKENS", "900")),
        )
        llm_mod_ids = {
            value.strip()
            for value in os.environ.get("HVA_LLM_MODS", "adversarial_interview").split(",")
            if value.strip()
        }
    engine = GameEngine(
        build_fact_store_from_env(),
        llm_decision_client=decision_client,
        llm_mod_ids=llm_mod_ids,
        llm_fallback=os.environ.get("HVA_LLM_FALLBACK", "true").lower() not in {"0", "false", "no"},
    )
    for mod in (
        TacticalDuel(),
        RacingStrategy(),
        DebateArena(),
        CrisisCoop(),
        AdversarialInterview(),
    ):
        engine.register(mod)
    return engine
