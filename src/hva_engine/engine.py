from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any
from uuid import uuid4

from hva_engine.agent_runtime import AgentBrain
from hva_engine.context import SharedBlackboard
from hva_engine.evaluation import MatchEvaluator
from hva_engine.models import (
    Action,
    ActorKind,
    GameEvent,
    MatchMode,
    MatchStatus,
    MatchView,
    Player,
)
from hva_engine.mods import CrisisCoop, DebateArena, RacingStrategy, TacticalDuel
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

    def add_event(self, event_type: str, actor_id: str | None = None, **payload: Any) -> None:
        self.events.append(
            GameEvent(seq=len(self.events) + 1, type=event_type, actor_id=actor_id, payload=payload)
        )


class GameEngine:
    def __init__(self) -> None:
        self.mods: dict[str, GameMod] = {}
        self.matches: dict[str, Match] = {}
        self.evaluator = MatchEvaluator()

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
        human = next((p for p in players if p.kind == ActorKind.HUMAN), None)
        brains = {
            p.id: AgentBrain(p.id, "coop" if "coop" in selected_mode.value else "opponent")
            for p in players
            if p.kind == ActorKind.AGENT
        }
        match = Match(
            id=match_id,
            mod=mod,
            mode=selected_mode,
            players=players,
            human_player_id=human.id if human else None,
            state=mod.initial_state(players, rng),
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
        new_state, emitted = match.mod.apply_action(match.state, actor_id, action, match.rng)
        match.state = new_state
        match.add_event(
            "action_applied",
            actor_id,
            action_type=action.type,
            action_payload=action.payload,
            source=source,
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
            action, trace = brain.decide(match.mod, match.state, legal, match.rng)
            if action not in legal:
                raise EngineError("Agent policy returned an illegal action")
            match.add_event("agent_decision", actor_id, action_type=action.type, **trace)
            emitted = self._apply(match, actor_id, action, source="baseline_agent")
            brain.remember(
                int(match.state.get("turn", len(match.events))),
                action,
                emitted,
                match.mod.scores(match.state).get(actor_id, 0.0),
            )
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
            return {
                "matches": len(items),
                "composite_score": round(
                    sum(item["composite_score"] for item in items) / len(items), 3
                ),
                "dimensions": {
                    key: round(sum(item["dimensions"][key] for item in items) / len(items), 3)
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


def build_default_engine() -> GameEngine:
    engine = GameEngine()
    for mod in (TacticalDuel(), RacingStrategy(), DebateArena(), CrisisCoop()):
        engine.register(mod)
    return engine
