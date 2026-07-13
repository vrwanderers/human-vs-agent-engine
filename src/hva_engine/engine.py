from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from random import Random
from typing import Any
from uuid import uuid4

from hva_engine.agent_runtime import AgentBrain
from hva_engine.character_cards import CharacterCardError, CharacterCardRegistry
from hva_engine.cognition import AgentIdentity, CognitiveProfile, RuntimeBehaviorPolicy
from hva_engine.context import SharedBlackboard
from hva_engine.evaluation import MatchEvaluator
from hva_engine.fact_graph import AgentFactGraph
from hva_engine.fact_store import FactStore, InMemoryFactStore, build_fact_store_from_env
from hva_engine.human_cognition import MemorySystem
from hva_engine.llm import LLMDecisionClient, OpenAICompatibleProvider
from hva_engine.memory_store import (
    InMemoryIndexedMemoryStore,
    LongTermMemoryStore,
    build_memory_store_from_env,
)
from hva_engine.models import (
    Action,
    ActorKind,
    AgentCharacterSelection,
    AgentTuning,
    EventVisibility,
    GameEvent,
    MatchMode,
    MatchStatus,
    MatchView,
    Player,
)
from hva_engine.mods import (
    AdversarialInterview,
    AgentTown,
    CrisisCoop,
    DebateArena,
    RacingStrategy,
    TacticalDuel,
)
from hva_engine.mods.base import GameMod
from hva_engine.observation import ObservationPolicy
from hva_engine.stimulus import RealityStatus, StimulusModality, StimulusPrivacy
from hva_engine.world_store import (
    InMemoryWorldStateStore,
    WorldStateStore,
    build_world_store_from_env,
)


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
    world_id: str | None = None
    world_revision: int | None = None

    @property
    def status(self) -> MatchStatus:
        return MatchStatus.FINISHED if self.mod.is_terminal(self.state) else MatchStatus.ACTIVE

    def add_event(
        self,
        event_type: str,
        actor_id: str | None = None,
        visibility: EventVisibility = EventVisibility.PUBLIC,
        **payload: Any,
    ) -> GameEvent:
        event = GameEvent(
            seq=len(self.events) + 1,
            type=event_type,
            actor_id=actor_id,
            visibility=visibility,
            payload=payload,
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
        character_cards: CharacterCardRegistry | None = None,
        memory_store: LongTermMemoryStore | None = None,
        world_store: WorldStateStore | None = None,
    ) -> None:
        self.mods: dict[str, GameMod] = {}
        self.matches: dict[str, Match] = {}
        self.evaluator = MatchEvaluator()
        self.fact_store = fact_store or InMemoryFactStore()
        self.llm_decision_client = llm_decision_client
        self.llm_mod_ids = llm_mod_ids or set()
        self.llm_fallback = llm_fallback
        self.agent_runtime = "llm" if llm_decision_client is not None else "baseline"
        self.character_cards = character_cards or CharacterCardRegistry.load_default()
        self.observation_policy = ObservationPolicy()
        self.memory_store = memory_store or InMemoryIndexedMemoryStore()
        self.world_store = world_store or InMemoryWorldStateStore()

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
        agent_characters: list[AgentCharacterSelection] | None = None,
        human_memory_id: str | None = None,
        agent_memory_owner_ids: list[str] | None = None,
        world_id: str | None = None,
        resume_world: bool = False,
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
        players = self._make_players(selected_mode, human_name, mod)
        if reverse_seats:
            players.reverse()
        selections = agent_characters or []
        agent_players = [player for player in players if player.kind == ActorKind.AGENT]
        if len(selections) > len(agent_players):
            raise EngineError(
                f"Received {len(selections)} character cards for {len(agent_players)} agents"
            )
        resolved_cards: dict[str, tuple[Any, str]] = {}
        supplied_memory_owners = agent_memory_owner_ids or []
        if len(supplied_memory_owners) > len(agent_players):
            raise EngineError(
                f"Received {len(supplied_memory_owners)} memory owners for "
                f"{len(agent_players)} agents"
            )
        memory_owner_ids = {
            player.id: supplied_memory_owners[index]
            for index, player in enumerate(agent_players[: len(supplied_memory_owners)])
        }
        for index, (player, selection) in enumerate(
            zip(agent_players, selections, strict=False)
        ):
            try:
                card, source_kind = self.character_cards.resolve(selection)
            except CharacterCardError as exc:
                raise EngineError(str(exc)) from exc
            player.name = card.name
            resolved_cards[player.id] = (card, source_kind)
            if selection.memory_owner_id is not None:
                if index < len(supplied_memory_owners):
                    raise EngineError(
                        "Provide a memory owner either with the character selection "
                        "or agent_memory_owner_ids, not both"
                    )
                memory_owner_ids[player.id] = selection.memory_owner_id
        requested_memory_owners = list(memory_owner_ids.values())
        if len(requested_memory_owners) != len(set(requested_memory_owners)):
            raise EngineError("memory_owner_id must be unique for each agent in a match")
        human = next((p for p in players if p.kind == ActorKind.HUMAN), None)
        participant_memory_ids = {
            player.id: (
                human_memory_id or player.id
                if player.kind == ActorKind.HUMAN
                else memory_owner_ids.get(player.id, player.id)
            )
            for player in players
        }
        state = mod.initial_state(players, rng)
        state["_memory_owner_ids"] = deepcopy(participant_memory_ids)
        world_revision: int | None = None
        if resume_world and not world_id:
            raise EngineError("resume_world requires world_id")
        if world_id:
            if mod.persistent_world_state(state) is None:
                raise EngineError(f"MOD {mod_id} does not support persistent worlds")
            snapshot = self.world_store.load(world_id)
            if resume_world:
                if snapshot is None:
                    raise EngineError(f"Unknown persistent world: {world_id}")
                if snapshot.mod_id != mod_id:
                    raise EngineError(
                        f"World {world_id} belongs to MOD {snapshot.mod_id}, not {mod_id}"
                    )
                state = mod.restore_persistent_world(state, snapshot.state)
                state["_memory_owner_ids"] = deepcopy(participant_memory_ids)
                world_revision = snapshot.revision
            elif snapshot is not None:
                raise EngineError(
                    f"World {world_id} already exists; set resume_world=true to continue it"
                )
            state["world_id"] = world_id
            state["world_revision"] = world_revision or 0
        tuning = agent_tuning or AgentTuning()
        behavior_policy = RuntimeBehaviorPolicy.from_tuning(tuning)
        brains: dict[str, AgentBrain] = {}
        for player in players:
            if player.kind != ActorKind.AGENT:
                continue
            role = "coop" if "coop" in selected_mode.value else "opponent"
            if player.id in resolved_cards:
                card, source_kind = resolved_cards[player.id]
                profile, identity = self.character_cards.instantiate(
                    card, behavior_policy, source_kind
                )
            else:
                generated_character = mod.agent_character(
                    state,
                    player,
                    role,
                    behavior_policy,
                    participant_memory_ids[player.id],
                    rng,
                )
                if generated_character is None:
                    profile = CognitiveProfile.sample(rng, role, behavior_policy)
                    identity = AgentIdentity.sample(player.name, profile, role, rng)
                else:
                    profile, identity = generated_character
            fact_graph = AgentFactGraph.from_identity(
                participant_memory_ids[player.id], identity, self.fact_store
            )
            brains[player.id] = AgentBrain(
                player.id,
                role,
                profile,
                identity,
                behavior_policy,
                fact_graph,
                participant_directory={
                    other.id: {
                        "memory_key": participant_memory_ids[other.id],
                        "display_name": other.name,
                        "kind": other.kind.value,
                    }
                    for other in players
                    if other.id != player.id
                },
                memory_system=MemorySystem(
                    owner_id=participant_memory_ids[player.id],
                    store=self.memory_store,
                ),
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
            world_id=world_id,
            world_revision=world_revision,
        )
        match.add_event(
            "match_created",
            mod_id=mod_id,
            mode=selected_mode.value,
            seed=seed,
            character_cards={
                player_id: (
                    card.id if source_kind == "builtin" else f"custom:{card.id}"
                )
                for player_id, (card, source_kind) in resolved_cards.items()
            },
            character_decision_model="runtime_cognition_not_scripted_actions",
        )
        self.matches[match_id] = match
        if world_id and not resume_world:
            self._save_world(match)
        elif world_id:
            match.add_event(
                "world_resumed",
                world_id=world_id,
                world_revision=world_revision,
            )
        self._run_agents(match)
        return self.view(match_id, match.human_player_id)

    def _make_players(
        self, mode: MatchMode, human_name: str, mod: GameMod
    ) -> list[Player]:
        agent_names = ("Astra", "Nova", "Mira", "Orion", "Sora", "Lin", "Iris", "Theo")
        agent_count = mod.agent_count_for_mode(mode.value)
        if not 1 <= agent_count <= len(agent_names):
            raise EngineError("MOD agent_count_for_mode must be between 1 and 8")
        agents = [
            Player(
                id=f"agent-{uuid4().hex[:8]}",
                name=agent_names[index],
                kind=ActorKind.AGENT,
            )
            for index in range(agent_count)
        ]
        if mode in {MatchMode.HUMAN_VS_AGENT, MatchMode.HUMAN_AGENT_COOP}:
            return [
                Player(id=f"human-{uuid4().hex[:8]}", name=human_name, kind=ActorKind.HUMAN),
                *agents,
            ]
        return agents

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
        return self.view(match_id, match.human_player_id)

    def publish_stimulus(
        self,
        match_id: str,
        *,
        target_agent_id: str,
        modality: StimulusModality | str,
        semantic_tags: list[str] | tuple[str, ...] = (),
        source_id: str = "world",
        intensity: float = 0.5,
        valence: float = 0.0,
        urgency: float = 0.3,
        novelty: float = 0.5,
        uncertainty: float | None = None,
        reality_status: RealityStatus | str | None = None,
        privacy: StimulusPrivacy | str = StimulusPrivacy.PUBLIC,
        causal_group: str | None = None,
    ) -> GameEvent:
        """Trusted ingestion boundary for vision/audio/touch/internal-event adapters.

        Publishing a stimulus only changes the Agent's next observation. It never mutates
        MOD state or the canonical fact graph and therefore cannot bypass legal actions.
        """

        match = self.get(match_id)
        if match.status == MatchStatus.FINISHED:
            raise EngineError("Cannot publish a stimulus to a finished match")
        if target_agent_id not in match.agent_brains:
            raise EngineError("Stimulus target must be an agent in this match")
        try:
            normalized_modality = StimulusModality(modality)
            normalized_privacy = StimulusPrivacy(privacy)
            normalized_status = (
                RealityStatus(reality_status)
                if reality_status is not None
                else {
                    StimulusModality.IMAGINATION: RealityStatus.IMAGINED,
                    StimulusModality.MEMORY: RealityStatus.REMEMBERED,
                    StimulusModality.WORLD_EVENT: RealityStatus.CANONICAL,
                }.get(normalized_modality, RealityStatus.OBSERVED)
            )
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        if (
            normalized_modality == StimulusModality.IMAGINATION
            and normalized_status != RealityStatus.IMAGINED
        ):
            raise EngineError("Imagination stimuli must keep reality_status=imagined")
        if (
            normalized_modality == StimulusModality.MEMORY
            and normalized_status != RealityStatus.REMEMBERED
        ):
            raise EngineError("Memory stimuli must keep reality_status=remembered")
        cleaned_tags = []
        for value in semantic_tags:
            cleaned = " ".join(str(value).split())[:80].lower()
            if cleaned and cleaned not in cleaned_tags:
                cleaned_tags.append(cleaned)
        numeric = {
            "intensity": max(0.0, min(1.0, float(intensity))),
            "valence": max(-1.0, min(1.0, float(valence))),
            "urgency": max(0.0, min(1.0, float(urgency))),
            "novelty": max(0.0, min(1.0, float(novelty))),
        }
        if uncertainty is not None:
            numeric["uncertainty"] = max(0.0, min(1.0, float(uncertainty)))
        event_actor = (
            target_agent_id
            if normalized_privacy == StimulusPrivacy.AGENT_PRIVATE
            else source_id
        )
        visibility = (
            EventVisibility.ENGINE_PRIVATE
            if normalized_privacy == StimulusPrivacy.AGENT_PRIVATE
            else EventVisibility.PUBLIC
        )
        return match.add_event(
            "sensory_stimulus",
            event_actor,
            visibility=visibility,
            stimulus={
                "source_id": " ".join(source_id.split())[:128] or "world",
                "target_id": target_agent_id,
                "modality": normalized_modality.value,
                "semantic_tags": cleaned_tags[:12],
                "reality_status": normalized_status.value,
                "privacy": normalized_privacy.value,
                "causal_group": (
                    " ".join(causal_group.split())[:96] if causal_group else "external"
                ),
                **numeric,
            },
        )

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
            and key not in {"_event_schedule", "_knowledge", "world"}
        )
        public_action = match.mod.public_action(action, actor_id)
        match.add_event(
            "action_applied",
            actor_id,
            action_type=public_action.type,
            action_payload=public_action.payload,
            source=source,
            scores_before=scores_before,
            scores_after=scores_after,
            changed_state_keys=changed_keys,
            leaders_after=self._leaders(scores_after),
        )
        for item in emitted:
            details = dict(item)
            event_type = details.pop("type")
            event_actor_id = details.pop("_actor_id", actor_id)
            visible_to = details.pop("_visible_to", None)
            if visible_to is not None:
                details["visible_to"] = list(visible_to)
            match.add_event(event_type, event_actor_id, **details)
        if match.mod.is_terminal(match.state):
            match.add_event("match_finished", scores=match.mod.scores(match.state))
        if match.blackboard is not None:
            shareable_outcomes = [
                item["type"]
                for item in emitted
                if "_visible_to" not in item
            ]
            outcomes = ",".join(shareable_outcomes) or "state_updated"
            match.blackboard.publish(
                actor_id,
                f"Actor {actor_id} used {action.type}; observed outcomes: {outcomes}",
                ("game_fact", action.type),
            )
        self._save_world(match)
        return emitted

    def _save_world(self, match: Match) -> None:
        if not match.world_id:
            return
        snapshot_state = match.mod.persistent_world_state(match.state)
        if snapshot_state is None:
            return
        snapshot = self.world_store.save(match.world_id, match.mod.id, snapshot_state)
        match.world_revision = snapshot.revision
        match.state["world_revision"] = snapshot.revision

    def world_metadata(self, world_id: str) -> dict[str, Any]:
        metadata = self.world_store.metadata(world_id)
        if metadata is None:
            raise EngineError(f"Unknown persistent world: {world_id}")
        return metadata

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
            observation = self.observation_policy.for_agent(
                mod=match.mod,
                state=match.state,
                scores=match.mod.scores(match.state),
                events=self._events_for_agent(match, actor_id),
                shared_facts=shared_facts,
                agent_id=actor_id,
            )
            brain.observe(
                match.mod,
                observation.state,
                observation.scores,
                observation.events,
                match.id,
                observation.shared_facts,
            )
            use_llm = self.llm_decision_client if match.mod.id in self.llm_mod_ids else None
            action, trace = brain.decide(
                match.mod,
                observation.state,
                legal,
                match.rng,
                decision_client=use_llm,
                llm_fallback=self.llm_fallback,
            )
            if action not in legal:
                raise EngineError("Agent policy returned an illegal action")
            score_before = match.mod.scores(match.state).get(actor_id, 0.0)
            private_influence = trace.pop("_private_influence_intent")
            private_reflex = trace.pop("_private_reflex")
            trace["observation_policy"] = observation.diagnostics
            observable_reflex = trace.get("involuntary_response", {})
            if observable_reflex.get("cues"):
                match.add_event(
                    "agent_involuntary_cue",
                    actor_id,
                    **observable_reflex,
                )
            if private_reflex.get("stimulus_frame", {}).get("stimulus_count", 0):
                match.add_event(
                    "agent_reflex_diagnostic",
                    actor_id,
                    visibility=EventVisibility.ENGINE_PRIVATE,
                    **private_reflex,
                )
            match.add_event(
                "agent_influence_intent",
                actor_id,
                visibility=EventVisibility.ENGINE_PRIVATE,
                **private_influence,
            )
            decision_event = match.add_event(
                "agent_decision",
                actor_id,
                visibility=EventVisibility.ENGINE_PRIVATE,
                action_type=action.type,
                **trace,
            )
            applied_payload = {
                **action.payload,
                "response_plan": trace.get("response_plan", {}),
            }
            if trace.get("utterance"):
                applied_payload["utterance"] = str(trace["utterance"])
            applied_action = action.model_copy(update={"payload": applied_payload})
            emitted = self._apply(
                match,
                actor_id,
                applied_action,
                source="llm_agent" if trace["decision_source"] == "llm" else "baseline_agent",
            )
            perceived_emitted = [
                item
                for item in emitted
                if "_visible_to" not in item or actor_id in item["_visible_to"]
            ]
            expected = set(trace["prediction"]["expected_events"])
            actual = {item["type"] for item in perceived_emitted}
            decision_event.payload["prediction_verified"] = bool(expected) and bool(
                expected & actual
            )
            brain.remember(
                int(match.state.get("turn", len(match.events))),
                action,
                perceived_emitted,
                score_before,
                match.mod.scores(match.state).get(actor_id, 0.0),
                trace,
            )
            decision_event.payload["psychological_matrix_after_outcome"] = (
                brain.cognition.psychology_view()
            )
            decision_event.payload["outcome_reappraisal"] = brain.last_outcome_reappraisal
            decision_event.payload["skill_after_outcome"] = brain.last_skill_update
            requested_reveals = trace.get("response_plan", {}).get(
                "reveal_fact_ids", []
            )
            reveal = brain.maybe_reveal_story(
                match.status == MatchStatus.FINISHED,
                requested_fact_ids=requested_reveals,
            )
            if reveal:
                match.add_event("story_reveal", actor_id, **reveal)
            if requested_reveals:
                match.add_event(
                    "story_reveal_diagnostic",
                    actor_id,
                    visibility=EventVisibility.ENGINE_PRIVATE,
                    **brain.last_story_reveal_diagnostic,
                )
            guard += 1
            if guard > 100:
                raise EngineError("Agent turn loop exceeded safety limit")

    def _events_for_agent(self, match: Match, agent_id: str) -> list[GameEvent]:
        """Return public history plus only this agent's own engine-private intent."""

        return [
            event
            for event in match.events
            if (
                event.visibility == EventVisibility.PUBLIC
                and (
                    "visible_to" not in event.payload
                    or agent_id in event.payload["visible_to"]
                )
            )
            or (
                event.visibility == EventVisibility.ENGINE_PRIVATE
                and event.actor_id == agent_id
            )
        ]

    def get(self, match_id: str) -> Match:
        try:
            return self.matches[match_id]
        except KeyError as exc:
            raise EngineError(f"Unknown match: {match_id}") from exc

    def view(self, match_id: str, viewer_id: str | None = None) -> MatchView:
        match = self.get(match_id)
        if viewer_id is not None and viewer_id not in {player.id for player in match.players}:
            raise EngineError("Viewer is not a player in this match")
        current = match.mod.current_player_id(match.state)
        legal = (
            match.mod.legal_actions(match.state, current)
            if current and viewer_id == current
            else []
        )
        return MatchView(
            id=match.id,
            mod_id=match.mod.id,
            mode=match.mode,
            status=match.status,
            players=deepcopy(match.players),
            human_player_id=match.human_player_id,
            current_player_id=current,
            state=deepcopy(match.mod.public_state(deepcopy(match.state), viewer_id)),
            legal_actions=deepcopy(legal),
            scores=deepcopy(match.mod.scores(match.state)),
            events=[
                deepcopy(event)
                for event in match.events
                if event.visibility == EventVisibility.PUBLIC
            ],
            agent_summaries={
                pid: deepcopy(brain.public_summary())
                for pid, brain in match.agent_brains.items()
            },
            world_id=match.world_id,
            world_revision=match.world_revision,
            view_scope="player" if viewer_id else "public",
            viewer_id=viewer_id,
        )

    def debug_view(self, match_id: str) -> MatchView:
        """Trusted-development view. Never expose this without authorization."""

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
            state=match.state,
            legal_actions=legal,
            scores=match.mod.scores(match.state),
            events=match.events,
            agent_summaries={pid: brain.summary() for pid, brain in match.agent_brains.items()},
            world_id=match.world_id,
            world_revision=match.world_revision,
            view_scope="admin_debug",
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
            "context_id": packet.context_id,
            "content_sha256": packet.content_sha256,
            "layers": packet.layers,
            "diagnostics": packet.diagnostics,
            "decision_context": packet.decision_context,
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
        memory_store=build_memory_store_from_env(),
        world_store=build_world_store_from_env(),
    )
    for mod in (
        AgentTown(),
        TacticalDuel(),
        RacingStrategy(),
        DebateArena(),
        CrisisCoop(),
        AdversarialInterview(),
    ):
        engine.register(mod)
    return engine
