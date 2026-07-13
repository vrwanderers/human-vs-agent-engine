from dataclasses import replace
from random import Random

from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.llm import LLMDecisionClient, LLMResponse
from hva_engine.models import Action, ActorKind, Player
from hva_engine.mods import AgentTown


class TownProvider:
    name = "town-provider"

    def __init__(self) -> None:
        self.calls = 0
        self.context_owners: list[str] = []

    def complete_sync(self, request):
        self.calls += 1
        self.context_owners.append(request.context_metadata["owner_agent_id"])
        return LLMResponse(
            '{"action_index":0,"reason":"observe before acting","fact_proposals":[]}',
            "town-test-model",
            {},
            {},
        )


def _wait(view):
    return next(action for action in view.legal_actions if action.type == "wait")


def test_town_creates_one_observer_and_three_independent_agents() -> None:
    engine = build_default_engine()
    manifest = engine.mods["agent_town"].manifest()
    assert manifest["agent_count_by_mode"]["human_agent_coop"] == 3
    assert manifest["supports_persistent_world"] is True
    assert {platform["format"] for platform in manifest["social_platforms"]} == {
        "microblog",
        "short_video",
    }

    view = engine.create_match("agent_town", seed=23)
    agents = [player for player in view.players if player.kind == ActorKind.AGENT]
    assert len(view.players) == 4
    assert len(agents) == 3
    assert view.current_player_id == view.human_player_id
    assert set(view.state["residents"]) == {player.id for player in view.players}
    assert len(engine.get(view.id).agent_brains) == 3


def test_each_observer_step_advances_three_agent_brains_and_public_town_state() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=23)
    for _ in range(12):
        view = engine.submit(view.id, view.human_player_id, _wait(view))

    match = engine.get(view.id)
    assert view.state["turn"] == 48
    assert view.state["day"] == 2
    assert all(brain.decisions == 12 for brain in match.agent_brains.values())
    assert all(
        summary["skill_learning"]["skill_count"] > 0
        for summary in view.agent_summaries.values()
    )
    assert any(
        summary["skill_learning"]["automatic_skills"]
        for summary in view.agent_summaries.values()
    )
    assert any(event.type == "town_moved" for event in view.events)
    assert any(event.type == "town_worked" for event in view.events)


def test_town_agent_only_simulation_runs_three_days_to_completion() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=9, mode="agent_coop")
    assert view.status == "finished"
    assert view.human_player_id is None
    assert len(view.players) == 4
    assert view.state["day"] == 4
    assert view.state["turn"] == 96
    assert all(
        summary["skill_learning"]["skill_count"]
        for summary in view.agent_summaries.values()
    )
    assert engine.evaluation(view.id)["valid_for_comparison"] is True


def test_town_skill_context_separates_routes_jobs_and_social_targets() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=5)
    mod = engine.mods["agent_town"]
    human_id = view.human_player_id
    assert human_id is not None
    actions = view.legal_actions
    route = next(action for action in actions if action.type == "move_to")
    route_context = mod.agent_skill_context(view.state, human_id, route)
    assert mod.agent_skill_id(route) == "town_navigation"
    assert route_context["route_id"].startswith("square->")
    assert route_context["weather"] == view.state["weather"]


def test_real_provider_path_keeps_three_town_contexts_isolated_and_untruncated() -> None:
    provider = TownProvider()
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider),
        llm_mod_ids={"agent_town"},
    )
    engine.register(AgentTown())
    view = engine.create_match("agent_town", seed=3)
    view = engine.submit(view.id, view.human_player_id, _wait(view))

    decisions = [
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    ]
    assert provider.calls == 3
    assert len(set(provider.context_owners)) == 3
    assert all(
        event.payload["context_policy"]["owner_agent_id"] == event.actor_id
        for event in decisions
    )
    assert all(
        not event.payload["context_policy"]["critical_sections_truncated"]
        for event in decisions
    )


def test_town_generates_stable_private_people_from_memory_owner_ids() -> None:
    engine = build_default_engine()
    owners = ["willow-astra", "willow-nova", "willow-mira"]
    first = engine.create_match(
        "agent_town", seed=3, agent_memory_owner_ids=owners
    )
    first_brains = list(engine.get(first.id).agent_brains.values())
    first_fingerprints = [
        (brain.profile.public_view(), brain.identity.private_view())
        for brain in first_brains
    ]

    assert [brain.memory_system.owner_id for brain in first_brains] == owners
    assert [brain.fact_graph.owner_id for brain in first_brains] == owners
    assert len({brain.identity.background for brain in first_brains}) == 3
    assert all(len(brain.identity.formative_memories) == 3 for brain in first_brains)
    assert all(len(brain.identity.lived_memories) == 2 for brain in first_brains)
    assert all(
        brain.identity.speech_style["constraint_source"]
        == "town_private_identity_generator"
        for brain in first_brains
    )
    assert all(
        summary["identity"]["background"] == "Not yet revealed"
        for summary in first.agent_summaries.values()
    )

    second = engine.create_match(
        "agent_town", seed=97, agent_memory_owner_ids=owners
    )
    second_brains = list(engine.get(second.id).agent_brains.values())
    assert [
        (brain.profile.public_view(), brain.identity.private_view())
        for brain in second_brains
    ] == first_fingerprints


def test_local_world_event_is_known_only_to_observers_until_discovered() -> None:
    mod = AgentTown()
    players = [
        Player(id="human", name="Observer", kind=ActorKind.HUMAN),
        Player(id="agent-a", name="Astra", kind=ActorKind.AGENT),
        Player(id="agent-b", name="Nova", kind=ActorKind.AGENT),
    ]
    rng = Random(11)
    state = mod.initial_state(players, rng)
    state["residents"]["agent-a"].update(
        {"location": "workshop", "x": 765, "y": 150}
    )
    state["residents"]["agent-b"].update(
        {"location": "farm", "x": 150, "y": 150}
    )
    state["minute_of_day"] = 13 * 60 + 30
    state["time"] = "13:30"
    state["_causal_state"]["facts"]["maintenance_risk"] = 0.95
    accident_rule = next(
        rule for rule in mod.event_director.RULES if rule.id == "workshop_accident"
    )
    mod.event_director.RULES = (replace(accident_rule, probability=1.0),)

    state, emitted = mod.apply_action(state, "human", Action(type="wait"), rng)
    accident = next(item for item in emitted if item.get("category") == "accident")
    event_id = accident["event_id"]
    assert accident["_visible_to"] == ["agent-a"]
    assert event_id in state["_knowledge"]["agent-a"]
    assert event_id not in state["_knowledge"]["agent-b"]
    assert any(
        event["id"] == event_id
        for event in mod.public_state(state, "agent-a")["world"]["event_history"]
    )
    assert all(
        event["id"] != event_id
        for event in mod.public_state(state, "agent-b")["world"]["event_history"]
    )
    assert mod.public_state(state, "agent-a")["world"]["risk_level"] >= 0.8
    assert mod.public_state(state, "agent-b")["world"]["risk_level"] <= 0.12

    response = next(
        action
        for action in mod.legal_actions(state, "agent-a")
        if action.type == "respond_incident"
    )
    effort_before = state["world"]["active_incidents"][0]["remaining_effort"]
    state, response_events = mod.apply_action(state, "agent-a", response, rng)
    assert state["world"]["active_incidents"][0]["remaining_effort"] < effort_before
    assert response_events[0]["type"] == "town_incident_response"

    move = next(
        action
        for action in mod.legal_actions(state, "agent-b")
        if action.type == "move_to" and action.payload["destination"] == "workshop"
    )
    state, movement_events = mod.apply_action(state, "agent-b", move, rng)
    assert any(event["type"] == "town_incident_discovered" for event in movement_events)
    assert event_id in state["_knowledge"]["agent-b"]


def test_three_day_world_model_emits_all_event_families_and_agents_react() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=9, mode="agent_coop")
    events = engine.get(view.id).events
    world_events = [event for event in events if event.type.startswith("town_world_")]
    categories = {event.payload["category"] for event in world_events}

    assert len(world_events) >= 13
    assert categories >= {
        "mayor_speech",
        "policy",
        "world_news",
        "accident",
        "weather",
        "natural_disaster",
        "town_announcement",
        "unexpected_event",
    }
    if "news" in categories:
        assert any(
            incident["rule_id"] == "flash_flood" and incident["status"] == "resolved"
            for incident in view.state["world"]["active_incidents"]
        )
    assert any(event.type == "town_incident_response" for event in events)
    assert any(event.type == "town_sheltered" for event in events)
    assert view.state["world"]["reactions"]
    assert "_event_schedule" not in view.state
    assert "_knowledge" not in view.state


def test_causal_director_records_causes_and_varies_event_trajectory_by_seed() -> None:
    trajectories = []
    for seed in (3, 23):
        engine = build_default_engine()
        view = engine.create_match("agent_town", seed=seed, mode="agent_coop")
        world_events = [
            event
            for event in engine.get(view.id).events
            if event.type.startswith("town_world_")
        ]
        assert all(event.payload.get("rule_id") for event in world_events)
        assert all(isinstance(event.payload.get("causes"), list) for event in world_events)
        assert view.state["world"]["causal_edges"]
        trajectories.append(
            [
                (
                    event.payload["rule_id"],
                    event.payload.get("headline"),
                    event.seq,
                )
                for event in world_events
            ]
        )
    assert trajectories[0] != trajectories[1]


def test_unobserved_local_event_does_not_leak_through_history_or_outcome_memory() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=23)
    match = engine.get(view.id)
    accident_rule = next(
        rule
        for rule in match.mod.event_director.RULES
        if rule.id == "workshop_accident"
    )
    match.mod.event_director.RULES = (replace(accident_rule, probability=1.0),)
    match.state["_causal_state"]["facts"]["maintenance_risk"] = 0.95
    match.state["minute_of_day"] = 9 * 60
    match.state["time"] = "09:00"
    for resident in match.state["residents"].values():
        resident.update({"location": "inn", "x": 460, "y": 520})
    view = engine.submit(view.id, view.human_player_id, _wait(view))

    accident = next(
        event for event in match.events if event.payload.get("category") == "accident"
    )
    assert accident.payload["visible_to"] == []
    assert all(
        accident not in engine._events_for_agent(match, agent_id)
        for agent_id in match.agent_brains
    )
    assert all(
        "town_world_local_accident" not in memory.outcome_events
        for brain in match.agent_brains.values()
        for memory in brain.memory
    )
    assert all(
        "world" not in event.payload.get("changed_state_keys", [])
        for event in match.events
        if event.type == "action_applied"
    )
