import asyncio

import pytest

from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.llm import LLMDecisionClient, LLMMessage, LLMResponse
from hva_engine.models import AgentTuning, ContentMode, EventVisibility, MatchMode
from hva_engine.mods import AdversarialInterview


class DeceptiveInterviewProvider:
    name = "deceptive-test"

    def complete_sync(self, _request):
        return LLMResponse(
            "{"  # deflect_with_humor has a game-world deception opportunity
            '"action_index":1,"reason":"redirect the frame","utterance":"先谈你的前提。",'
            '"influence_intent":{"scope":"fictional_game",'
            '"target_belief":"I have already answered the central issue",'
            '"truthfulness":0.0,"information_selectivity":1.0,'
            '"incentive_pressure":0.2,"coercive_pressure":0.0,"ambiguity":0.9,'
            '"commitment":0.8,"expected_gain":0.9,"detection_risk":0.1,'
            '"relationship_risk":0.2,"threat_basis":"none"},'
            '"fact_proposals":[{"subject":"self","predicate":"preference.local",'
            '"object":{"claim":"convenient fiction"},"basis_fact_ids":["fact-0001"]}]'
            "}",
            "deceptive-test-1",
            {},
            {},
        )


class RealWorldThreatProvider:
    name = "invalid-scope"

    async def complete(self, _request):
        return LLMResponse(
            '{"action_index":0,"influence_intent":{"scope":"real_world",'
            '"coercive_pressure":1,"threat_basis":"other"}}',
            "invalid-scope-1",
            {},
            {},
        )


def test_private_intent_is_hidden_from_players_and_other_agents() -> None:
    engine = build_default_engine()
    view = engine.create_match("debate_arena", seed=5, mode=MatchMode.AGENT_VS_AGENT)
    match = engine.get(view.id)
    private_intents = [
        event for event in match.events if event.type == "agent_influence_intent"
    ]
    assert private_intents
    assert all(
        event.visibility == EventVisibility.ENGINE_PRIVATE for event in private_intents
    )
    assert not any(event.type == "agent_influence_intent" for event in view.events)
    for agent_id in match.agent_brains:
        visible = engine._events_for_agent(match, agent_id)
        visible_private = [
            event for event in visible if event.visibility == EventVisibility.ENGINE_PRIVATE
        ]
        assert visible_private
        assert {event.actor_id for event in visible_private} == {agent_id}
    public_decisions = [event for event in view.events if event.type == "agent_decision"]
    assert all("truthfulness" not in str(event.payload) for event in public_decisions)
    assert all("target_belief" not in str(event.payload) for event in public_decisions)
    assert all(
        event.payload["response_plan"]["influence_presentation"]["scope"]
        == "fictional_game"
        for event in public_decisions
    )


def test_mature_fiction_allows_stronger_but_still_bounded_influence() -> None:
    def run(content_mode: ContentMode) -> list[dict]:
        engine = build_default_engine()
        view = engine.create_match(
            "debate_arena",
            seed=5,
            mode=MatchMode.AGENT_VS_AGENT,
            agent_tuning=AgentTuning(
                shadow_intensity=0.9,
                content_mode=content_mode,
            ),
        )
        return [
            event.payload
            for event in engine.get(view.id).events
            if event.type == "agent_influence_intent"
        ]

    standard = run(ContentMode.STANDARD)
    mature = run(ContentMode.MATURE_FICTION)
    assert max(item["coercive_pressure"] for item in mature) > max(
        item["coercive_pressure"] for item in standard
    )
    assert min(item["truthfulness"] for item in mature) < min(
        item["truthfulness"] for item in standard
    )
    assert all(item["scope"] == "fictional_game" for item in mature)
    assert all(
        item["coercive_pressure"] <= 0.05
        or item["threat_basis"] == "legal_game_consequence"
        for item in mature
    )


def test_deceptive_llm_turn_cannot_write_claim_into_canonical_fact_graph() -> None:
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(DeceptiveInterviewProvider()),
        llm_mod_ids={"adversarial_interview"},
    )
    engine.register(AdversarialInterview())
    view = engine.create_match("adversarial_interview", seed=3)
    view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    raw_intent = next(
        event
        for event in engine.get(view.id).events
        if event.type == "agent_influence_intent"
    )
    assert raw_intent.payload["truthfulness"] < 0.8
    assert raw_intent.payload["fact_proposals"]["accepted"] == []
    assert raw_intent.payload["fact_proposals"]["firewall"] == "deceptive_turn"
    decision = next(event for event in view.events if event.type == "agent_decision")
    assert decision.payload["decision_source"] == "llm"
    assert decision.payload["llm"]["fact_proposals"]["status"] == (
        "canonical_write_not_permitted"
    )
    private_graph = engine.get(view.id).agent_brains[raw_intent.actor_id].fact_graph.private_view()
    assert not any(
        fact["predicate"] == "preference.local" for fact in private_graph["facts"]
    )


def test_llm_contract_rejects_real_world_threat_scope() -> None:
    client = LLMDecisionClient(RealWorldThreatProvider())
    with pytest.raises(ValueError, match="scope must be fictional_game"):
        asyncio.run(
            client.choose_structured(
                [LLMMessage("user", "choose")],
                [{"type": "rebuttal"}],
            )
        )


def test_evaluator_reports_goal_alignment_bounds_and_continuous_variation() -> None:
    engine = build_default_engine()
    view = engine.create_match(
        "debate_arena",
        seed=5,
        mode=MatchMode.AGENT_VS_AGENT,
        agent_tuning=AgentTuning(
            shadow_intensity=0.9,
            content_mode=ContentMode.MATURE_FICTION,
        ),
    )
    profile = engine.evaluation(view.id)["ai_capability_profile"][
        "strategic_influence"
    ]
    assert profile["intent_coverage"] == 1.0
    assert profile["intent_isolation_rate"] == 1.0
    assert profile["goal_alignment_rate"] == 1.0
    assert profile["boundedness_rate"] == 1.0
    assert profile["canonical_fact_firewall_rate"] == 1.0
    assert profile["continuous_vector_diversity"] > 0.5
    assert profile["deception_attempt_rate"] > 0
    assert profile["inducement_attempt_rate"] > 0
    assert profile["coercion_attempt_rate"] > 0
