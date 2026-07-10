import asyncio

import pytest

from hva_engine.context import ContextComposer, ContextPolicy, SharedBlackboard, SharedFact
from hva_engine.llm import LLMDecisionClient, LLMMessage, LLMResponse, ProviderRegistry
from hva_engine.models import Action
from hva_engine.mods import DebateArena


def test_private_context_is_owned_and_other_agent_memory_is_not_present() -> None:
    composer = ContextComposer(ContextPolicy(memory_char_budget=320, recent_memory_items=2))
    memory_a = [
        {
            "turn": index,
            "situation": "private-alpha-secret" + ("x" * 80),
            "action": "evidence",
            "outcome_events": ["audience_shift"],
        }
        for index in range(8)
    ]
    packet = composer.compose(
        match_id="m1",
        agent_id="agent-a",
        role="coop",
        mod=DebateArena(),
        state={"turn": 8},
        world_model={"objective": "shared_success"},
        memory=memory_a,
        legal_actions=[Action(type="evidence")],
        shared_facts=[SharedFact(1, "agent-b", "public team fact")],
    )
    rendered = "\n".join(message.content for message in packet.messages)
    assert packet.owner_agent_id == "agent-a"
    assert "public team fact" in rendered
    assert "private-beta-secret" not in rendered
    assert packet.diagnostics["memory_compressed"] is True
    assert packet.diagnostics["sharing"] == "sanitized_team_facts_only"


def test_blackboard_is_team_scoped() -> None:
    board = SharedBlackboard({"a", "b"})
    board.publish("a", "Threat decreased after stabilization", ("game_fact",))
    assert len(board.view_for("b")) == 1
    assert board.view_for("outsider") == []
    with pytest.raises(ValueError, match="team members"):
        board.publish("outsider", "inject this")


def test_layer_budget_preserves_every_header_instead_of_tail_slicing() -> None:
    policy = ContextPolicy(total_char_budget=5_000, memory_char_budget=1_200)
    packet = ContextComposer(policy).compose(
        match_id="long-match",
        agent_id="agent-a",
        role="opponent",
        mod=DebateArena(),
        state={"oversized": "state" * 1_000},
        world_model={"uncertainty": 0.4},
        memory=[{"turn": 1, "action": "evidence", "detail": "memory" * 1_000}],
        legal_actions=[Action(type="evidence")],
        shared_facts=[],
        identity={"background": "history" * 50},
        fact_graph={"facts": [{"object": "fact" * 1_000}]},
    )
    rendered = "\n".join(message.content for message in packet.messages)
    assert packet.diagnostics["char_count"] <= policy.total_char_budget
    assert packet.diagnostics["truncated_sections"]
    for layer in range(7, 15):
        assert f"[L{layer}" in rendered


class FakeProvider:
    name = "fake"

    async def complete(self, _request):
        return LLMResponse('{"action_index": 1}', "fake-1", {}, {})


def test_provider_registry_and_llm_decision_are_rule_constrained() -> None:
    registry = ProviderRegistry()
    registry.register(FakeProvider())
    client = LLMDecisionClient(registry.get("fake"))
    index, response = asyncio.run(
        client.choose_action(
            [LLMMessage("user", "choose")],
            [{"type": "a"}, {"type": "b"}],
        )
    )
    assert index == 1
    assert response.model == "fake-1"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeProvider())
