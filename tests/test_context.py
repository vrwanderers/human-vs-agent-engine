import asyncio
import json

import httpx
import pytest

from hva_engine.context import ContextComposer, ContextPolicy, SharedBlackboard, SharedFact
from hva_engine.llm import (
    LLMDecisionClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    OpenAICompatibleProvider,
    ProviderRegistry,
)
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


def test_two_agent_context_packets_have_distinct_owners_and_no_canary_leakage() -> None:
    composer = ContextComposer()

    def packet(agent_id: str, private_canary: str):
        return composer.compose(
            match_id="isolation-match",
            agent_id=agent_id,
            role="opponent",
            mod=DebateArena(),
            state={"turn": 2},
            world_model={"objective": "outperform_opponents"},
            memory=[{"id": f"memory-{agent_id}", "content": private_canary}],
            legal_actions=[Action(type="evidence")],
            shared_facts=[],
            identity={"name": agent_id, "private_canary": private_canary},
        )

    packet_a = packet("agent-a", "ALPHA-PRIVATE-CANARY")
    packet_b = packet("agent-b", "BETA-PRIVATE-CANARY")
    serialized_a = json.dumps(packet_a.provider_metadata(), ensure_ascii=False)
    serialized_b = json.dumps(packet_b.provider_metadata(), ensure_ascii=False)

    assert packet_a.context_id != packet_b.context_id
    assert packet_a.owner_agent_id == "agent-a"
    assert packet_b.owner_agent_id == "agent-b"
    assert "BETA-PRIVATE-CANARY" not in serialized_a
    assert "ALPHA-PRIVATE-CANARY" not in serialized_b


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


def test_priority_budget_keeps_decision_critical_layers_auditable() -> None:
    packet = ContextComposer(ContextPolicy(total_char_budget=8_000)).compose(
        match_id="priority-match",
        agent_id="agent-a",
        role="opponent",
        mod=DebateArena(),
        state={"observation": "pressure" * 1_000},
        world_model={"objective": "hold a defensible position"},
        memory=[{"detail": "memory" * 1_000}],
        legal_actions=[Action(type="evidence"), Action(type="rebuttal")],
        shared_facts=[],
        decision_tendencies={
            "semantics": "fallible_motivational_attraction_not_action_command",
            "actions": [
                {"action_index": 0, "attraction": 0.61, "rank": 1},
                {"action_index": 1, "attraction": 0.39, "rank": 2},
            ],
        },
        appraisal={"coping": "assert_boundary"},
        current_plan={"strategic_goal": "hold a defensible position"},
    )

    diagnostics = packet.diagnostics
    assert diagnostics["decision_tendencies_layered"] is True
    assert diagnostics["section_allocations"]["legal_actions"] > 0
    assert diagnostics["section_original_chars"]["current_observation"] > 0
    assert diagnostics["section_capped_chars"]["current_observation"] <= 4_800
    assert "legal_actions" not in diagnostics["critical_sections_truncated"]


def test_provider_and_manual_bridge_share_one_budgeted_context_contract() -> None:
    packet = ContextComposer().compose(
        match_id="parity-match",
        agent_id="agent-private-a",
        role="opponent",
        mod=DebateArena(),
        state={"turn": 3},
        world_model={"objective": "defend position"},
        memory=[{"id": "memory-a", "content": "alpha private recollection"}],
        legal_actions=[Action(type="evidence"), Action(type="rebuttal")],
        shared_facts=[],
        persona={"display_rule": "mask_fear"},
        identity={"name": "Alpha", "background": "private history"},
        cognitive_state={"psychological_matrix": {"stress": 0.72}},
        fact_graph={
            "owner_id": "agent-private-a",
            "facts": [
                {
                    "id": "fact-a",
                    "predicate": "identity.name",
                    "object": "Alpha",
                    "visibility": "private",
                    "status": "active",
                }
            ],
        },
        current_plan={"strategic_goal": "defend position"},
    )

    rendered = "\n".join(message.content for message in packet.messages)
    metadata = packet.provider_metadata()
    assert metadata["context_id"] == packet.context_id
    assert metadata["content_sha256"] == packet.content_sha256
    assert metadata["owner_agent_id"] == "agent-private-a"
    assert packet.diagnostics["provider_context_source"] == (
        "same_context_packet_as_manual_bridge"
    )
    for payload in packet.decision_context["section_payloads"].values():
        assert payload in rendered


def test_remote_provider_transports_exact_messages_without_engine_debug_metadata() -> None:
    provider = OpenAICompatibleProvider(
        name="compatible",
        base_url="https://provider.invalid/v1",
        model="test-model",
    )
    request = LLMRequest(
        messages=[LLMMessage("system", "private agent A context")],
        context_metadata={"owner_agent_id": "agent-a", "debug_only": "not transported"},
    )

    payload = provider._payload(request)

    assert payload["messages"] == [
        {"role": "system", "content": "private agent A context"}
    ]
    assert "context_metadata" not in payload
    assert "debug_only" not in json.dumps(payload)


def test_openai_compatible_sync_transport_retries_429_and_records_attempts(
    monkeypatch,
) -> None:
    responses = [
        httpx.Response(429, request=httpx.Request("POST", "https://provider.test")),
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://provider.test"),
            json={
                "model": "retry-model",
                "choices": [{"message": {"content": '{"action_index":0}'}}],
                "usage": {"total_tokens": 12},
            },
        ),
    ]

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return responses.pop(0)

    monkeypatch.setattr("hva_engine.llm.httpx.Client", FakeClient)
    monkeypatch.setattr("hva_engine.llm.time.sleep", lambda _seconds: None)
    provider = OpenAICompatibleProvider(
        name="retry-test",
        base_url="https://provider.test/v1",
        model="retry-model",
        max_retries=1,
    )
    response = provider.complete_sync(
        LLMRequest(messages=[LLMMessage("user", "choose")])
    )
    assert response.content == '{"action_index":0}'
    assert response.telemetry["attempt_count"] == 2
    assert response.telemetry["transport"] == "httpx_sync_debug"


def test_decision_client_rejects_context_metadata_that_does_not_match_messages() -> None:
    client = LLMDecisionClient(FakeProvider())

    with pytest.raises(ValueError, match="metadata does not match"):
        asyncio.run(
            client.choose_action(
                [LLMMessage("user", "agent-a-context")],
                [{"type": "evidence"}, {"type": "rebuttal"}],
                context_metadata={
                    "context_id": "ctx-from-another-agent",
                    "content_sha256": "not-the-message-digest",
                    "owner_agent_id": "agent-b",
                },
            )
        )


def test_truncated_context_payload_remains_valid_json() -> None:
    rendered = ContextComposer()._valid_truncated_payload(
        json.dumps({"oversized": "context" * 200}), 240
    )
    parsed = json.loads(rendered)
    assert len(rendered) <= 240
    assert parsed["section_truncated"] is True


class FakeProvider:
    name = "fake"

    async def complete(self, _request):
        return LLMResponse('{"action_index": 1}', "fake-1", {}, {})


class InvalidPlanProvider:
    name = "invalid-plan"

    async def complete(self, _request):
        return LLMResponse(
            '{"action_index":0,"response_plan":'
            '{"strategy_weights":{"invented_illegal_strategy":1}}}',
            "fake-2",
            {},
            {},
        )


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


def test_llm_response_plan_rejects_strategies_outside_legal_actions() -> None:
    client = LLMDecisionClient(InvalidPlanProvider())
    with pytest.raises(ValueError, match="illegal strategy"):
        asyncio.run(
            client.choose_action_and_facts(
                [LLMMessage("user", "choose")],
                [{"type": "answer_honestly"}, {"type": "set_boundary"}],
            )
        )
