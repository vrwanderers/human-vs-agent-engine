from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hva_engine.api import app
from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.llm import LLMDecisionClient, LLMResponse
from hva_engine.models import AgentCharacterSelection
from hva_engine.mods import AdversarialInterview, DebateArena


class CaptureProvider:
    name = "capture-provider"

    def __init__(self, answers: list[str] | None = None) -> None:
        self.requests = []
        self.answers = answers or ["我会直接回答这个问题。"]

    def complete_sync(self, request):
        self.requests.append(request)
        answer = self.answers[(len(self.requests) - 1) % len(self.answers)]
        return LLMResponse(
            json.dumps(
                {
                    "action_index": 0,
                    "reason": "respond to the current question without inventing facts",
                    "utterance": answer,
                    "fact_proposals": [],
                },
                ensure_ascii=False,
            ),
            "capture-model-1",
            {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            {},
        )


class HiddenStateDebate(DebateArena):
    id = "hidden_state_debate"

    def initial_state(self, players, rng):
        state = super().initial_state(players, rng)
        state["private_canary"] = "OTHER-PLAYER-PRIVATE-CANARY"
        return state

    def public_state(self, state, _viewer_id=None):
        visible = dict(state)
        visible.pop("private_canary", None)
        return visible


def _finish_interview(engine: GameEngine, seed: int = 23):
    view = engine.create_match(
        "adversarial_interview",
        seed=seed,
        agent_characters=[AgentCharacterSelection(card_id="dou_e")],
    )
    while view.status == "active":
        action = max(
            view.legal_actions,
            key=lambda candidate: float(candidate.payload.get("severity", 0.0)),
        )
        view = engine.submit(view.id, view.human_player_id, action)
    return view


def test_agent_receives_viewer_scoped_state_without_authoritative_canary() -> None:
    provider = CaptureProvider()
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider),
        llm_mod_ids={HiddenStateDebate.id},
        llm_fallback=False,
    )
    engine.register(HiddenStateDebate())
    view = engine.create_match(HiddenStateDebate.id, seed=5)
    assert engine.get(view.id).state["private_canary"] == "OTHER-PLAYER-PRIVATE-CANARY"
    assert "private_canary" not in view.state

    view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    rendered = "\n".join(
        message.content for request in provider.requests for message in request.messages
    )
    assert "OTHER-PLAYER-PRIVATE-CANARY" not in rendered
    decision = next(
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    )
    assert decision.payload["observation_policy"]["viewer_scoped"] is True
    assert "private_canary" not in decision.payload["observation_policy"][
        "visible_state_keys"
    ]


def test_public_view_redacts_agent_internals_and_admin_debug_requires_token(
    monkeypatch,
) -> None:
    client = TestClient(app)
    created = client.post(
        "/api/matches", json={"mod_id": "adversarial_interview", "seed": 31}
    ).json()
    acted = client.post(
        f"/api/matches/{created['id']}/actions",
        json={
            "actor_id": created["human_player_id"],
            "action": created["legal_actions"][0],
        },
    ).json()
    summary = next(iter(acted["agent_summaries"].values()))
    forbidden = {
        "psychological_matrix",
        "opponent_model",
        "social_beliefs",
        "current_plan",
        "decision_tendencies",
        "recent_memory",
        "last_retrieval",
        "world_model",
        "behavior_policy",
    }
    assert forbidden.isdisjoint(summary)
    assert not any(event["type"] == "agent_decision" for event in acted["events"])
    agent_actions = [
        event
        for event in acted["events"]
        if event["type"] == "action_applied"
        and event["actor_id"] != acted["human_player_id"]
    ]
    assert agent_actions
    assert all("response_plan" not in event["payload"]["action_payload"] for event in agent_actions)

    denied = client.get(f"/api/debug/matches/{created['id']}")
    assert denied.status_code == 403
    monkeypatch.setenv("HVA_DEBUG_TOKEN", "mvp11-test-token")
    debug = client.get(
        f"/api/debug/matches/{created['id']}",
        headers={"x-hva-debug-token": "mvp11-test-token"},
    )
    assert debug.status_code == 200
    assert debug.json()["view_scope"] == "admin_debug"
    assert "psychological_matrix" in next(iter(debug.json()["agent_summaries"].values()))


def test_mvp11_behavior_score_separates_templates_from_varied_llm_output() -> None:
    baseline_engine = build_default_engine()
    baseline = _finish_interview(baseline_engine)
    baseline_eval = baseline_engine.evaluation(baseline.id)
    reveals = [event for event in baseline.events if event.type == "story_reveal"]
    assert reveals
    causal_triggers = {
        "agent_requested_reveal",
        "pressure_or_setback",
        "trust_or_pattern_recognition",
        "commitment_or_finale",
    }
    assert all(
        event.payload["trigger"] in causal_triggers for event in reveals
    )

    answers = [
        "失败不是一枚供人观赏的勋章；我承认那次选择伤害了别人，也改变了我。",
        "你说我的身份只是表演，但我记得沉默时付出的代价，那不是一句标签能抹去的。",
        "如果我的善意里混有自保，我愿意承认矛盾，不把复杂动机装成纯洁。",
        "恐惧确实影响过判断；解释它不是免责，我仍要对当时的决定负责。",
        "这句嘲讽让我愤怒，不过愤怒不能替我作证，所以我先回答事实。",
        "我无法证明自己完全真实，只能让今天的选择与过去承担的后果彼此一致。",
    ]
    provider = CaptureProvider(answers)
    llm_engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider),
        llm_mod_ids={"adversarial_interview"},
        llm_fallback=False,
    )
    llm_engine.register(AdversarialInterview())
    llm_view = _finish_interview(llm_engine)
    llm_eval = llm_engine.evaluation(llm_view.id)

    baseline_human = baseline_eval["dimensions"]["ai_human_likeness"]
    llm_human = llm_eval["dimensions"]["ai_human_likeness"]
    assert llm_human >= baseline_human + 0.15
    assert (
        llm_eval["ai_capability_profile"]["language_behavior"]["score"]
        > baseline_eval["ai_capability_profile"]["language_behavior"]["score"]
    )
    assert llm_eval["provider_execution"]["successful_decisions"] == 6
    assert llm_eval["provider_execution"]["fallback_rate"] == 0.0
    assert llm_eval["provider_execution"]["usage"]["total_tokens"] == 780
    assert llm_eval["valid_for_provider_comparison"] is True


def test_mvp11_reports_unmeasured_human_experience_instead_of_inventing_score() -> None:
    engine = build_default_engine()
    view = engine.create_match("debate_arena", seed=9, mode="agent_vs_agent")
    evaluation = engine.evaluation(view.id)
    assert evaluation["version"] == "mvp-11"
    assert len(evaluation["config_sha256"]) == 64
    assert evaluation["score_layers"]["structural_integrity"]["score"] == 1.0
    assert evaluation["score_layers"]["player_experience"]["score"] is None
    assert evaluation["composite_semantics"] == "research_proxy_not_human_validation"
