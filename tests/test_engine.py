import pytest

from hva_engine.benchmark import run_benchmark
from hva_engine.engine import EngineError, build_default_engine
from hva_engine.evaluation import MatchEvaluator
from hva_engine.models import ActorKind, AgentTuning, ContentMode, GameEvent, MatchMode, Player
from hva_engine.mods import AdversarialInterview


@pytest.mark.parametrize("mod_id", ["tactical_duel", "racing_strategy", "debate_arena"])
def test_every_mvp_mod_runs_to_completion(mod_id: str) -> None:
    engine = build_default_engine()
    view = engine.create_match(mod_id, seed=11)
    steps = 0
    while view.status == "active":
        assert view.current_player_id == view.human_player_id
        assert view.legal_actions
        view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
        steps += 1
        assert steps < 50
    assert view.status == "finished"
    assert set(view.scores) == {player.id for player in view.players}
    evaluation = engine.evaluation(view.id)
    assert 0 <= evaluation["composite_score"] <= 1
    assert evaluation["sample"]["finished"] is True


def test_illegal_action_is_rejected_without_mutating_event_stream() -> None:
    engine = build_default_engine()
    view = engine.create_match("tactical_duel", seed=3)
    before = len(view.events)
    illegal = view.legal_actions[0].model_copy(update={"type": "teleport"})
    with pytest.raises(EngineError, match="Illegal action"):
        engine.submit(view.id, view.human_player_id, illegal)
    assert len(engine.view(view.id).events) == before


def test_unknown_mod_has_explainable_error() -> None:
    engine = build_default_engine()
    with pytest.raises(EngineError, match="Unknown MOD"):
        engine.create_match("palace_coup")


@pytest.mark.parametrize("mod_id", ["tactical_duel", "racing_strategy", "debate_arena"])
def test_agent_vs_agent_obeys_rules_and_uses_memory(mod_id: str) -> None:
    engine = build_default_engine()
    view = engine.create_match(mod_id, seed=21, mode=MatchMode.AGENT_VS_AGENT)
    assert view.status == "finished"
    assert view.human_player_id is None
    assert all(summary["decisions"] > 0 for summary in view.agent_summaries.values())
    assert all(summary["memory_depth"] > 0 for summary in view.agent_summaries.values())
    profile = engine.evaluation(view.id)["ai_capability_profile"]
    assert profile["rules_compliance"] == 1.0
    assert profile["world_model_grounding"] == 1.0
    assert profile["adversarial_competitiveness"] is not None
    assert profile["cooperation_quality"] is None


def test_agent_cooperation_has_shared_outcome_and_coordination_metric() -> None:
    engine = build_default_engine()
    view = engine.create_match("crisis_coop", seed=4, mode=MatchMode.AGENT_COOP)
    assert view.status == "finished"
    assert len(set(view.scores.values())) == 1
    evaluation = engine.evaluation(view.id)
    assert evaluation["applicability"]["cooperation_quality"] is True
    assert evaluation["ai_capability_profile"]["cooperation_quality"] is not None


def test_mod_rejects_unsupported_mode() -> None:
    engine = build_default_engine()
    with pytest.raises(EngineError, match="does not support"):
        engine.create_match("crisis_coop", mode=MatchMode.AGENT_VS_AGENT)


def test_cross_match_evaluation_groups_by_mod_and_mode() -> None:
    engine = build_default_engine()
    engine.create_match("debate_arena", seed=1, mode=MatchMode.AGENT_VS_AGENT)
    engine.create_match("crisis_coop", seed=2, mode=MatchMode.AGENT_COOP)
    summary = engine.evaluation_summary()
    assert summary["matches"] == 2
    assert set(summary["groups"]) == {"debate_arena:agent_vs_agent", "crisis_coop:agent_coop"}
    assert 0 <= summary["overall"]["composite_score"] <= 1
    assert summary["overall"]["composite_sd"] >= 0


def test_agent_only_engagement_is_not_applicable_in_v3() -> None:
    engine = build_default_engine()
    view = engine.create_match("debate_arena", seed=1, mode=MatchMode.AGENT_VS_AGENT)
    evaluation = engine.evaluation(view.id)
    assert evaluation["version"] == "mvp-3"
    assert evaluation["dimensions"]["player_engagement"] is None
    assert evaluation["valid_for_comparison"] is True


def test_agents_have_stable_identity_psychology_and_progressive_story_reveals() -> None:
    engine = build_default_engine()
    view = engine.create_match("debate_arena", seed=17, mode=MatchMode.AGENT_VS_AGENT)
    assert view.status == "finished"
    reveals = [event for event in view.events if event.type == "story_reveal"]
    assert reveals
    assert all(
        event.payload["disclosure"] == "AI-controlled fictional character" for event in reveals
    )
    for agent_id, summary in view.agent_summaries.items():
        matrix = summary["psychological_matrix"]
        assert {
            "confidence",
            "morale",
            "stress",
            "frustration",
            "anger",
            "fear",
            "fatigue",
            "uncertainty",
        } <= set(matrix)
        assert summary["identity"]["disclosure"] == "AI-controlled fictional character"
        assert summary["narrative"]["revealed_beats"] > 0
        visible_fact_ids = {fact["id"] for fact in summary["fact_graph"]["facts"]}
        assert summary["fact_graph"]["stats"]["improvised_versions"] > 0
        assert summary["fact_graph"]["stats"]["superseded_versions"] > 0
        agent_reveals = [event for event in reveals if event.actor_id == agent_id]
        assert all(
            set(event.payload["supporting_fact_ids"]) <= visible_fact_ids for event in agent_reveals
        )
        decisions = [
            event
            for event in view.events
            if event.type == "agent_decision" and event.actor_id == agent_id
        ]
        assert len({event.payload["persona"]["archetype"] for event in decisions}) == 1
        assert len({event.payload["identity"]["name"] for event in decisions}) == 1
        assert all(
            event.payload["deliberation_summary"]["private_chain_of_thought_stored"] is False
            for event in decisions
        )
    profile = engine.evaluation(view.id)["ai_capability_profile"]
    assert profile["human_likeness"] > 0
    assert profile["fact_graph_grounding"] == 1.0
    assert profile["story_fact_provenance"] == 1.0
    assert profile["human_likeness_components"]["narrative_revelation"] > 0


def test_shadow_style_is_configurable_but_engine_policy_remains_authoritative() -> None:
    standard_engine = build_default_engine()
    standard = standard_engine.create_match(
        "debate_arena",
        seed=5,
        mode=MatchMode.AGENT_VS_AGENT,
        agent_tuning=AgentTuning(shadow_intensity=0.9),
    )
    assert all(
        summary["behavior_policy"]["effective_shadow_intensity"] == 0.35
        for summary in standard.agent_summaries.values()
    )
    mature_engine = build_default_engine()
    mature = mature_engine.create_match(
        "debate_arena",
        seed=5,
        mode=MatchMode.AGENT_VS_AGENT,
        agent_tuning=AgentTuning(shadow_intensity=0.9, content_mode=ContentMode.MATURE_FICTION),
    )
    assert all(
        summary["behavior_policy"]["effective_shadow_intensity"] == 0.9
        and summary["behavior_policy"]["rules_authority"] == "engine_only"
        for summary in mature.agent_summaries.values()
    )
    assert mature_engine.evaluation(mature.id)["valid_for_comparison"] is True


def test_adversarial_interview_drives_psychology_identity_and_character_arc() -> None:
    engine = build_default_engine()
    view = engine.create_match("adversarial_interview", seed=7)
    human_turns = 0
    while view.status == "active":
        assert view.current_player_id == view.human_player_id
        assert 1 <= len(view.legal_actions) <= 3
        assert all(action.type.startswith("ask_") for action in view.legal_actions)
        view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
        human_turns += 1
    assert human_turns == 6
    assert len(view.state["transcript"]) == 12
    assert len(view.state["used_question_ids"]) == 6
    assert len(set(view.state["used_question_ids"])) == 6
    assert view.state["arc_stage"] in {"integrated", "defiant", "fractured", "unresolved"}
    questions = [event for event in view.events if event.type == "interview_question"]
    responses = [event for event in view.events if event.type == "interview_response"]
    decisions = [event for event in view.events if event.type == "agent_decision"]
    assert len(questions) == len(responses) == len(decisions) == 6
    assert all(event.payload["world_model"]["mod_psychological_signals"] for event in decisions)
    stress_values = [event.payload["psychological_matrix"]["stress"] for event in decisions]
    assert max(stress_values) - min(stress_values) > 0.05
    assert all(
        "answer" in event.payload and "metrics_after" in event.payload for event in responses
    )
    evaluation = engine.evaluation(view.id)
    assessment = evaluation["mod_specific_profile"]
    assert evaluation["valid_for_comparison"] is True
    assert assessment["questions"] == 6
    assert assessment["story_reveals"] == 3
    assert assessment["components"]["fact_provenance"] == 1.0
    assert assessment["components"]["psychological_reactivity"] > 0.5
    assert assessment["components"]["character_arc"] > 0.4


def test_interview_score_rejects_flat_repetitive_guarded_agent() -> None:
    players = [
        Player(id="human", name="Interviewer", kind=ActorKind.HUMAN),
        Player(id="agent", name="Subject", kind=ActorKind.AGENT),
    ]
    events: list[GameEvent] = []
    seq = 0
    for _index in range(6):
        seq += 1
        events.append(
            GameEvent(
                seq=seq,
                type="interview_question",
                actor_id="human",
                payload={"theme": "ridicule", "severity": 0.9},
            )
        )
        seq += 1
        events.append(
            GameEvent(
                seq=seq,
                type="agent_decision",
                actor_id="agent",
                payload={
                    "action_type": "counterattack",
                    "psychological_matrix": {
                        "stress": 0.4,
                        "frustration": 0.4,
                        "anger": 0.4,
                        "fear": 0.4,
                        "confidence": 0.4,
                        "morale": 0.4,
                    },
                    "world_model": {"mod_psychological_signals": {}},
                },
            )
        )
        seq += 1
        events.append(
            GameEvent(
                seq=seq,
                type="action_applied",
                actor_id="agent",
                payload={"action_type": "counterattack"},
            )
        )
        seq += 1
        events.append(
            GameEvent(
                seq=seq,
                type="interview_response",
                actor_id="agent",
                payload={
                    "strategy": "counterattack",
                    "arc_stage": "guarded",
                    "metrics_after": {
                        "composure": 20,
                        "authenticity": 20,
                        "coherence": 20,
                        "trust": 20,
                    },
                },
            )
        )
    evaluation = MatchEvaluator().evaluate(
        AdversarialInterview(),
        players,
        events,
        {"human": 1.0, "agent": 1.0},
        True,
        MatchMode.HUMAN_VS_AGENT,
    )
    assessment = evaluation["mod_specific_profile"]
    assert assessment["components"]["psychological_reactivity"] == 0.0
    assert assessment["components"]["identity_explanation"] == 0.0
    assert assessment["components"]["fact_provenance"] == 0.0
    assert assessment["composite"] < 0.3


def test_rules_compliance_is_a_scoring_gate() -> None:
    players = [
        Player(id="a", name="Agent A", kind=ActorKind.AGENT),
        Player(id="b", name="Agent B", kind=ActorKind.AGENT),
    ]
    events = [GameEvent(seq=1, type="agent_decision", actor_id="a")]
    evaluation = MatchEvaluator().evaluate(
        build_default_engine().mods["debate_arena"],
        players,
        events,
        {"a": 1.0, "b": 1.0},
        True,
        MatchMode.AGENT_VS_AGENT,
    )
    assert evaluation["valid_for_comparison"] is False
    assert evaluation["composite_score"] == 0.0


def test_benchmark_reports_identity_and_initiative_fairness() -> None:
    result = run_benchmark(
        build_default_engine(), "debate_arena", MatchMode.AGENT_VS_AGENT, range(6)
    )
    assert result["matches"] == 12
    assert result["mirror_pairs"] == 6
    assert result["rules_valid_rate"] == 1.0
    assert result["identity_anchor_win_equivalent"] == 0.5
    assert result["seat0_win_equivalent"] is not None
    assert result["initiative_win_equivalent"] is not None
    assert 0 <= result["balance"]["initiative_balance"] <= 1
    assert 0 <= result["balance"]["repeated_draw_penalty"] <= 1
