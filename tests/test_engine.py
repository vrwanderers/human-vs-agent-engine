import pytest

from hva_engine.engine import EngineError, build_default_engine
from hva_engine.models import MatchMode


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
