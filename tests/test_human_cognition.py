from hva_engine.engine import build_default_engine
from hva_engine.human_cognition import (
    DecisionMode,
    MemorySystem,
    PlanState,
    SocialBelief,
    appraise,
    select_decision_mode,
)
from hva_engine.models import MatchMode


def test_memory_retrieval_combines_relevance_importance_and_recency() -> None:
    memory = MemorySystem(reflection_threshold=10)
    old_relevant = memory.record(
        turn=1,
        content="A boundary stopped a hostile identity interview",
        action="set_boundary",
        outcome_events=["pressure_reduced"],
        score_delta=0.8,
        surprise=0.8,
        emotional_intensity=0.9,
        tags=("identity", "hostile"),
    )
    memory.record(
        turn=9,
        content="A routine racing pit stop",
        action="pit",
        outcome_events=["lap_progress"],
        score_delta=0.0,
        surprise=0.0,
        emotional_intensity=0.0,
        tags=("racing",),
    )
    retrieved = memory.retrieve(
        "hostile identity boundary", current_turn=10, mood_valence=-0.4, limit=1
    )
    assert retrieved[0]["id"] == old_relevant.id
    assert retrieved[0]["retrieval_score"] > 0


def test_reflection_requires_threshold_and_memory_evidence() -> None:
    memory = MemorySystem(reflection_threshold=0.9)
    memory.record(
        turn=1,
        content="First failed counterattack",
        action="counterattack",
        outcome_events=["trust_lost"],
        score_delta=-0.5,
        surprise=0.8,
        emotional_intensity=0.7,
    )
    assert memory.maybe_reflect(1) is None
    memory.record(
        turn=2,
        content="Second failed counterattack",
        action="counterattack",
        outcome_events=["pressure_increased"],
        score_delta=-0.4,
        surprise=0.6,
        emotional_intensity=0.8,
    )
    reflection = memory.maybe_reflect(2)
    assert reflection is not None
    assert len(reflection.evidence_memory_ids) >= 2
    assert "revisable belief" in reflection.belief


def test_appraisal_drives_coping_without_directly_selecting_an_action() -> None:
    appraisal = appraise(
        score_delta=-0.4,
        margin=-0.8,
        surprise=0.7,
        mod_signals={"stress": 0.25, "anger": 0.2},
        hostile_severity=0.95,
        uncertainty=0.65,
    )
    assert appraisal.social_threat > 0.7
    assert appraisal.goal_congruence < 0.5
    assert appraisal.coping in {"assert_boundary", "protect_self"}


def test_social_beliefs_and_plans_update_gradually() -> None:
    belief = SocialBelief()
    belief.update(hostile_severity=0.9, cooperative_signal=0.0, observed=True)
    first = belief.public_view()
    belief.update(hostile_severity=0.9, cooperative_signal=0.0, observed=True)
    second = belief.public_view()
    assert second["perceived_hostility"] > first["perceived_hostility"]
    assert second["trust"] < first["trust"]
    assert second["predicted_intent"] in {"test_boundaries", "escalate"}

    plan = PlanState(goal="build_position", age=2)
    assert not plan.update(
        desired_goal="build_position", surprise=0.1, stress=0.3, goal_congruence=0.7
    )
    assert plan.age == 3
    assert plan.update(
        desired_goal="reduce_exposure", surprise=0.9, stress=0.5, goal_congruence=0.3
    )
    assert plan.last_replan_reason == "goal_incongruence"


def test_decision_mode_represents_dual_process_bounded_rationality() -> None:
    assert (
        select_decision_mode(stress=0.9, uncertainty=0.7, stakes=0.9)
        == DecisionMode.HABITUAL
    )
    assert (
        select_decision_mode(stress=0.2, uncertainty=0.8, stakes=0.8)
        == DecisionMode.DELIBERATIVE
    )


def test_engine_emits_research_grounded_human_likeness_evidence() -> None:
    engine = build_default_engine()
    view = engine.create_match("debate_arena", seed=19, mode=MatchMode.AGENT_VS_AGENT)
    decisions = [event for event in view.events if event.type == "agent_decision"]
    assert decisions
    assert any(event.payload["retrieved_memory_ids"] for event in decisions[2:])
    assert all(event.payload["appraisal"] for event in decisions)
    assert all(event.payload["current_plan"] for event in decisions)
    assert all(event.payload["activated_traits"] for event in decisions)
    assert all("expression_gap" in event.payload["response_plan"] for event in decisions)
    components = engine.evaluation(view.id)["ai_capability_profile"][
        "human_likeness_components"
    ]
    assert components["memory_retrieval_grounding"] > 0
    assert components["appraisal_emotion_coherence"] > 0
    assert components["situation_trait_activation"] == 1.0
