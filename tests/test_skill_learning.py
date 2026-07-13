from hva_engine.memory_store import InMemoryIndexedMemoryStore
from hva_engine.skill_learning import SkillLearningSystem, SkillStage
from hva_engine.stimulus import (
    DeliberationGate,
    FastAppraisalEngine,
    TemporalBinder,
)


def _train(
    system: SkillLearningSystem,
    context: dict[str, str],
    *,
    start_turn: int,
    attempts: int = 5,
) -> None:
    for turn in range(start_turn, start_turn + attempts):
        system.record_execution(
            "navigate",
            context,
            turn=turn,
            success=True,
            surprise=0.0,
            guided=True,
        )


def test_skill_progresses_from_guided_practice_to_contextual_automaticity() -> None:
    system = SkillLearningSystem("agent-a", InMemoryIndexedMemoryStore())
    route = {"mod": "city", "route_id": "home-to-station"}

    novel = system.readiness("navigate", route, current_turn=0)
    assert novel.stage == SkillStage.NOVEL
    assert novel.guidance_required

    _train(system, route, start_turn=1)
    mastered = system.readiness("navigate", route, current_turn=6)
    assert mastered.stage == SkillStage.AUTOMATIC
    assert mastered.automatic
    assert mastered.context_attempts == 5
    assert mastered.confidence >= 0.72


def test_mastered_skill_returns_to_guidance_in_an_unfamiliar_context() -> None:
    system = SkillLearningSystem("agent-a", InMemoryIndexedMemoryStore())
    familiar = {"mod": "city", "route_id": "home-to-station"}
    unfamiliar = {"mod": "city", "route_id": "hotel-to-hospital"}
    _train(system, familiar, start_turn=1)

    transferred = system.readiness("navigate", unfamiliar, current_turn=7)
    assert transferred.stage == SkillStage.AUTOMATIC
    assert not transferred.automatic
    assert transferred.reason == "skill_known_but_context_is_new"

    _train(system, unfamiliar, start_turn=8)
    learned = system.readiness("navigate", unfamiliar, current_turn=13)
    assert learned.automatic


def test_repeated_failure_degrades_then_successful_retraining_recovers_skill() -> None:
    system = SkillLearningSystem("agent-a", InMemoryIndexedMemoryStore())
    route = {"mod": "city", "route_id": "home-to-station"}
    _train(system, route, start_turn=1)

    for turn in (6, 7):
        system.record_execution(
            "navigate",
            route,
            turn=turn,
            success=False,
            surprise=1.0,
            guided=False,
            automatic=True,
        )
    degraded = system.readiness("navigate", route, current_turn=8)
    assert degraded.stage == SkillStage.DEGRADED
    assert not degraded.automatic

    _train(system, route, start_turn=8, attempts=3)
    recovered = system.readiness("navigate", route, current_turn=11)
    assert recovered.stage == SkillStage.AUTOMATIC
    assert recovered.automatic


def test_skill_state_persists_in_owner_scoped_long_term_memory() -> None:
    store = InMemoryIndexedMemoryStore()
    first_runtime = SkillLearningSystem("agent-a", store)
    route = {"mod": "city", "route_id": "home-to-station"}
    _train(first_runtime, route, start_turn=1)

    restored_runtime = SkillLearningSystem("agent-a", store)
    restored = restored_runtime.readiness("navigate", route, current_turn=6)
    isolated = SkillLearningSystem("agent-b", store).readiness(
        "navigate", route, current_turn=6
    )
    assert restored.automatic
    assert isolated.stage == SkillStage.NOVEL


def test_gate_requires_guidance_until_candidate_is_automatic() -> None:
    gate = DeliberationGate()
    frame = TemporalBinder().bind([])
    appraisal = FastAppraisalEngine().appraise(frame, uncertainty=0.1)
    common = {
        "provider_available": True,
        "frame": frame,
        "appraisal": appraisal,
        "decision_mode": "habitual",
        "legal_action_types": ("follow_route",),
        "procedural_values": {"follow_route": 1.0},
        "previous_action": "follow_route",
        "text_interaction": False,
        "plan_revised": False,
    }

    guided = gate.evaluate(
        **common,
        skill_candidates=[
            {
                "action_index": 0,
                "skill_id": "navigate",
                "attempts": 4,
                "confidence": 0.7,
                "automatic": False,
                "preferred_by_local_policy": True,
            }
        ],
    )
    automatic = gate.evaluate(
        **common,
        skill_candidates=[
            {
                "action_index": 0,
                "skill_id": "navigate",
                "attempts": 5,
                "confidence": 0.86,
                "automatic": True,
                "preferred_by_local_policy": True,
            }
        ],
    )
    assert guided.should_deliberate
    assert "skill_not_yet_automatic" in guided.reasons
    assert not automatic.should_deliberate
    assert automatic.automatic_action_indices == (0,)


def test_gate_reconsiders_when_local_policy_prefers_a_new_skill() -> None:
    gate = DeliberationGate()
    frame = TemporalBinder().bind([])
    decision = gate.evaluate(
        provider_available=True,
        frame=frame,
        appraisal=FastAppraisalEngine().appraise(frame, uncertainty=0.1),
        decision_mode="habitual",
        legal_action_types=("move", "attack"),
        procedural_values={},
        previous_action="move",
        text_interaction=False,
        plan_revised=False,
        skill_candidates=[
            {
                "action_index": 0,
                "skill_id": "move",
                "attempts": 8,
                "confidence": 0.9,
                "automatic": True,
                "preferred_by_local_policy": False,
            },
            {
                "action_index": 1,
                "skill_id": "attack",
                "attempts": 0,
                "confidence": 0.23,
                "automatic": False,
                "preferred_by_local_policy": True,
            },
        ],
    )
    assert decision.should_deliberate
    assert decision.automatic_action_indices == (0,)
    assert "skill_not_yet_automatic" in decision.reasons
