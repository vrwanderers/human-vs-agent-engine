from random import Random

from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.llm import LLMDecisionClient, LLMResponse
from hva_engine.models import GameEvent
from hva_engine.mods import TacticalDuel
from hva_engine.stimulus import (
    DeliberationGate,
    FastAppraisalEngine,
    PerceptionAdapter,
    RealityStatus,
    RealityStatusGuard,
    ReflexController,
    StimulusModality,
    StimulusPrivacy,
    TemporalBinder,
)


class CountingProvider:
    name = "counting-provider"

    def __init__(self) -> None:
        self.calls = 0

    def complete_sync(self, _request):
        self.calls += 1
        return LLMResponse(
            '{"action_index":0,"reason":"planned","fact_proposals":[]}',
            "counting-model",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {},
        )


def test_perception_adapter_normalizes_external_and_internal_modalities() -> None:
    viewer = "agent-a"
    events = [
        GameEvent(
            seq=index,
            type="sensory_stimulus",
            actor_id=viewer if modality in {"memory", "imagination", "interoception"} else "world",
            payload={
                "stimulus": {
                    "target_id": viewer,
                    "source_id": "self" if modality in {"memory", "imagination"} else "world",
                    "modality": modality,
                    "semantic_tags": [modality, "family"],
                    "intensity": 0.7,
                }
            },
        )
        for index, modality in enumerate(
            (
                "language",
                "vision",
                "audio",
                "touch",
                "interoception",
                "memory",
                "imagination",
                "world_event",
            ),
            start=1,
        )
    ]
    stimuli = PerceptionAdapter().from_game_events(events, viewer_id=viewer)
    assert {stimulus.modality for stimulus in stimuli} == set(StimulusModality)
    statuses = {stimulus.modality: stimulus.reality_status for stimulus in stimuli}
    assert statuses[StimulusModality.IMAGINATION] == RealityStatus.IMAGINED
    assert statuses[StimulusModality.MEMORY] == RealityStatus.REMEMBERED
    assert statuses[StimulusModality.WORLD_EVENT] == RealityStatus.CANONICAL
    assert all(stimulus.target_id == viewer for stimulus in stimuli)


def test_temporal_binding_and_fast_appraisal_create_bounded_compound_trigger() -> None:
    events = [
        GameEvent(
            seq=1,
            type="sensory_stimulus",
            actor_id="player",
            payload={
                "stimulus": {
                    "target_id": "agent",
                    "modality": "language",
                    "semantic_tags": ["family", "accusation"],
                    "intensity": 0.72,
                    "valence": -0.8,
                    "urgency": 0.55,
                    "causal_group": "confrontation",
                }
            },
        ),
        GameEvent(
            seq=2,
            type="sensory_stimulus",
            actor_id="world",
            payload={
                "stimulus": {
                    "target_id": "agent",
                    "modality": "vision",
                    "semantic_tags": ["family", "old_photo"],
                    "intensity": 0.66,
                    "valence": -0.5,
                    "urgency": 0.4,
                    "causal_group": "confrontation",
                }
            },
        ),
    ]
    stimuli = PerceptionAdapter().from_game_events(events, viewer_id="agent")
    frame = TemporalBinder().bind(stimuli)
    appraisal = FastAppraisalEngine().appraise(
        frame,
        identity_themes=("family",),
        sensitive_topics=("family",),
        stress=0.55,
        fear=0.4,
        uncertainty=0.3,
    )
    assert frame.modalities == (StimulusModality.LANGUAGE, StimulusModality.VISION)
    assert 0 < frame.compound_gain <= 0.28
    assert frame.intensity <= 1.0
    assert appraisal.memory_resonance > 0.3
    assert appraisal.personal_relevance > 0.5
    assert appraisal.action_readiness > 0.4

    reflex = ReflexController().respond(
        frame,
        appraisal,
        turn=1,
        conscientiousness=0.65,
        agreeableness=0.45,
        neuroticism=0.7,
        stress=0.6,
        rng=Random(4),
    )
    assert reflex.cues
    assert len(reflex.cues) <= 2
    assert reflex.observable_view()["interpretation_policy"].endswith("not_truth_or_lie_oracle")


def test_reality_guard_does_not_promote_fantasy_or_memory_to_fact() -> None:
    assert RealityStatusGuard.can_write_canonical_fact(RealityStatus.CANONICAL)
    assert not RealityStatusGuard.can_write_canonical_fact(RealityStatus.IMAGINED)
    assert not RealityStatusGuard.can_write_canonical_fact(RealityStatus.REMEMBERED)
    assert not RealityStatusGuard.can_write_canonical_fact(RealityStatus.INFERRED)


def test_deliberation_gate_escalates_social_ambiguity_but_skips_known_routine() -> None:
    adapter = PerceptionAdapter()
    social_frame = TemporalBinder().bind(
        adapter.from_game_events(
            [
                GameEvent(
                    seq=1,
                    type="interview_question",
                    actor_id="human",
                    payload={"theme": "identity", "severity": 0.9, "prompt": "Who are you?"},
                )
            ],
            viewer_id="agent",
        )
    )
    social_appraisal = FastAppraisalEngine().appraise(
        social_frame, stress=0.6, fear=0.4, uncertainty=0.65
    )
    gate = DeliberationGate()
    social = gate.evaluate(
        provider_available=True,
        frame=social_frame,
        appraisal=social_appraisal,
        decision_mode="deliberative",
        legal_action_types=("answer", "set_boundary"),
        procedural_values={},
        previous_action=None,
        text_interaction=True,
        plan_revised=True,
    )
    routine = gate.evaluate(
        provider_available=True,
        frame=TemporalBinder().bind([]),
        appraisal=FastAppraisalEngine().appraise(
            TemporalBinder().bind([]), uncertainty=0.1
        ),
        decision_mode="habitual",
        legal_action_types=("follow_route",),
        procedural_values={"follow_route": 0.7},
        previous_action="follow_route",
        text_interaction=False,
        plan_revised=False,
    )
    assert social.should_deliberate
    assert "social_reply_required" in social.reasons
    assert not routine.should_deliberate
    assert "handled_by_reflex_or_routine" in routine.reasons


def test_private_imagination_affects_reflex_but_is_not_public_or_canonical() -> None:
    engine = build_default_engine()
    view = engine.create_match("adversarial_interview", seed=12)
    agent_id = next(player.id for player in view.players if player.id != view.human_player_id)
    stimulus = engine.publish_stimulus(
        view.id,
        target_agent_id=agent_id,
        modality=StimulusModality.IMAGINATION,
        semantic_tags=("family", "loss"),
        source_id=agent_id,
        intensity=0.8,
        valence=-0.8,
        urgency=0.55,
        privacy=StimulusPrivacy.AGENT_PRIVATE,
    )
    assert stimulus.visibility == "engine_private"
    assert all(event.type != "sensory_stimulus" for event in engine.view(view.id).events)

    view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    events = engine.get(view.id).events
    decision = next(event for event in events if event.type == "agent_decision")
    assert "imagination" in decision.payload["stimulus_frame"]["modalities"]
    diagnostic = next(event for event in events if event.type == "agent_reflex_diagnostic")
    assert "imagined" in diagnostic.payload["stimulus_frame"]["reality_statuses"]
    cues = [event for event in events if event.type == "agent_involuntary_cue"]
    assert cues
    assert cues[0].seq < decision.seq
    all_facts = engine.get(view.id).agent_brains[agent_id].fact_graph.private_view()["facts"]
    assert all("imagined" not in str(fact).lower() for fact in all_facts)
    profile = engine.evaluation(view.id)["ai_capability_profile"]["implicit_control"]
    assert profile["epistemic_boundary_rate"] == 1.0
    assert profile["cue_before_deliberation_rate"] == 1.0
    assert "imagination" in profile["modalities_observed"]


def test_action_heavy_mod_uses_provider_only_when_deliberation_gate_requires_it() -> None:
    provider = CountingProvider()
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider),
        llm_mod_ids={"tactical_duel"},
    )
    engine.register(TacticalDuel())
    view = engine.create_match("tactical_duel", seed=22)
    while view.status == "active":
        view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    decisions = [
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    ]
    assert 0 < provider.calls < len(decisions)
    assert provider.calls >= 5
    assert any(event.payload["llm_skip"] for event in decisions)
    assert all(event.payload["deliberation_gate"] for event in decisions)
    assert all(event.payload["selected_skill"] for event in decisions)
    assert all(event.payload["skill_after_outcome"] for event in decisions)
    assert any(
        event.payload["selected_skill"]["automatic"] for event in decisions
    )
    evaluation = engine.evaluation(view.id)
    assert evaluation["ai_capability_profile"]["rules_compliance"] == 1.0
    implicit = evaluation["ai_capability_profile"]["implicit_control"]
    assert implicit["deliberation_gate_coverage"] == 1.0
    assert implicit["provider_skip_rate"] > 0
    assert evaluation["provider_execution"]["skipped_decisions"] > 0
