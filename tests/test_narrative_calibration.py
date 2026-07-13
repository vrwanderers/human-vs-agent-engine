from dataclasses import replace
from random import Random

import pytest

from hva_engine.character_dynamics import NarrativeDynamics
from hva_engine.cognition import (
    AgentIdentity,
    CognitiveProfile,
    CognitiveState,
    RuntimeBehaviorPolicy,
)
from hva_engine.engine import build_default_engine
from hva_engine.human_cognition import appraise
from hva_engine.models import Action, AgentTuning
from hva_engine.narrative_calibration import (
    NarrativeCalibrationEvaluator,
    NarrativeDatasetError,
    NarrativeDecisionModel,
    load_reference_cases,
    validate_reference_dataset,
)


def test_reference_dataset_is_paraphrased_licensed_and_medium_diverse() -> None:
    metadata, cases = load_reference_cases()
    assert metadata["contains_source_text"] is False
    assert metadata["dataset_type"] == "human_authored_character_reference"
    assert {case.medium for case in cases} >= {
        "novel",
        "play",
        "film",
        "television",
        "biography",
    }
    assert {case.reference_class for case in cases} == {
        "fictional_character",
        "biographical_subject",
    }
    assert len(cases) >= 38
    assert {case.original_language for case in cases} >= {"zh", "es", "sa"}
    assert all(case.source_url.startswith("https://") for case in cases)


def test_dataset_rejects_source_text_redistribution() -> None:
    metadata, cases = load_reference_cases()
    invalid = {**metadata, "contains_source_text": True}
    with pytest.raises(NarrativeDatasetError, match="must not redistribute"):
        validate_reference_dataset(invalid, cases)


def test_decision_model_does_not_read_recorded_outcome() -> None:
    _metadata, cases = load_reference_cases()
    case = cases[0]
    model = NarrativeDecisionModel()
    original = model.predict(case)
    alternate_ground_truth = next(
        option.id for option in case.options if option.id != case.observed_option
    )
    changed_label = replace(case, observed_option=alternate_ground_truth)
    changed = model.predict(changed_label)
    assert original.option_id == changed.option_id
    assert original.option_scores == changed.option_scores


def test_narrative_mechanism_beats_simple_negative_controls() -> None:
    metadata, cases = load_reference_cases()
    report = NarrativeCalibrationEvaluator().run(
        cases,
        set(metadata["holdout_case_ids"]),
        set(metadata["holdout_work_groups"]),
    )
    assert report["calibration_status"] == "prototype_not_independently_annotated"
    assert report["not_real_human_behavior_data"] is True
    assert report["biographies_are_not_behavioral_telemetry"] is True
    assert len(report["known_limitations"]) == 3
    assert report["components"]["decision_match"] > 0.75
    assert report["discriminative_margin"] > 0.05
    assert report["components"]["decision_match"] > max(
        report["negative_controls"].values()
    )
    assert report["negative_controls"]["self_preservation_only"] < 0.5
    ablation = report["mechanism_ablation"][
        "without_temptation_social_pressure_and_rationalization"
    ]
    assert ablation["delta"] > 0
    assert "macbeth_chooses_regicide" in ablation["failures"]
    assert report["holdout"]["cases"] >= 4
    assert report["holdout"]["train_holdout_work_overlap"] == 0
    assert set(report["by_reference_class"]) == {
        "fictional_character",
        "biographical_subject",
    }
    assert report["version"] == "narrative-calibration-v4"
    assert report["multilingual_coverage"]["chinese_literature_cases"] >= 11
    assert report["multilingual_coverage"]["modern_chinese_fiction_cases"] >= 4


def test_biographical_cards_require_institutional_evidence() -> None:
    metadata, cases = load_reference_cases()
    biographical = next(
        case for case in cases if case.reference_class == "biographical_subject"
    )
    invalid_case = replace(biographical, source_form="screen_narrative")
    invalid_cases = [invalid_case if case.id == biographical.id else case for case in cases]
    with pytest.raises(NarrativeDatasetError, match="institutional evidence"):
        validate_reference_dataset(metadata, invalid_cases)


def test_slow_character_state_retains_consequences_and_changes_biases() -> None:
    rng = Random(7)
    policy = RuntimeBehaviorPolicy.from_tuning(AgentTuning())
    profile = CognitiveProfile.sample(rng, "opponent", policy)
    identity = AgentIdentity.sample("Astra", profile, "opponent", rng)
    dynamics = NarrativeDynamics.from_identity(identity, profile, False)
    cognition = CognitiveState(stress=0.72, uncertainty=0.65, morale=0.35)
    appraisal = appraise(
        score_delta=-0.5,
        margin=-0.8,
        surprise=0.8,
        mod_signals={"stress": 0.25},
        hostile_severity=0.9,
        uncertainty=0.65,
    )
    dynamics.update_before_decision(appraisal, cognition, social_trust=0.25)
    legal = [Action(type="counterattack"), Action(type="answer_honestly")]
    before = dynamics.action_biases(legal)
    dynamics.record_consequence(
        action_type="counterattack",
        score_delta=-0.7,
        surprise=0.8,
        cognition=cognition,
        profile=profile,
    )
    after = dynamics.action_biases(legal)
    assert dynamics.consequence_trace
    assert dynamics.moral_injury > 0
    assert dynamics.identity_dissonance > 0
    assert dynamics.impulse_pressure > 0
    assert dynamics.self_licensing > 0
    assert before != after


def test_match_evaluation_exposes_narrative_dynamics_without_private_reasoning() -> None:
    engine = build_default_engine()
    view = engine.create_match("adversarial_interview", seed=23)
    while view.status == "active":
        view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    decisions = [
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    ]
    assert all(event.payload["narrative_dynamics"]["active_conflict"] for event in decisions)
    assert all(event.payload["narrative_action_affordance"] for event in decisions)
    assert any(
        event.payload["narrative_dynamics"]["consequence_count"] > 0 for event in decisions[1:]
    )
    assert any(event.type == "delayed_narrative_consequence" for event in view.events)
    assert all(
        event.payload["deliberation_summary"]["private_chain_of_thought_stored"] is False
        for event in decisions
    )
    evaluation = engine.evaluation(view.id)
    components = evaluation["ai_capability_profile"]["human_likeness_components"]
    assert evaluation["version"] == "mvp-11"
    assert components["motivational_conflict"] > 0
    assert components["consequence_hysteresis"] > 0
    assert components["distortion_pressure"] > 0
    assert components["commitment_conflict"] > 0
    assert components["delayed_consequence_realization"] > 0
