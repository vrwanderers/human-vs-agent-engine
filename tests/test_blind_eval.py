from hva_engine.blind_eval import BlindSample, InMemoryBlindEvaluationStore


def test_blind_trials_hide_conditions_and_aggregate_human_ratings() -> None:
    store = InMemoryBlindEvaluationStore()
    trial = store.create_trial(
        "town-naturalness-v1",
        BlindSample("baseline", ("A baseline line",), {"provider": "none"}),
        BlindSample("real_llm", ("An LLM line",), {"provider": "secret-provider"}),
        seed=7,
    )
    assert all(
        "condition_id" not in sample and "provider" not in sample["metadata"]
        for sample in trial["samples"].values()
    )
    store.submit_rating(
        {
            "study_id": "town-naturalness-v1",
            "trial_id": trial["trial_id"],
            "rater_id": "human-rater-01",
            "preferred": "A",
            "a_naturalness": 6,
            "b_naturalness": 4,
            "a_identity_consistency": 6,
            "b_identity_consistency": 4,
            "a_contextual_fit": 5,
            "b_contextual_fit": 4,
            "a_dramatic_interest": 6,
            "b_dramatic_interest": 3,
            "notes": "A felt less templated",
        }
    )
    summary = store.summary("town-naturalness-v1")
    assert summary["ratings"] == 1
    assert summary["calibration_status"] == "insufficient_human_ratings"
    assert sum(
        condition["preference_wins"] for condition in summary["conditions"].values()
    ) == 1
