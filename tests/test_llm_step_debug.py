import io
import json

from hva_engine.llm_step_debug import run_interactive_interview


def test_stdio_step_debug_requires_six_llm_answers_without_fallback() -> None:
    answer = {
        "action_index": 0,
        "reason": "answer within the canonical identity",
        "utterance": "我会回答，但不会把一句羞辱误认成完整事实。",
        "response_plan": {
            "strategy_weights": {"answer_honestly": 0.7, "set_boundary": 0.3},
            "intensity": 0.6,
            "emotional_display": "guarded",
            "stance_tags": ["direct"],
            "reveal_fact_ids": [],
        },
        "influence_intent": {
            "scope": "fictional_game",
            "target_belief": "the interviewer sees a bounded answer",
            "truthfulness": 1.0,
            "information_selectivity": 0.2,
            "incentive_pressure": 0.0,
            "coercive_pressure": 0.0,
            "ambiguity": 0.1,
            "commitment": 0.7,
            "expected_gain": 0.5,
            "detection_risk": 0.0,
            "relationship_risk": 0.1,
            "threat_basis": "none",
        },
        "fact_proposals": [],
    }
    input_stream = io.StringIO(
        "".join(json.dumps(answer, ensure_ascii=False) + "\n" for _ in range(6))
    )
    output_stream = io.StringIO()
    report = run_interactive_interview(
        input_stream=input_stream,
        output_stream=output_stream,
        context_output="summary",
    )
    events = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert sum(event["type"] == "llm_decision_request" for event in events) == 6
    assert sum(event["type"] == "step_result" for event in events) == 6
    assert events[-1]["type"] == "test_complete"
    assert report["llm_decisions"] == 6
    assert report["fallback_decisions"] == 0
    assert report["evaluation"]["valid_for_comparison"] is True
    assert all(
        step["parsed_decision"]["source"] == "llm" for step in report["steps"]
    )
    assert all(step["context_diagnostics"] for step in report["steps"])
    assert all(
        not step["context_diagnostics"]["critical_sections_truncated"]
        for step in report["steps"]
    )
    requests = [event for event in events if event["type"] == "llm_decision_request"]
    assert all(
        request["context_contract"]["source"]
        == "same_context_packet_as_remote_provider"
        for request in requests
    )
    assert all(
        request["context_contract"]["extra_brain_snapshot"] is False
        for request in requests
    )
    assert all(request["context_contract"]["context_id"] for request in requests)
    assert all(
        request["snapshot"]["compression"]["source"]
        == "same_budgeted_sections_as_provider_messages"
        for request in requests
    )


def test_stdio_step_debug_retries_a_malformed_fact_proposal() -> None:
    invalid = {
        "action_index": 0,
        "fact_proposals": [
            {
                "subject_id": "agent-001",
                "predicate": "preference.local",
                "object_json": {"style": "direct"},
                "basis_fact_ids": ["fact-0001"],
            }
        ],
    }
    valid = {
        "action_index": 0,
        "reason": "stay within the canonical identity",
        "fact_proposals": [],
    }
    input_stream = io.StringIO(
        json.dumps(invalid) + "\n"
        + "".join(json.dumps(valid) + "\n" for _ in range(6))
    )
    output_stream = io.StringIO()

    report = run_interactive_interview(
        input_stream=input_stream,
        output_stream=output_stream,
        context_output="summary",
    )

    events = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    rejected = [event for event in events if event["type"] == "llm_decision_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["step"] == 1
    assert rejected[0]["retry"] is True
    assert report["llm_decisions"] == 6
    assert report["fallback_decisions"] == 0
