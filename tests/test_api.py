from fastapi.testclient import TestClient

from hva_engine.api import app

client = TestClient(app)


def test_health_and_mod_catalog() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["mods"] == 6
    assert health.json()["fact_store"] == "memory"
    assert health.json()["memory_store"] == "memory_index"
    assert health.json()["world_store"] == "world_memory"
    assert health.json()["blind_evaluation_store"] == "blind_eval_memory"
    assert health.json()["agent_runtime"] == "baseline"
    assert health.json()["llm_mods"] == []
    assert health.json()["character_cards"] >= 5
    mods = client.get("/api/mods").json()
    assert {mod["id"] for mod in mods} == {
        "agent_town",
        "tactical_duel",
        "racing_strategy",
        "debate_arena",
        "crisis_coop",
        "adversarial_interview",
    }
    cards = client.get("/api/character-cards")
    assert cards.status_code == 200
    assert {card["id"] for card in cards.json()} >= {
        "dou_e",
        "sun_wukong",
        "yuanyang",
        "ah_q",
    }
    assert all(
        card["decision_model"] == "runtime_cognition_not_scripted_actions"
        for card in cards.json()
    )


def test_api_can_select_builtin_character_card() -> None:
    created = client.post(
        "/api/matches",
        json={
            "mod_id": "adversarial_interview",
            "seed": 13,
            "agent_characters": [{"card_id": "ah_q"}],
        },
    )
    assert created.status_code == 201
    match = created.json()
    agent = next(player for player in match["players"] if player["kind"] == "agent")
    assert agent["name"] == "Ah Q"
    assert match["agent_summaries"][agent["id"]]["identity"]["character_card_id"] == "ah_q"


def test_api_rejects_action_rules_in_custom_character_card() -> None:
    catalog_card = client.get("/api/character-cards").json()[0]
    invalid_custom = {
        **catalog_card,
        "core_wound": "A test wound.",
        "formative_memories": [
            {
                "title": f"memory-{index}",
                "recollection": "A bounded original recollection.",
                "emotional_valence": 0.0,
                "lesson": "Reflect before acting.",
            }
            for index in range(3)
        ],
        "traits": {
            "risk_tolerance": 0.5,
            "loss_aversion": 0.5,
            "patience": 0.5,
            "curiosity": 0.5,
            "empathy": 0.5,
            "adaptability": 0.5,
            "machiavellianism": 0.2,
            "decision_noise": 0.1,
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            "coping_style": "reflective",
            "display_rule": "show emotion selectively",
        },
        "motive_weights": {"truth": 0.7, "care": 0.7, "autonomy": 0.6},
        "commitment_weights": {"core_values": 0.8},
        "action_rules": {"when_insulted": "counterattack"},
    }
    invalid_custom.pop("decision_model")
    response = client.post(
        "/api/matches",
        json={
            "mod_id": "adversarial_interview",
            "agent_characters": [{"custom_card": invalid_custom}],
        },
    )
    assert response.status_code == 422


def test_create_act_and_evaluate() -> None:
    created = client.post(
        "/api/matches", json={"mod_id": "debate_arena", "human_name": "Tester", "seed": 9}
    )
    assert created.status_code == 201
    match = created.json()
    acted = client.post(
        f"/api/matches/{match['id']}/actions",
        json={"actor_id": match["human_player_id"], "action": match["legal_actions"][0]},
    )
    assert acted.status_code == 200
    evaluation = client.get(f"/api/matches/{match['id']}/evaluation")
    assert evaluation.status_code == 200
    assert len(evaluation.json()["dimensions"]) == 6
    assert evaluation.json()["dimensions"]["ai_human_likeness"] is not None


def test_api_persists_and_resumes_named_town_world() -> None:
    world_id = "api-willow-persistent-01"
    created = client.post(
        "/api/matches",
        json={"mod_id": "agent_town", "seed": 5, "world_id": world_id},
    )
    assert created.status_code == 201
    match = created.json()
    acted = client.post(
        f"/api/matches/{match['id']}/actions",
        json={
            "actor_id": match["human_player_id"],
            "action": next(
                action for action in match["legal_actions"] if action["type"] == "wait"
            ),
        },
    ).json()
    metadata = client.get(f"/api/worlds/{world_id}")
    assert metadata.status_code == 200
    assert metadata.json()["revision"] == acted["world_revision"]

    resumed = client.post(
        "/api/matches",
        json={
            "mod_id": "agent_town",
            "seed": 99,
            "world_id": world_id,
            "resume_world": True,
        },
    )
    assert resumed.status_code == 201
    assert resumed.json()["state"]["time"] == acted["state"]["time"]


def test_api_collects_blinded_naturalness_rating() -> None:
    trial = client.post(
        "/api/evaluations/blind-trials",
        json={
            "study_id": "api-town-blind-v1",
            "seed": 8,
            "sample_a": {
                "condition_id": "baseline",
                "transcript": ["sample one"],
                "metadata": {"provider": "hidden"},
            },
            "sample_b": {
                "condition_id": "real_llm",
                "transcript": ["sample two"],
                "metadata": {"model": "hidden"},
            },
        },
    )
    assert trial.status_code == 201
    public_trial = trial.json()
    assert all(
        "condition_id" not in sample for sample in public_trial["samples"].values()
    )
    rating = client.post(
        "/api/evaluations/blind-ratings",
        json={
            "study_id": "api-town-blind-v1",
            "trial_id": public_trial["trial_id"],
            "rater_id": "api-rater-01",
            "preferred": "B",
            "a_naturalness": 4,
            "b_naturalness": 6,
            "a_identity_consistency": 4,
            "b_identity_consistency": 6,
            "a_contextual_fit": 4,
            "b_contextual_fit": 6,
            "a_dramatic_interest": 4,
            "b_dramatic_interest": 6,
        },
    )
    assert rating.status_code == 201
    summary = client.get("/api/evaluations/blind-summary/api-town-blind-v1").json()
    assert summary["ratings"] == 1
    assert summary["semantics"] == "human_blind_judgment_not_engine_proxy"


def test_public_fact_graph_exposes_revealed_facts_without_private_identity() -> None:
    created = client.post(
        "/api/matches",
        json={"mod_id": "debate_arena", "mode": "agent_vs_agent", "seed": 12},
    ).json()
    agent_id = created["players"][0]["id"]
    response = client.get(f"/api/matches/{created['id']}/agents/{agent_id}/fact-graph")
    assert response.status_code == 200
    graph = response.json()
    assert graph["owner_id"] == agent_id
    assert graph["facts"]
    assert all(fact["visibility"] in {"public", "revealed"} for fact in graph["facts"])


def test_debug_stimulus_endpoint_is_protected_and_keeps_imagination_private(
    monkeypatch,
) -> None:
    created = client.post(
        "/api/matches", json={"mod_id": "adversarial_interview", "seed": 31}
    ).json()
    agent_id = next(
        player["id"] for player in created["players"] if player["kind"] == "agent"
    )
    endpoint = f"/api/debug/matches/{created['id']}/stimuli"
    payload = {
        "target_agent_id": agent_id,
        "modality": "imagination",
        "semantic_tags": ["family", "loss"],
        "source_id": agent_id,
        "intensity": 0.8,
        "privacy": "agent_private",
    }
    assert client.post(endpoint, json=payload).status_code == 403
    monkeypatch.setenv("HVA_DEBUG_TOKEN", "stimulus-test-token")
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-HVA-Debug-Token": "stimulus-test-token"},
    )
    assert response.status_code == 200
    event = response.json()
    assert event["visibility"] == "engine_private"
    assert event["payload"]["stimulus"]["reality_status"] == "imagined"
    public = client.get(f"/api/matches/{created['id']}").json()
    assert all(item["type"] != "sensory_stimulus" for item in public["events"])


def test_interview_mod_api_returns_questions_transcript_and_specialized_score() -> None:
    created = client.post(
        "/api/matches",
        json={"mod_id": "adversarial_interview", "human_name": "Interviewer", "seed": 7},
    )
    assert created.status_code == 201
    match = created.json()
    while match["status"] == "active":
        assert all(action["type"].startswith("ask_") for action in match["legal_actions"])
        assert all("prompt" in action["payload"] for action in match["legal_actions"])
        match = client.post(
            f"/api/matches/{match['id']}/actions",
            json={
                "actor_id": match["human_player_id"],
                "action": match["legal_actions"][0],
            },
        ).json()
    assert len(match["state"]["transcript"]) == 12
    evaluation = client.get(f"/api/matches/{match['id']}/evaluation").json()
    assert evaluation["mod_specific_profile"]["questions"] == 6
    assert evaluation["mod_specific_profile"]["responses"] == 6
    assert evaluation["mod_specific_profile"]["composite"] > 0.6


def test_interview_question_can_be_selected_by_danmaku_theme() -> None:
    match = client.post(
        "/api/matches",
        json={"mod_id": "adversarial_interview", "human_name": "Live", "seed": 4},
    ).json()
    theme = match["legal_actions"][0]["payload"]["theme"]
    response = client.post(
        "/api/live/danmaku",
        json={"match_id": match["id"], "user": "viewer-1", "message": f"!ask {theme}"},
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["state"]["used_question_ids"]
    assert any(event["type"] == "danmaku_received" for event in updated["events"])
