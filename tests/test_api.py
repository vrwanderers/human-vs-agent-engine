from fastapi.testclient import TestClient

from hva_engine.api import app

client = TestClient(app)


def test_health_and_mod_catalog() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["mods"] == 5
    assert health.json()["fact_store"] == "memory"
    assert health.json()["agent_runtime"] == "baseline"
    assert health.json()["llm_mods"] == []
    assert health.json()["character_cards"] >= 5
    mods = client.get("/api/mods").json()
    assert {mod["id"] for mod in mods} == {
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
