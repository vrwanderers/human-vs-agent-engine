from fastapi.testclient import TestClient

from hva_engine.api import app

client = TestClient(app)


def test_health_and_mod_catalog() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["mods"] == 4
    assert health.json()["fact_store"] == "memory"
    mods = client.get("/api/mods").json()
    assert {mod["id"] for mod in mods} == {
        "tactical_duel",
        "racing_strategy",
        "debate_arena",
        "crisis_coop",
    }


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
