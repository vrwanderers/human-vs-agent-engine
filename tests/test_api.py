from fastapi.testclient import TestClient

from hva_engine.api import app

client = TestClient(app)


def test_health_and_mod_catalog() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["mods"] == 4
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
    assert len(evaluation.json()["dimensions"]) == 5
