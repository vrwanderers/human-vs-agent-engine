from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_godot_project_references_complete_agent_town_scene() -> None:
    project = (ROOT / "godot/project.godot").read_text("utf-8")
    scene = (ROOT / "godot/main.tscn").read_text("utf-8")
    main = (ROOT / "godot/main.gd").read_text("utf-8")
    town = (ROOT / "godot/town_world.gd").read_text("utf-8")

    assert 'run/main_scene="res://main.tscn"' in project
    assert 'path="res://main.gd"' in scene
    assert 'preload("res://town_world.gd")' in main
    assert '"mod_id": mod_id' in main
    assert 'mod_id == "agent_town"' in main
    assert all(
        owner_id in main
        for owner_id in ("willow-astra", "willow-nova", "willow-mira")
    )
    assert "town_world_" in main
    assert "town_rumor_seeded" in main
    assert "verify_claim" in main
    assert "investigate_claim" in main
    assert "social_media" in main
    assert "active_incidents" in town
    assert "func auto_step()" in main
    assert "func draw_resident(" in town
    assert "func draw_building(" in town
    assert "signal resident_selected" in town
