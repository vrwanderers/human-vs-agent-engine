from hva_engine.engine import EngineError, GameEngine
from hva_engine.mods import AgentTown
from hva_engine.world_store import InMemoryWorldStateStore, SQLiteWorldStateStore


def _wait(view):
    return next(action for action in view.legal_actions if action.type == "wait")


def test_town_world_snapshot_resumes_clock_causality_and_social_history() -> None:
    world_store = InMemoryWorldStateStore()
    engine = GameEngine(world_store=world_store)
    engine.register(AgentTown())
    owners = ["persistent-astra", "persistent-nova", "persistent-mira"]
    first = engine.create_match(
        "agent_town",
        seed=3,
        world_id="willow-persistent-world",
        agent_memory_owner_ids=owners,
    )
    for _ in range(6):
        first = engine.submit(first.id, first.human_player_id, _wait(first))
    metadata = engine.world_metadata("willow-persistent-world")
    saved_day = first.state["day"]
    saved_time = first.state["time"]
    saved_events = len(first.state["world"]["event_history"])
    saved_posts = len(first.state["social_media"]["posts"])

    resumed = engine.create_match(
        "agent_town",
        seed=97,
        world_id="willow-persistent-world",
        resume_world=True,
        agent_memory_owner_ids=owners,
    )
    assert resumed.world_revision == metadata["revision"]
    assert resumed.state["day"] == saved_day
    assert resumed.state["time"] == saved_time
    assert len(resumed.state["world"]["event_history"]) == saved_events
    assert len(resumed.state["social_media"]["posts"]) == saved_posts
    assert resumed.state["session_end_day"] == saved_day + 2
    assert any(event.type == "world_resumed" for event in resumed.events)

    resumed = engine.submit(resumed.id, resumed.human_player_id, _wait(resumed))
    assert resumed.world_revision > metadata["revision"]


def test_existing_world_requires_explicit_resume() -> None:
    engine = GameEngine(world_store=InMemoryWorldStateStore())
    engine.register(AgentTown())
    engine.create_match("agent_town", world_id="willow-no-overwrite")
    try:
        engine.create_match("agent_town", world_id="willow-no-overwrite")
    except EngineError as exc:
        assert "resume_world=true" in str(exc)
    else:
        raise AssertionError("Existing world should not be overwritten implicitly")


def test_sqlite_world_store_round_trips_json_snapshot(tmp_path) -> None:
    store = SQLiteWorldStateStore(tmp_path / "worlds.sqlite3")
    first = store.save("world-a", "agent_town", {"day": 2, "nested": {"rain": True}})
    second = store.save("world-a", "agent_town", {"day": 3, "nested": {"rain": False}})
    loaded = store.load("world-a")

    assert first.revision == 1
    assert second.revision == 2
    assert loaded is not None
    assert loaded.revision == 2
    assert loaded.state == {"day": 3, "nested": {"rain": False}}
