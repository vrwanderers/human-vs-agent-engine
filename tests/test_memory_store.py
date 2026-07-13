from __future__ import annotations

from hva_engine.engine import build_default_engine
from hva_engine.human_cognition import MemorySystem
from hva_engine.memory_store import (
    InMemoryIndexedMemoryStore,
    MemoryDocument,
    SQLiteIndexedMemoryStore,
)
from hva_engine.models import AgentCharacterSelection


def _record(
    memory: MemorySystem,
    *,
    turn: int,
    content: str,
    action: str = "respond",
    score_delta: float = 0.0,
    surprise: float = 0.0,
    emotional_intensity: float = 0.0,
    tags: tuple[str, ...] = (),
) -> str:
    return memory.record(
        turn=turn,
        content=content,
        action=action,
        outcome_events=["state_changed"],
        score_delta=score_delta,
        surprise=surprise,
        emotional_intensity=emotional_intensity,
        tags=tags,
    ).id


def test_short_term_memory_expires_without_promoting_routine_noise() -> None:
    store = InMemoryIndexedMemoryStore()
    memory = MemorySystem(
        owner_id="agent-a",
        store=store,
        short_term_ttl_turns=3,
        long_term_promotion_threshold=0.8,
    )
    memory_id = _record(memory, turn=1, content="Routine low-salience observation")

    assert memory.retrieve("routine observation", current_turn=3)[0]["id"] == memory_id
    assert memory.retrieve("routine observation", current_turn=5) == []
    assert store.count("agent-a") == 0
    assert memory.public_view()["short_term"]["forgotten_count"] == 1


def test_salient_memory_survives_ttl_and_new_runtime_instance() -> None:
    store = InMemoryIndexedMemoryStore()
    first_runtime = MemorySystem(owner_id="agent-a", store=store, short_term_ttl_turns=2)
    memory_id = _record(
        first_runtime,
        turn=1,
        content="A hostile identity challenge exposed my childhood fear",
        action="set_boundary",
        score_delta=0.8,
        surprise=0.9,
        emotional_intensity=0.9,
        tags=("identity", "childhood"),
    )

    restored_runtime = MemorySystem(owner_id="agent-a", store=store)
    retrieved = restored_runtime.retrieve(
        "identity childhood boundary", current_turn=40, mood_valence=-0.2
    )

    assert retrieved[0]["id"] == memory_id
    assert retrieved[0]["storage_tier"] == "long_term"
    assert {"identity", "emotion"} <= set(retrieved[0]["categories"])


def test_semantic_reflection_and_procedural_experience_are_restored() -> None:
    store = InMemoryIndexedMemoryStore()
    memory = MemorySystem(owner_id="agent-a", store=store, reflection_threshold=0.9)
    _record(
        memory,
        turn=1,
        content="A failed counterattack damaged trust",
        action="counterattack",
        score_delta=-0.5,
        surprise=0.8,
        emotional_intensity=0.7,
    )
    _record(
        memory,
        turn=2,
        content="A second counterattack increased pressure",
        action="counterattack",
        score_delta=-0.4,
        surprise=0.6,
        emotional_intensity=0.8,
    )
    reflection = memory.maybe_reflect(2)
    assert reflection is not None

    restored = MemorySystem(owner_id="agent-a", store=store)

    assert restored.reflections[-1].id == reflection.id
    assert restored.reflections[-1].evidence_memory_ids == reflection.evidence_memory_ids
    assert restored.procedural_values()["counterattack"] < 0


def test_inverted_index_limits_candidates_and_enforces_owner_isolation() -> None:
    store = InMemoryIndexedMemoryStore()
    for index in range(200):
        owner_id = "agent-a"
        marker = "unique_boundary_memory" if index == 5 else f"routine_{index}"
        store.upsert(
            MemoryDocument(
                owner_id=owner_id,
                id=f"memory-{index:06d}",
                turn=index,
                kind="episodic",
                summary=marker,
                content=marker,
                action="respond",
                outcome_events=(),
                score_delta=0.0,
                importance=0.5,
                emotional_valence=0.0,
                surprise=0.0,
                categories=("experience",),
            )
        )

    matches = store.query(
        "agent-a", terms={"unique_boundary_memory"}, recent_limit=8, candidate_limit=24
    )
    diagnostics = store.diagnostics("agent-a")["last_query"]

    assert "memory-000005" in {item.id for item in matches}
    assert diagnostics["candidates_loaded"] < diagnostics["total_records"]
    assert diagnostics["full_scan"] is False
    assert store.query("agent-b", terms={"unique_boundary_memory"}) == []


def test_sqlite_store_persists_normalized_chinese_memory_indexes(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    first_store = SQLiteIndexedMemoryStore(path)
    first_runtime = MemorySystem(owner_id="dou-e", store=first_store)
    memory_id = _record(
        first_runtime,
        turn=2,
        content="童年的冤屈让我在身份被质疑时先保护家人，再解释事实。",
        action="invoke_memory",
        score_delta=0.7,
        surprise=0.9,
        emotional_intensity=0.8,
        tags=("身份", "家庭"),
    )
    first_store.close()

    restored_store = SQLiteIndexedMemoryStore(path)
    restored_runtime = MemorySystem(owner_id="dou-e", store=restored_store)
    retrieved = restored_runtime.retrieve("童年 身份 家人", current_turn=80)
    diagnostics = restored_store.diagnostics("dou-e")

    assert retrieved[0]["id"] == memory_id
    assert diagnostics["term_index_size"] > diagnostics["records"]
    assert diagnostics["category_index_size"] >= 3
    restored_store.close()


def test_stable_memory_owner_restores_character_experience_across_matches() -> None:
    engine = build_default_engine()
    selection = AgentCharacterSelection(
        card_id="dou_e", memory_owner_id="dou-e-campaign-01"
    )
    first = engine.create_match(
        "adversarial_interview", seed=3, agent_characters=[selection]
    )
    first_question = max(
        first.legal_actions,
        key=lambda candidate: float(candidate.payload.get("severity", 0.0)),
    )
    engine.submit(first.id, first.human_player_id, first_question)
    first_agent = next(iter(engine.get(first.id).agent_brains.values()))
    assert first_agent.memory_system.public_view()["long_term"]["records"] > 0

    second = engine.create_match(
        "adversarial_interview", seed=4, agent_characters=[selection]
    )
    second_question = max(
        second.legal_actions,
        key=lambda candidate: float(candidate.payload.get("severity", 0.0)),
    )
    engine.submit(second.id, second.human_player_id, second_question)
    second_decision = next(
        event
        for event in engine.get(second.id).events
        if event.type == "agent_decision"
    )

    assert second_decision.payload["retrieved_memory_ids"]
    assert any(
        item["storage_tier"] == "long_term"
        for item in second_decision.payload["retrieved_memories"]
    )
