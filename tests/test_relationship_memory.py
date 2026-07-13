from __future__ import annotations

from hva_engine.engine import build_default_engine
from hva_engine.memory_store import InMemoryIndexedMemoryStore
from hva_engine.models import (
    AgentCharacterSelection,
    AgentTuning,
    CharacterCardSpec,
    ContentMode,
    GameEvent,
)
from hva_engine.relationship_memory import RelationshipMemory
from hva_engine.speech_style import SpeechStyleRealizer


def test_relationship_profile_separates_observation_inference_and_reported_background() -> None:
    store = InMemoryIndexedMemoryStore()
    memory = RelationshipMemory("agent-owner", store)
    profile = memory.load_or_create("human-alice", "Alice", "human")
    profile.observe(
        GameEvent(
            seq=1,
            type="interview_question",
            actor_id="human-1",
            payload={
                "theme": "identity",
                "severity": 0.92,
                "prompt": "Who are you when your story is challenged?",
            },
        ),
        evidence_memory_id="memory-question-1",
        cooperative=False,
    )
    profile.observe(
        GameEvent(
            seq=2,
            type="story_reveal",
            actor_id="human-1",
            payload={
                "beat": {
                    "title": "a childhood move",
                    "recollection": "Alice said her family moved often.",
                    "lesson": "Home is built through repeated care.",
                }
            },
        ),
        evidence_memory_id="memory-disclosure-1",
        cooperative=False,
    )
    memory.persist(profile)

    restored = RelationshipMemory("agent-owner", store).load_or_create(
        "human-alice", "Alice", "human"
    )
    view = restored.private_view()

    assert view["relationship"]["attitude"] != "unfamiliar"
    assert view["relationship"]["hostility"] > 0.2
    assert view["impressions"][0]["epistemic_status"] == "inferred_pattern"
    assert view["impressions"][0]["evidence_memory_ids"] == ["memory-question-1"]
    assert view["reported_background"][0]["epistemic_status"] == "publicly_reported"
    assert view["sensitive_points"][0]["topic"] == "identity"
    assert "not canonical" in view["epistemic_warning"]


def test_relationship_memory_is_owner_scoped() -> None:
    store = InMemoryIndexedMemoryStore()
    first = RelationshipMemory("agent-a", store)
    profile = first.load_or_create("human-alice", "Alice", "human")
    profile.observe(
        GameEvent(
            seq=1,
            type="action_applied",
            actor_id="human-1",
            payload={"action_type": "rebuttal"},
        ),
        evidence_memory_id="memory-a-1",
        cooperative=False,
    )
    first.persist(profile)

    isolated = RelationshipMemory("agent-b", store).load_or_create(
        "human-alice", "Alice", "human"
    )

    assert isolated.interaction_count == 0
    assert isolated.impressions == {}


def test_engine_restores_relationship_with_same_human_across_matches() -> None:
    engine = build_default_engine()
    character = AgentCharacterSelection(
        card_id="dou_e", memory_owner_id="dou-e-relationship-campaign"
    )
    first = engine.create_match(
        "adversarial_interview",
        human_name="Alice",
        human_memory_id="human-alice",
        seed=13,
        agent_characters=[character],
    )
    severe = max(
        first.legal_actions,
        key=lambda action: float(action.payload.get("severity", 0.0)),
    )
    engine.submit(first.id, first.human_player_id, severe)
    first_brain = next(iter(engine.get(first.id).agent_brains.values()))
    first_profile = next(iter(first_brain.relationship_profiles.values()))
    first_interactions = first_profile.interaction_count
    first_familiarity = first_profile.familiarity

    second = engine.create_match(
        "adversarial_interview",
        human_name="Alice",
        human_memory_id="human-alice",
        seed=14,
        agent_characters=[character],
    )
    severe_again = max(
        second.legal_actions,
        key=lambda action: float(action.payload.get("severity", 0.0)),
    )
    engine.submit(second.id, second.human_player_id, severe_again)
    second_brain = next(iter(engine.get(second.id).agent_brains.values()))
    second_profile = next(iter(second_brain.relationship_profiles.values()))

    assert second_profile.interaction_count > first_interactions
    assert second_profile.familiarity > first_familiarity
    assert second_profile.sensitive_points
    assert any(
        fact.predicate == "belief.relationship_impression"
        for fact in second_brain.fact_graph._facts.values()
    )
    public_summary = next(iter(second.agent_summaries.values()))
    assert "relationship_memory" not in public_summary


def _family_card() -> CharacterCardSpec:
    engine = build_default_engine()
    payload = engine.character_cards.cards["dou_e"].model_dump()
    payload.update(
        {
            "id": "family_tester",
            "name": "Lin",
            "lived_memories": [
                {
                    "title": "雨后公园的一天",
                    "recollection": (
                        "雨停以后，我和伴侣带女儿阿禾去河边公园。她踩着水洼笑，"
                        "我们在旧树下分吃温热的饭团。"
                    ),
                    "emotional_valence": 0.88,
                    "lesson": "平凡的陪伴比漂亮的承诺更可靠。",
                    "people": ["伴侣", "女儿阿禾"],
                    "themes": ["家庭", "子女", "陪伴"],
                    "place": "河边公园",
                    "time_period": "女儿七岁时的春天",
                }
            ],
            "speech_style": {
                "voice_register": "plain",
                "education_voice": "limited_formal_schooling_but_experienced",
                "vocabulary_complexity": 0.22,
                "sentence_complexity": 0.25,
                "directness": 0.82,
                "roughness": 0.76,
                "warmth": 0.68,
                "humor": 0.18,
                "philosophical_abstraction": 0.08,
                "technical_jargon": 0.02,
                "verbosity": 0.28,
                "verbal_habits": ["用生活经验解释抽象问题"],
            },
            "source_work": None,
            "source_url": None,
            "source_policy": "user_supplied_original",
        }
    )
    return CharacterCardSpec.model_validate(payload)


def test_injected_family_memory_is_canonical_private_and_retrievable() -> None:
    engine = build_default_engine()
    card = _family_card()
    view = engine.create_match(
        "adversarial_interview",
        seed=21,
        agent_characters=[
            AgentCharacterSelection(
                custom_card=card, memory_owner_id="lin-family-campaign"
            )
        ],
    )
    brain = next(iter(engine.get(view.id).agent_brains.values()))

    retrieved = brain.memory_system.retrieve(
        "女儿 阿禾 家庭 公园 陪伴", current_turn=1, limit=3
    )
    unrelated = brain.memory_system.retrieve(
        "racing tyre pit stop weather", current_turn=1, limit=3
    )

    assert any("雨后公园" in item["content"] for item in retrieved)
    family_memory = next(item for item in retrieved if "雨后公园" in item["content"])
    assert family_memory["storage_tier"] == "long_term"
    assert {"autobiographical", "family", "identity"} <= set(
        family_memory["categories"]
    )
    private_predicates = {
        fact.predicate for fact in brain.fact_graph._facts.values()
    }
    assert "history.lived_memory.雨后公园的一天" in private_predicates
    public_summary = next(iter(view.agent_summaries.values()))
    assert "雨后公园" not in str(public_summary)
    assert "女儿阿禾" not in str(public_summary)
    assert all("雨后公园" not in item["content"] for item in unrelated)


def test_speech_style_realizer_distinguishes_plain_academic_and_mature_rough_voice() -> None:
    realizer = SpeechStyleRealizer()
    semantic = "我接受质疑，但证据和羞辱不是同一回事；我们应当分别判断。"
    plain, _ = realizer.realize(
        semantic,
        {
            "voice_register": "plain",
            "sentence_complexity": 0.2,
            "directness": 0.8,
            "roughness": 0.1,
            "verbosity": 0.5,
        },
        mature_fiction=False,
    )
    academic, _ = realizer.realize(
        semantic,
        {"voice_register": "academic", "verbosity": 0.7},
        mature_fiction=False,
    )
    rough, diagnostic = realizer.realize(
        semantic,
        {"voice_register": "colloquial", "roughness": 0.9, "verbosity": 0.5},
        mature_fiction=True,
    )

    assert plain.startswith("我就实在说吧")
    assert "；" not in plain
    assert academic.startswith("从证据和逻辑上说")
    assert "屁话" in rough
    assert diagnostic["mature_roughness_enabled"] is True


def test_baseline_interview_uses_character_speech_style_without_changing_action() -> None:
    engine = build_default_engine()
    view = engine.create_match(
        "adversarial_interview",
        seed=22,
        agent_tuning=AgentTuning(content_mode=ContentMode.MATURE_FICTION),
        agent_characters=[AgentCharacterSelection(custom_card=_family_card())],
    )
    view = engine.submit(view.id, view.human_player_id, view.legal_actions[0])
    subject_line = next(
        row["text"] for row in view.state["transcript"] if row["speaker"] == "subject"
    )
    decision = next(
        event
        for event in engine.get(view.id).events
        if event.type == "agent_decision"
    )

    assert "屁话" in subject_line
    assert decision.payload["speech_style"]["diagnostic"]["source"] == (
        "engine_baseline_realizer"
    )
    assert decision.payload["speech_style"]["content_and_action_independent"] is True
    context = engine.context_preview(view.id, decision.actor_id)
    rendered_context = "\n".join(message["content"] for message in context["messages"])
    assert '"voice_register":"plain"' in rendered_context
    assert "never which action to choose" in rendered_context
