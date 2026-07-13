from random import Random

from hva_engine.engine import build_default_engine
from hva_engine.models import ActorKind, Player
from hva_engine.mods import AgentTown
from hva_engine.social_media import town_social_hub


def test_social_hub_preserves_reshare_provenance_and_hides_truth() -> None:
    hub = town_social_hub()
    state = hub.initial_state()
    claim_id = hub.register_claim(
        state,
        text="The dam has collapsed",
        source_event_id="event-flood",
        truth_status="false",
        asserted_by="anonymous",
        refuting_event_ids=("event-correction",),
    )
    original = hub.publish(
        state,
        platform_id="town_short_video",
        author_id="anonymous",
        author_name="Anonymous",
        content="Urgent video",
        content_kind="short_video",
        source_event_ids=("event-flood",),
        claim_id=claim_id,
        turn=1,
        distortion=0.6,
    )
    reshared = hub.publish(
        state,
        platform_id="town_short_video",
        author_id="agent-a",
        author_name="Astra",
        content="Unverified reshare",
        source_event_ids=("event-flood",),
        claim_id=claim_id,
        parent_post_id=original["id"],
        turn=2,
        distortion=0.2,
    )

    assert reshared["provenance"] == [original["id"]]
    assert reshared["distortion"] == 0.8
    assert hub.verify_claim(
        state,
        claim_id=claim_id,
        verifier_id="agent-a",
        evidence_event_ids=[],
        turn=3,
    )["result"] == "unresolved"
    assert hub.verify_claim(
        state,
        claim_id=claim_id,
        verifier_id="agent-a",
        evidence_event_ids=["event-flood"],
        turn=4,
    )["result"] == "unresolved"
    assert hub.verify_claim(
        state,
        claim_id=claim_id,
        verifier_id="agent-a",
        evidence_event_ids=["event-correction"],
        turn=5,
    )["result"] == "verified_false"
    assert "_truth_status" not in hub.public_state(state)["claims"][claim_id]


def test_town_agents_use_phone_discuss_reshare_and_verify_claims() -> None:
    engine = build_default_engine()
    view = engine.create_match("agent_town", seed=7, mode="agent_coop")
    events = engine.get(view.id).events
    action_types = {
        event.payload.get("action_type")
        for event in events
        if event.type == "action_applied"
    }

    assert {"check_phone", "publish_post", "comment_post", "verify_claim"} <= action_types
    assert any(event.type == "town_rumor_seeded" for event in events)
    assert any(event.type == "town_claim_verified" for event in events)
    assert any(event.type == "town_social_reshared" for event in events)
    assert all(
        event.payload.get("parent_post_id")
        for event in events
        if event.type == "town_social_reshared"
    )
    assert all(
        "_truth_status" not in claim
        for claim in view.state["social_media"]["claims"].values()
    )


def test_feed_affinity_and_provenance_shape_epistemic_appraisal() -> None:
    hub = town_social_hub()
    state = hub.initial_state()
    trusted = hub.publish(
        state,
        platform_id="town_weibo",
        author_id="trusted-neighbor",
        author_name="Trusted Neighbor",
        content="A sourced local report",
        turn=1,
    )
    viral = hub.publish(
        state,
        platform_id="town_weibo",
        author_id="viral-stranger",
        author_name="Viral Stranger",
        content="A distorted viral report",
        turn=1,
        distortion=0.65,
    )
    feed = hub.feed(
        state,
        "town_weibo",
        affinity_by_author={"trusted-neighbor": 1.0, "viral-stranger": 0.0},
    )
    assert feed[0]["id"] == trusted["id"]
    assert hub.appraise_report(state, trusted["id"], source_trust=0.9)[
        "credibility_estimate"
    ] > hub.appraise_report(state, viral["id"], source_trust=0.9)[
        "credibility_estimate"
    ]


def test_town_investigation_acquires_evidence_before_verification() -> None:
    mod = AgentTown()
    actor = Player(id="human", name="Investigator", kind=ActorKind.HUMAN)
    state = mod.initial_state([actor], Random(11))
    state["_memory_owner_ids"] = {actor.id: actor.id}
    state["residents"][actor.id]["location"] = "library"
    claim_id = mod.social_hub.register_claim(
        state["social_media"],
        text="The dam has completely collapsed",
        source_event_id="world-flood",
        truth_status="false",
        asserted_by="anonymous-video",
    )
    post = mod.social_hub.publish(
        state["social_media"],
        platform_id="town_short_video",
        author_id="anonymous-video",
        author_name="Anonymous Video",
        content="The dam is gone",
        content_kind="short_video",
        source_event_ids=("world-flood",),
        claim_id=claim_id,
        turn=0,
        distortion=0.72,
    )
    state["_social_knowledge"][actor.id]["seen_posts"].append(post["id"])

    investigate = next(
        action
        for action in mod.legal_actions(state, actor.id)
        if action.type == "investigate_claim"
    )
    investigated, investigation_events = mod.apply_action(
        state, actor.id, investigate, Random(12)
    )
    evidence_event = next(
        event for event in investigation_events if event["type"] == "town_claim_investigated"
    )
    assert evidence_event["evidence_relation"] == "refutes"

    verify = next(
        action
        for action in mod.legal_actions(investigated, actor.id)
        if action.type == "verify_claim"
    )
    verified, verification_events = mod.apply_action(
        investigated, actor.id, verify, Random(13)
    )
    result = next(
        event for event in verification_events if event["type"] == "town_claim_verified"
    )
    assert result["result"] == "verified_false"
    public_claim = mod.public_state(verified, actor.id)["social_media"]["claims"][claim_id]
    assert "_truth_status" not in public_claim
    assert "_evidence" not in public_claim
