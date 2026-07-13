from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol


class SocialPlatformAdapter(Protocol):
    id: str
    display_name: str

    def manifest(self) -> dict[str, Any]: ...

    def normalize_content(self, content: str) -> str: ...


@dataclass(frozen=True)
class ConfiguredSocialPlatform:
    id: str
    display_name: str
    format: str
    content_limit: int
    capabilities: tuple[str, ...]
    ranking: str = "recency_engagement"

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "format": self.format,
            "content_limit": self.content_limit,
            "capabilities": list(self.capabilities),
            "ranking": self.ranking,
        }

    def normalize_content(self, content: str) -> str:
        return " ".join(content.split())[: self.content_limit]


class SocialMediaHub:
    """MOD-neutral social graph with claim provenance and explicit verification."""

    def __init__(self, platforms: tuple[SocialPlatformAdapter, ...]) -> None:
        self.platforms = {platform.id: platform for platform in platforms}

    def initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": "hva.social-media.v1",
            "platforms": {
                platform_id: platform.manifest()
                for platform_id, platform in self.platforms.items()
            },
            "posts": [],
            "claims": {},
            "comments": [],
            "accounts": {},
            "next_post": 1,
            "next_claim": 1,
            "next_comment": 1,
            "revision": 0,
        }

    @staticmethod
    def register_account(
        state: dict[str, Any],
        *,
        account_id: str,
        display_name: str,
        reputation: float = 0.5,
        verified: bool = False,
    ) -> dict[str, Any]:
        account = state.setdefault("accounts", {}).setdefault(
            account_id,
            {
                "id": account_id,
                "display_name": display_name,
                "reputation": round(max(0.0, min(1.0, reputation)), 3),
                "verified": verified,
                "posts": 0,
            },
        )
        if verified and not account["verified"]:
            account["verified"] = True
            account["reputation"] = max(float(account["reputation"]), 0.72)
        return account

    def register_claim(
        self,
        state: dict[str, Any],
        *,
        text: str,
        source_event_id: str | None,
        truth_status: str,
        asserted_by: str,
        supporting_event_ids: tuple[str, ...] = (),
        refuting_event_ids: tuple[str, ...] = (),
    ) -> str:
        claim_id = f"claim-{int(state['next_claim']):05d}"
        state["next_claim"] += 1
        state["claims"][claim_id] = {
            "id": claim_id,
            "text": " ".join(text.split())[:320],
            "source_event_id": source_event_id,
            "asserted_by": asserted_by,
            "public_status": "unverified",
            "verification_history": [],
            "_truth_status": truth_status,
            "_evidence": {
                "supports": list(dict.fromkeys(supporting_event_ids)),
                "refutes": list(dict.fromkeys(refuting_event_ids)),
            },
        }
        return claim_id

    def attach_evidence(
        self,
        state: dict[str, Any],
        *,
        claim_id: str,
        event_id: str,
        relation: str,
    ) -> None:
        """Register a fact-to-claim relation without exposing hidden truth to an actor."""

        if relation not in {"supports", "refutes"}:
            raise ValueError("Claim evidence relation must be 'supports' or 'refutes'")
        evidence = state["claims"][claim_id].setdefault(
            "_evidence", {"supports": [], "refutes": []}
        )
        if event_id not in evidence[relation]:
            evidence[relation].append(event_id)
            state["revision"] += 1

    def publish(
        self,
        state: dict[str, Any],
        *,
        platform_id: str,
        author_id: str,
        author_name: str,
        content: str,
        turn: int,
        content_kind: str = "text",
        source_event_ids: tuple[str, ...] = (),
        claim_id: str | None = None,
        parent_post_id: str | None = None,
        distortion: float = 0.0,
        official: bool = False,
    ) -> dict[str, Any]:
        platform = self.platforms[platform_id]
        account = self.register_account(
            state,
            account_id=author_id,
            display_name=author_name,
            reputation=0.76 if official else 0.5,
            verified=official,
        )
        post_id = f"post-{int(state['next_post']):05d}"
        state["next_post"] += 1
        parent = self.post(state, parent_post_id) if parent_post_id else None
        provenance = [*parent.get("provenance", []), parent_post_id] if parent else []
        post = {
            "id": post_id,
            "platform_id": platform_id,
            "author_id": author_id,
            "author_name": author_name,
            "content": platform.normalize_content(content),
            "content_kind": content_kind,
            "source_event_ids": list(source_event_ids),
            "claim_id": claim_id,
            "parent_post_id": parent_post_id,
            "provenance": [value for value in provenance if value],
            "distortion": round(
                min(1.0, float(parent.get("distortion", 0.0)) + distortion)
                if parent
                else min(1.0, distortion),
                3,
            ),
            "official": official,
            "turn": turn,
            "engagement": {"views": 0, "reshares": 0, "comments": 0},
        }
        state["posts"].append(post)
        account["posts"] += 1
        state["revision"] += 1
        if parent:
            parent["engagement"]["reshares"] += 1
        return deepcopy(post)

    def feed(
        self,
        state: dict[str, Any],
        platform_id: str,
        *,
        limit: int = 8,
        affinity_by_author: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        affinities = affinity_by_author or {}
        candidates = [
            post for post in state["posts"] if post["platform_id"] == platform_id
        ]
        candidates.sort(
            key=lambda post: (
                -(
                    int(post["engagement"]["reshares"]) * 3
                    + int(post["engagement"]["comments"]) * 2
                    + int(post["engagement"]["views"]) * 0.05
                    + float(post["distortion"]) * 2
                    + max(0.0, min(1.0, affinities.get(post["author_id"], 0.0))) * 3
                    + (1.0 if post["official"] else 0.0)
                ),
                -int(post["turn"]),
                post["id"],
            )
        )
        return [deepcopy(post) for post in candidates[:limit]]

    @staticmethod
    def appraise_report(
        state: dict[str, Any],
        post_id: str,
        *,
        source_trust: float | None = None,
    ) -> dict[str, Any]:
        """Return bounded epistemic cues, never the claim's hidden truth value."""

        post = SocialMediaHub.post(state, post_id)
        claim = state["claims"].get(post.get("claim_id"), {})
        account = state.get("accounts", {}).get(post["author_id"], {})
        trust = max(
            0.0,
            min(
                1.0,
                float(source_trust)
                if source_trust is not None
                else float(account.get("reputation", 0.5)),
            ),
        )
        public_status = str(claim.get("public_status", "unverified"))
        if public_status == "verified_true":
            credibility = 0.97
            uncertainty = 0.08
        elif public_status == "verified_false":
            credibility = 0.03
            uncertainty = 0.08
        elif public_status == "disputed":
            credibility = 0.5
            uncertainty = 0.72
        else:
            provenance_penalty = min(0.72, float(post.get("distortion", 0.0)))
            credibility = (0.28 + 0.72 * trust) * (1.0 - provenance_penalty)
            if post.get("official"):
                credibility = min(0.92, credibility + 0.12)
            uncertainty = 0.92 - 0.45 * abs(credibility - 0.5)
        root_post_id = (
            post.get("provenance", [])[0]
            if post.get("provenance")
            else post["id"]
        )
        return {
            "post_id": post["id"],
            "claim_id": post.get("claim_id"),
            "author_id": post["author_id"],
            "source_trust": round(trust, 3),
            "credibility_estimate": round(max(0.0, min(1.0, credibility)), 3),
            "uncertainty": round(max(0.0, min(1.0, uncertainty)), 3),
            "independent_source_key": root_post_id,
            "public_status": public_status,
            "basis": "public_provenance_status_and_viewer_source_trust_not_hidden_truth",
        }

    def mark_viewed(self, state: dict[str, Any], post_ids: list[str]) -> None:
        wanted = set(post_ids)
        for post in state["posts"]:
            if post["id"] in wanted:
                post["engagement"]["views"] += 1

    def comment(
        self,
        state: dict[str, Any],
        *,
        post_id: str,
        author_id: str,
        author_name: str,
        content: str,
        turn: int,
        stance: str,
    ) -> dict[str, Any]:
        post = self.post(state, post_id)
        comment = {
            "id": f"comment-{int(state['next_comment']):05d}",
            "post_id": post_id,
            "author_id": author_id,
            "author_name": author_name,
            "content": " ".join(content.split())[:240],
            "stance": stance,
            "turn": turn,
        }
        state["next_comment"] += 1
        state["comments"].append(comment)
        state["revision"] += 1
        post["engagement"]["comments"] += 1
        return deepcopy(comment)

    def verify_claim(
        self,
        state: dict[str, Any],
        *,
        claim_id: str,
        verifier_id: str,
        evidence_event_ids: list[str],
        turn: int,
    ) -> dict[str, Any]:
        claim = state["claims"][claim_id]
        supplied = set(evidence_event_ids)
        evidence = claim.get("_evidence", {})
        supports = supplied & set(evidence.get("supports", []))
        refutes = supplied & set(evidence.get("refutes", []))
        if supports and refutes:
            public_status = "disputed"
        elif supports:
            public_status = "verified_true"
        elif refutes:
            public_status = "verified_false"
        else:
            public_status = "unresolved"
        result = {
            "verifier_id": verifier_id,
            "turn": turn,
            "evidence_event_ids": list(evidence_event_ids),
            "result": public_status,
        }
        claim["public_status"] = public_status
        claim["verification_history"].append(result)
        state["revision"] += 1
        return deepcopy(result)

    @staticmethod
    def post(state: dict[str, Any], post_id: str | None) -> dict[str, Any]:
        return next(post for post in state["posts"] if post["id"] == post_id)

    @staticmethod
    def public_state(state: dict[str, Any]) -> dict[str, Any]:
        public = deepcopy(state)
        for claim in public["claims"].values():
            claim.pop("_truth_status", None)
            claim.pop("_evidence", None)
        return public


def town_social_hub() -> SocialMediaHub:
    return SocialMediaHub(
        (
            ConfiguredSocialPlatform(
                id="town_weibo",
                display_name="风铃微博",
                format="microblog",
                content_limit=280,
                capabilities=("post", "reshare", "comment", "claim", "fact_check"),
            ),
            ConfiguredSocialPlatform(
                id="town_short_video",
                display_name="风铃短视频",
                format="short_video",
                content_limit=160,
                capabilities=("video_caption", "reshare", "comment", "claim"),
                ranking="engagement_recency_controversy",
            ),
        )
    )
