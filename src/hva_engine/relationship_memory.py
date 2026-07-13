from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from hva_engine.memory_store import LongTermMemoryStore, MemoryDocument
from hva_engine.models import GameEvent


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _target_token(target_key: str) -> str:
    digest = hashlib.sha256(target_key.encode("utf-8")).hexdigest()[:24]
    return f"relation-target:{digest}"


@dataclass
class PersonBelief:
    key: str
    statement: str
    epistemic_status: str
    confidence: float
    evidence_memory_ids: tuple[str, ...]
    last_updated_turn: int

    def private_view(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "statement": self.statement,
            "epistemic_status": self.epistemic_status,
            "confidence": round(self.confidence, 3),
            "evidence_memory_ids": list(self.evidence_memory_ids),
            "last_updated_turn": self.last_updated_turn,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PersonBelief:
        return cls(
            key=str(value["key"]),
            statement=str(value["statement"]),
            epistemic_status=str(value["epistemic_status"]),
            confidence=float(value["confidence"]),
            evidence_memory_ids=tuple(value.get("evidence_memory_ids", ())),
            last_updated_turn=int(value.get("last_updated_turn", 0)),
        )


@dataclass
class SensitivePoint:
    topic: str
    kind: str
    salience: float
    confidence: float
    evidence_memory_ids: tuple[str, ...]
    last_updated_turn: int

    def private_view(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "kind": self.kind,
            "salience": round(self.salience, 3),
            "confidence": round(self.confidence, 3),
            "evidence_memory_ids": list(self.evidence_memory_ids),
            "last_updated_turn": self.last_updated_turn,
            "warning": "interaction clue, not a canonical vulnerability fact",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SensitivePoint:
        return cls(
            topic=str(value["topic"]),
            kind=str(value["kind"]),
            salience=float(value["salience"]),
            confidence=float(value["confidence"]),
            evidence_memory_ids=tuple(value.get("evidence_memory_ids", ())),
            last_updated_turn=int(value.get("last_updated_turn", 0)),
        )


@dataclass
class RelationshipProfile:
    target_token: str
    target_label: str
    target_kind: str
    trust: float = 0.5
    respect: float = 0.5
    warmth: float = 0.5
    hostility: float = 0.0
    familiarity: float = 0.0
    attitude: str = "unfamiliar"
    interaction_count: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    impressions: dict[str, PersonBelief] = field(default_factory=dict)
    background_beliefs: dict[str, PersonBelief] = field(default_factory=dict)
    sensitive_points: dict[str, SensitivePoint] = field(default_factory=dict)
    last_updated_turn: int = 0

    def _refresh_attitude(self) -> None:
        if self.hostility >= 0.68 and self.trust <= 0.32:
            self.attitude = "hostile"
        elif self.hostility >= 0.42 or self.trust <= 0.4:
            self.attitude = "guarded"
        elif self.warmth >= 0.66 and self.trust >= 0.62:
            self.attitude = "warm"
        elif self.trust >= 0.58:
            self.attitude = "cautiously_positive"
        elif self.familiarity < 0.12:
            self.attitude = "unfamiliar"
        else:
            self.attitude = "cautious"

    @staticmethod
    def _merge_evidence(existing: tuple[str, ...], memory_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*existing, memory_id)))[-8:]

    def remember_impression(
        self,
        *,
        key: str,
        statement: str,
        epistemic_status: str,
        confidence: float,
        evidence_memory_id: str,
        turn: int,
    ) -> None:
        previous = self.impressions.get(key)
        evidence = self._merge_evidence(
            previous.evidence_memory_ids if previous else (), evidence_memory_id
        )
        self.impressions[key] = PersonBelief(
            key=key,
            statement=statement,
            epistemic_status=epistemic_status,
            confidence=_clamp(
                max(confidence, previous.confidence * 0.92 if previous else 0.0)
            ),
            evidence_memory_ids=evidence,
            last_updated_turn=turn,
        )

    def remember_background(
        self,
        *,
        key: str,
        statement: str,
        confidence: float,
        evidence_memory_id: str,
        turn: int,
    ) -> None:
        previous = self.background_beliefs.get(key)
        evidence = self._merge_evidence(
            previous.evidence_memory_ids if previous else (), evidence_memory_id
        )
        self.background_beliefs[key] = PersonBelief(
            key=key,
            statement=statement,
            epistemic_status="publicly_reported",
            confidence=_clamp(confidence),
            evidence_memory_ids=evidence,
            last_updated_turn=turn,
        )

    def remember_sensitive_point(
        self,
        *,
        topic: str,
        kind: str,
        salience: float,
        confidence: float,
        evidence_memory_id: str,
        turn: int,
    ) -> None:
        key = f"{kind}:{topic}"
        previous = self.sensitive_points.get(key)
        evidence = self._merge_evidence(
            previous.evidence_memory_ids if previous else (), evidence_memory_id
        )
        self.sensitive_points[key] = SensitivePoint(
            topic=topic,
            kind=kind,
            salience=_clamp(
                0.65 * previous.salience + 0.35 * salience if previous else salience
            ),
            confidence=_clamp(
                max(confidence, previous.confidence + 0.08 if previous else confidence)
            ),
            evidence_memory_ids=evidence,
            last_updated_turn=turn,
        )

    def observe(
        self,
        event: GameEvent,
        *,
        evidence_memory_id: str,
        cooperative: bool,
    ) -> None:
        self.interaction_count += 1
        self.familiarity = _clamp(self.familiarity + 0.07)
        self.last_updated_turn = max(self.last_updated_turn, event.seq)
        if event.type == "action_applied":
            action = str(event.payload.get("action_type", "unknown"))
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            cooperative_actions = {"coordinate", "research", "stabilize", "share", "support"}
            confrontational_actions = {"rebuttal", "counterattack", "threaten", "betray"}
            cooperative_signal = 1.0 if cooperative and action in cooperative_actions else 0.0
            hostile_signal = 0.65 if action in confrontational_actions else 0.0
            self.trust = _clamp(
                0.86 * self.trust
                + 0.14 * (0.55 + 0.4 * cooperative_signal - 0.45 * hostile_signal)
            )
            self.warmth = _clamp(
                0.88 * self.warmth + 0.12 * (0.5 + 0.42 * cooperative_signal)
            )
            self.hostility = _clamp(0.82 * self.hostility + 0.18 * hostile_signal)
            count = self.action_counts[action]
            qualifier = "repeatedly chooses" if count >= 2 else "has chosen"
            self.remember_impression(
                key=f"behavior:{action}",
                statement=f"{self.target_label} {qualifier} {action} in observed play.",
                epistemic_status="inferred_pattern",
                confidence=min(0.92, 0.38 + 0.13 * count),
                evidence_memory_id=evidence_memory_id,
                turn=event.seq,
            )
        elif event.type == "interview_question":
            severity = _clamp(float(event.payload.get("severity", 0.0)))
            topic = str(event.payload.get("theme", "unknown"))[:80]
            self.hostility = _clamp(0.68 * self.hostility + 0.32 * severity)
            self.trust = _clamp(0.82 * self.trust + 0.18 * (1 - severity))
            self.warmth = _clamp(0.9 * self.warmth + 0.1 * (1 - severity))
            style = "uses severe confrontational questions" if severity >= 0.72 else (
                "tests boundaries through pointed questions"
                if severity >= 0.42
                else "uses relatively measured questions"
            )
            self.remember_impression(
                key="interaction_style",
                statement=f"{self.target_label} {style}.",
                epistemic_status="inferred_pattern",
                confidence=min(0.9, 0.5 + 0.08 * self.interaction_count),
                evidence_memory_id=evidence_memory_id,
                turn=event.seq,
            )
            self.remember_sensitive_point(
                topic=topic,
                kind="salient_topic",
                salience=severity,
                confidence=min(0.88, 0.42 + 0.1 * self.interaction_count),
                evidence_memory_id=evidence_memory_id,
                turn=event.seq,
            )
        elif event.type == "interview_response":
            topic = str(event.payload.get("theme", "unknown"))[:80]
            intensity = _clamp(float(event.payload.get("intensity", 0.5)))
            severity = _clamp(float(event.payload.get("severity", 0.5)))
            strategy = str(event.payload.get("strategy", "respond"))
            self.remember_sensitive_point(
                topic=topic,
                kind="observed_reaction_trigger",
                salience=max(intensity, severity),
                confidence=min(0.9, 0.48 + 0.08 * self.interaction_count),
                evidence_memory_id=evidence_memory_id,
                turn=event.seq,
            )
            self.remember_impression(
                key=f"coping:{strategy}",
                statement=f"{self.target_label} has responded with {strategy} under pressure.",
                epistemic_status="observed_behavior",
                confidence=min(0.9, 0.52 + 0.06 * self.interaction_count),
                evidence_memory_id=evidence_memory_id,
                turn=event.seq,
            )
        elif event.type == "story_reveal":
            beat = event.payload.get("beat", {})
            if isinstance(beat, dict):
                title = str(beat.get("title", "revealed_background"))[:100]
                recollection = str(beat.get("recollection", ""))[:500]
                lesson = str(beat.get("lesson", ""))[:240]
                statement = " ".join(part for part in (title, recollection, lesson) if part)
                self.remember_background(
                    key=f"disclosure:{title}",
                    statement=statement,
                    confidence=0.92,
                    evidence_memory_id=evidence_memory_id,
                    turn=event.seq,
                )
        self.respect = _clamp(
            0.9 * self.respect + 0.1 * (0.62 if self.interaction_count else 0.5)
        )
        self._refresh_attitude()

    def private_view(self) -> dict[str, Any]:
        return {
            "target_label": self.target_label,
            "target_kind": self.target_kind,
            "relationship": {
                "trust": round(self.trust, 3),
                "respect": round(self.respect, 3),
                "warmth": round(self.warmth, 3),
                "hostility": round(self.hostility, 3),
                "familiarity": round(self.familiarity, 3),
                "attitude": self.attitude,
                "interaction_count": self.interaction_count,
            },
            "behavioral_patterns": dict(sorted(self.action_counts.items())),
            "impressions": [
                item.private_view() for item in self.impressions.values()
            ],
            "reported_background": [
                item.private_view() for item in self.background_beliefs.values()
            ],
            "sensitive_points": [
                item.private_view() for item in self.sensitive_points.values()
            ],
            "epistemic_warning": (
                "Impressions and sensitive points are revisable beliefs, not canonical facts. "
                "Reported background records what the target publicly disclosed."
            ),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "target_token": self.target_token,
            "target_label": self.target_label,
            "target_kind": self.target_kind,
            "trust": self.trust,
            "respect": self.respect,
            "warmth": self.warmth,
            "hostility": self.hostility,
            "familiarity": self.familiarity,
            "attitude": self.attitude,
            "interaction_count": self.interaction_count,
            "action_counts": self.action_counts,
            "impressions": [item.private_view() for item in self.impressions.values()],
            "background_beliefs": [
                item.private_view() for item in self.background_beliefs.values()
            ],
            "sensitive_points": [
                item.private_view() for item in self.sensitive_points.values()
            ],
            "last_updated_turn": self.last_updated_turn,
        }

    @classmethod
    def from_metadata(cls, value: dict[str, Any]) -> RelationshipProfile:
        profile = cls(
            target_token=str(value["target_token"]),
            target_label=str(value["target_label"]),
            target_kind=str(value.get("target_kind", "unknown")),
            trust=float(value.get("trust", 0.5)),
            respect=float(value.get("respect", 0.5)),
            warmth=float(value.get("warmth", 0.5)),
            hostility=float(value.get("hostility", 0.0)),
            familiarity=float(value.get("familiarity", 0.0)),
            attitude=str(value.get("attitude", "cautious")),
            interaction_count=int(value.get("interaction_count", 0)),
            action_counts={
                str(key): int(count)
                for key, count in dict(value.get("action_counts", {})).items()
            },
            last_updated_turn=int(value.get("last_updated_turn", 0)),
        )
        profile.impressions = {
            item.key: item
            for raw in value.get("impressions", [])
            if isinstance(raw, dict)
            for item in (PersonBelief.from_dict(raw),)
        }
        profile.background_beliefs = {
            item.key: item
            for raw in value.get("background_beliefs", [])
            if isinstance(raw, dict)
            for item in (PersonBelief.from_dict(raw),)
        }
        profile.sensitive_points = {
            f"{item.kind}:{item.topic}": item
            for raw in value.get("sensitive_points", [])
            if isinstance(raw, dict)
            for item in (SensitivePoint.from_dict(raw),)
        }
        return profile


class RelationshipMemory:
    """Persists one revisable, evidence-backed social model per target identity."""

    def __init__(self, owner_id: str, store: LongTermMemoryStore) -> None:
        self.owner_id = owner_id
        self.store = store

    def load_or_create(
        self, target_key: str, target_label: str, target_kind: str
    ) -> RelationshipProfile:
        token = _target_token(target_key)
        candidates = self.store.query(
            self.owner_id,
            terms=set(),
            tags=(token,),
            recent_limit=0,
            candidate_limit=4,
        )
        for document in candidates:
            if "relationship_profile" not in document.categories:
                continue
            metadata = document.metadata
            if metadata.get("target_token") == token:
                profile = RelationshipProfile.from_metadata(metadata)
                profile.target_label = target_label
                profile.target_kind = target_kind
                return profile
        return RelationshipProfile(
            target_token=token,
            target_label=target_label,
            target_kind=target_kind,
        )

    def persist(self, profile: RelationshipProfile) -> None:
        views = profile.private_view()
        statements = [
            *(item["statement"] for item in views["impressions"]),
            *(item["statement"] for item in views["reported_background"]),
            *(
                f"{item['kind']} {item['topic']}"
                for item in views["sensitive_points"]
            ),
        ]
        summary = (
            f"Relationship with {profile.target_label}: {profile.attitude}; "
            f"trust={profile.trust:.2f}, hostility={profile.hostility:.2f}."
        )
        self.store.upsert(
            MemoryDocument(
                owner_id=self.owner_id,
                id=f"relationship-{profile.target_token.split(':', 1)[1]}",
                turn=profile.last_updated_turn,
                kind="semantic",
                summary=summary,
                content=" ".join((summary, *statements))[:4_000],
                action="",
                outcome_events=(),
                score_delta=profile.trust - profile.hostility,
                importance=max(0.5, profile.familiarity, profile.hostility),
                emotional_valence=max(-1.0, min(1.0, profile.trust - profile.hostility)),
                surprise=0.0,
                categories=("relationship", "relationship_profile", "person_model"),
                tags=(profile.target_token, "relationship", f"attitude:{profile.attitude}"),
                metadata={
                    **profile.to_metadata(),
                    "source": "relationship_profile",
                    "epistemic_status": "revisable_social_model",
                },
            )
        )
