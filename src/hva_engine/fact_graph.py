from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hva_engine.fact_store import FactStore, InMemoryFactStore


class FactGraphError(ValueError):
    pass


class FactVisibility(StrEnum):
    PRIVATE = "private"
    REVEALED = "revealed"
    PUBLIC = "public"


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


@dataclass
class FactNode:
    id: str
    subject: str
    predicate: str
    object: Any
    confidence: float
    source: str
    immutable: bool
    visibility: FactVisibility
    status: FactStatus = FactStatus.ACTIVE
    supersedes: str | None = None
    revision: int = 1

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "immutable": self.immutable,
            "visibility": self.visibility.value,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "revision": self.revision,
        }


class AgentFactGraph:
    """Versioned identity truth with constrained space for in-character improvisation."""

    RESERVED_PREFIXES = ("identity.", "history.formative_memory.")
    ALLOWED_DYNAMIC_PREDICATES = {
        "state.current_intention",
        "state.psychological_matrix",
        "belief.opponent_pattern",
        "belief.relationship_impression",
        "belief.interpretation",
        "preference.local",
        "goal.local",
        "history.backstory_detail",
        "history.past_acquaintance",
        "dialogue.conversational_detail",
        "dialogue.rumor",
    }

    def __init__(self, owner_id: str, store: FactStore | None = None) -> None:
        self.owner_id = owner_id
        self.store = store or InMemoryFactStore()
        self._facts: dict[str, FactNode] = {}
        self._sequence = 0
        self._rejections: list[dict[str, Any]] = []
        self.formative_memory_fact_ids: dict[str, str] = {}

    @classmethod
    def from_identity(
        cls, owner_id: str, identity: Any, store: FactStore | None = None
    ) -> AgentFactGraph:
        graph = cls(owner_id, store)
        graph._add_core("identity.name", identity.name, FactVisibility.PUBLIC)
        graph._add_core("identity.disclosure", identity.disclosure, FactVisibility.PUBLIC)
        graph._add_core("identity.social_style", identity.social_style, FactVisibility.PUBLIC)
        graph._add_core("identity.background", identity.background)
        graph._add_core("identity.aspiration", identity.aspiration)
        graph._add_core("identity.core_wound", identity.core_wound)
        graph._add_core("identity.values", list(identity.values))
        for memory in identity.formative_memories:
            fact = graph._add_core(f"history.formative_memory.{memory.title}", memory.public_view())
            graph.formative_memory_fact_ids[memory.title] = fact.id
        return graph

    def _add_core(
        self,
        predicate: str,
        value: Any,
        visibility: FactVisibility = FactVisibility.PRIVATE,
    ) -> FactNode:
        return self._insert(
            subject=self.owner_id,
            predicate=predicate,
            value=value,
            confidence=1.0,
            source="identity_seed",
            immutable=True,
            visibility=visibility,
        )

    def propose(
        self,
        *,
        subject: str,
        predicate: str,
        value: Any,
        basis_fact_ids: list[str],
        confidence: float = 0.65,
        supersedes_fact_id: str | None = None,
    ) -> FactNode:
        """Accept a bounded improvisation or reject it without mutating canonical truth."""

        try:
            if subject == "self":
                subject = self.owner_id
            self._validate_text(subject, "subject", 120)
            if subject != self.owner_id:
                raise FactGraphError("Agent proposals may only add facts about their own character")
            self._validate_text(predicate, "predicate", 120)
            self._validate_value(value)
            if predicate.startswith(self.RESERVED_PREFIXES):
                raise FactGraphError("Agent proposals cannot write reserved identity/history facts")
            if predicate not in self.ALLOWED_DYNAMIC_PREDICATES:
                raise FactGraphError(f"Dynamic predicate is not allowed: {predicate}")
            if not basis_fact_ids:
                raise FactGraphError("Improvised facts require at least one active basis fact")
            for fact_id in basis_fact_ids:
                basis = self._facts.get(fact_id)
                if basis is None or basis.status != FactStatus.ACTIVE:
                    raise FactGraphError(f"Unknown or inactive basis fact: {fact_id}")
            return self._insert_revision(
                subject=subject,
                predicate=predicate,
                value=value,
                confidence=confidence,
                source="agent_improvisation:" + ",".join(basis_fact_ids),
                supersedes_fact_id=supersedes_fact_id,
            )
        except FactGraphError as exc:
            self._rejections.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "value": value,
                    "reason": str(exc),
                }
            )
            self.store.save_rejection(self.owner_id, self._rejections[-1])
            raise

    def upsert_runtime(self, predicate: str, value: Any) -> FactNode:
        if predicate not in self.ALLOWED_DYNAMIC_PREDICATES:
            raise FactGraphError(f"Runtime predicate is not allowed: {predicate}")
        active = self._active_for(self.owner_id, predicate)
        if active and self._same_value(active.object, value):
            return active
        return self._insert_revision(
            subject=self.owner_id,
            predicate=predicate,
            value=value,
            confidence=0.9,
            source="engine_observation",
            supersedes_fact_id=active.id if active else None,
        )

    def reveal(self, fact_id: str) -> FactNode:
        try:
            fact = self._facts[fact_id]
        except KeyError as exc:
            raise FactGraphError(f"Unknown fact: {fact_id}") from exc
        if fact.status != FactStatus.ACTIVE:
            raise FactGraphError("Cannot reveal an inactive fact")
        fact.visibility = FactVisibility.REVEALED
        self.store.save_fact(self.owner_id, fact)
        return fact

    def private_view(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "facts": [
                fact.view() for fact in self._facts.values() if fact.status == FactStatus.ACTIVE
            ],
            "recent_revision_history": [
                fact.view()
                for fact in list(self._facts.values())[-8:]
                if fact.status == FactStatus.SUPERSEDED
            ],
            "constraints": {
                "reserved_prefixes": list(self.RESERVED_PREFIXES),
                "allowed_dynamic_predicates": sorted(self.ALLOWED_DYNAMIC_PREDICATES),
                "revision_requires_explicit_supersedes": True,
                "improvisation_requires_basis": True,
            },
        }

    def public_view(self) -> dict[str, Any]:
        visible = [
            fact.view()
            for fact in self._facts.values()
            if fact.visibility in {FactVisibility.PUBLIC, FactVisibility.REVEALED}
            and fact.status == FactStatus.ACTIVE
        ]
        return {
            "owner_id": self.owner_id,
            "facts": visible,
            "stats": {
                "total_versions": len(self._facts),
                "active_facts": sum(
                    fact.status == FactStatus.ACTIVE for fact in self._facts.values()
                ),
                "visible_facts": len(visible),
                "rejected_proposals": len(self._rejections),
                "improvised_versions": sum(
                    fact.source.startswith("agent_improvisation") for fact in self._facts.values()
                ),
                "superseded_versions": sum(
                    fact.status == FactStatus.SUPERSEDED for fact in self._facts.values()
                ),
            },
        }

    def _insert_revision(
        self,
        *,
        subject: str,
        predicate: str,
        value: Any,
        confidence: float,
        source: str,
        supersedes_fact_id: str | None,
    ) -> FactNode:
        active = self._active_for(subject, predicate)
        if active and self._same_value(active.object, value):
            return active
        if active:
            if active.immutable:
                raise FactGraphError(f"Fact conflicts with immutable canonical fact {active.id}")
            if supersedes_fact_id != active.id:
                raise FactGraphError(f"Revision must explicitly supersede active fact {active.id}")
            active.status = FactStatus.SUPERSEDED
            self.store.save_fact(self.owner_id, active)
            revision = active.revision + 1
            visibility = active.visibility
        else:
            if supersedes_fact_id is not None:
                raise FactGraphError("Cannot supersede a fact that is not active for this relation")
            revision = 1
            visibility = FactVisibility.PRIVATE
        return self._insert(
            subject=subject,
            predicate=predicate,
            value=value,
            confidence=confidence,
            source=source,
            immutable=False,
            visibility=visibility,
            supersedes=supersedes_fact_id,
            revision=revision,
        )

    def _insert(
        self,
        *,
        subject: str,
        predicate: str,
        value: Any,
        confidence: float,
        source: str,
        immutable: bool,
        visibility: FactVisibility,
        supersedes: str | None = None,
        revision: int = 1,
    ) -> FactNode:
        self._sequence += 1
        fact = FactNode(
            id=f"fact-{self._sequence:04d}",
            subject=subject,
            predicate=predicate,
            object=value,
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            immutable=immutable,
            visibility=visibility,
            supersedes=supersedes,
            revision=revision,
        )
        self._facts[fact.id] = fact
        self.store.save_fact(self.owner_id, fact)
        return fact

    def _active_for(self, subject: str, predicate: str) -> FactNode | None:
        return next(
            (
                fact
                for fact in reversed(list(self._facts.values()))
                if fact.subject == subject
                and fact.predicate == predicate
                and fact.status == FactStatus.ACTIVE
            ),
            None,
        )

    def _validate_text(self, value: str, label: str, limit: int) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise FactGraphError(f"{label} must be non-empty and at most {limit} characters")

    def _validate_value(self, value: Any) -> None:
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            raise FactGraphError("Fact value must be JSON serializable") from exc
        if len(rendered) > 1000:
            raise FactGraphError("Fact value exceeds 1000 characters")

    def _same_value(self, first: Any, second: Any) -> bool:
        return json.dumps(first, sort_keys=True, default=str) == json.dumps(
            second, sort_keys=True, default=str
        )
