from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hva_engine.memory_store import (
    InMemoryIndexedMemoryStore,
    LongTermMemoryStore,
    MemoryDocument,
    tokenize_memory_text,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokens(value: str) -> set[str]:
    return tokenize_memory_text(value)


def _memory_summary(content: str, action: str, outcome_events: tuple[str, ...]) -> str:
    normalized = " ".join(content.strip().split())
    if not normalized:
        normalized = f"Action {action} led to {','.join(outcome_events) or 'a state change'}"
    return normalized if len(normalized) <= 180 else f"{normalized[:177].rstrip()}..."


def _memory_categories(
    content: str,
    action: str,
    outcome_events: tuple[str, ...],
    tags: tuple[str, ...],
) -> tuple[str, ...]:
    text = " ".join((content, action, *outcome_events, *tags)).lower()
    text_tokens = _tokens(text)

    def contains(keyword: str) -> bool:
        if any("\u4e00" <= character <= "\u9fff" for character in keyword):
            return keyword in text
        return any(
            token == keyword or (len(keyword) >= 5 and token.startswith(keyword))
            for token in text_tokens
        )

    categories = {"experience"}
    keyword_groups = {
        "identity": (
            "identity",
            "family",
            "childhood",
            "origin",
            "身份",
            "家庭",
            "童年",
            "身世",
            "故乡",
        ),
        "relationship": (
            "trust",
            "ally",
            "betray",
            "opponent",
            "friend",
            "信任",
            "盟友",
            "背叛",
            "对手",
            "朋友",
        ),
        "family": (
            "family",
            "parent",
            "mother",
            "father",
            "spouse",
            "child",
            "daughter",
            "son",
            "家庭",
            "父亲",
            "母亲",
            "伴侣",
            "孩子",
            "女儿",
            "儿子",
        ),
        "emotion": (
            "fear",
            "anger",
            "stress",
            "frustrat",
            "shame",
            "恐惧",
            "愤怒",
            "压力",
            "沮丧",
            "羞耻",
        ),
        "world": (
            "crisis",
            "country",
            "virus",
            "court",
            "rule",
            "危机",
            "国家",
            "病毒",
            "宫廷",
            "规则",
        ),
    }
    for category, keywords in keyword_groups.items():
        if any(contains(keyword) for keyword in keywords):
            categories.add(category)
    strategic_terms = ("goal", "plan", "strategy", "目标", "计划", "策略")
    if action or any(contains(token) for token in strategic_terms):
        categories.add("strategy")
    return tuple(sorted(categories))


class MemoryKind(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class DecisionMode(StrEnum):
    HABITUAL = "habitual"
    DELIBERATIVE = "deliberative"


@dataclass
class CognitiveMemory:
    id: str
    turn: int
    kind: MemoryKind
    content: str
    action: str
    outcome_events: tuple[str, ...]
    score_delta: float
    importance: float
    emotional_valence: float
    surprise: float
    tags: tuple[str, ...] = ()
    summary: str = ""
    categories: tuple[str, ...] = ()
    stored_long_term: bool = False
    recency_distance: int = 0
    epistemic_status: str = "experienced"
    source: str = "runtime_memory"
    supporting_fact_id: str | None = None
    last_access_turn: int = 0
    access_count: int = 0

    def public_view(self, retrieval_score: float | None = None) -> dict[str, Any]:
        view: dict[str, Any] = {
            "id": self.id,
            "turn": self.turn,
            "kind": self.kind.value,
            "content": self.content,
            "action": self.action,
            "outcome_events": list(self.outcome_events),
            "score_delta": round(self.score_delta, 3),
            "importance": round(self.importance, 3),
            "emotional_valence": round(self.emotional_valence, 3),
            "surprise": round(self.surprise, 3),
            "tags": list(self.tags),
            "summary": self.summary,
            "categories": list(self.categories),
            "storage_tier": "long_term" if self.stored_long_term else "short_term",
            "epistemic_status": self.epistemic_status,
            "source": self.source,
            "supporting_fact_id": self.supporting_fact_id,
        }
        if retrieval_score is not None:
            view["retrieval_score"] = round(retrieval_score, 3)
        return view


@dataclass(frozen=True)
class Reflection:
    id: str
    turn: int
    belief: str
    confidence: float
    evidence_memory_ids: tuple[str, ...]
    revision_of: str | None = None

    def public_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turn": self.turn,
            "belief": self.belief,
            "confidence": round(self.confidence, 3),
            "evidence_memory_ids": list(self.evidence_memory_ids),
            "revision_of": self.revision_of,
        }


@dataclass
class MemorySystem:
    """TTL short-term memory plus owner-scoped indexed long-term memory.

    The implementation is deterministic so it can act as a testable baseline. An LLM may
    phrase a reflection later, but it cannot remove the evidence requirement.
    """

    owner_id: str = "standalone"
    store: LongTermMemoryStore = field(default_factory=InMemoryIndexedMemoryStore)
    short_term_ttl_turns: int = 6
    short_term_limit: int = 16
    long_term_promotion_threshold: float = 0.48
    reflection_threshold: float = 1.55
    episodic: list[CognitiveMemory] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
    procedural: dict[str, list[float]] = field(default_factory=dict)
    _unreflected_importance: float = 0.0
    _forgotten_short_term: int = 0

    def __post_init__(self) -> None:
        if self.short_term_ttl_turns < 1:
            raise ValueError("short_term_ttl_turns must be at least 1")
        if self.short_term_limit < 1:
            raise ValueError("short_term_limit must be at least 1")
        if self.reflections:
            return
        documents = self.store.list_by_category(self.owner_id, "reflection", limit=12)
        self.reflections = [
            Reflection(
                id=document.id,
                turn=document.turn,
                belief=document.content,
                confidence=float(document.metadata.get("confidence", 0.5)),
                evidence_memory_ids=tuple(document.metadata.get("evidence_memory_ids", ())),
                revision_of=document.metadata.get("revision_of"),
            )
            for document in reversed(documents)
        ]

    @staticmethod
    def _from_document(document: MemoryDocument) -> CognitiveMemory:
        return CognitiveMemory(
            id=document.id,
            turn=document.turn,
            kind=MemoryKind(document.kind),
            content=document.content,
            action=document.action,
            outcome_events=document.outcome_events,
            score_delta=document.score_delta,
            importance=document.importance,
            emotional_valence=document.emotional_valence,
            surprise=document.surprise,
            tags=document.tags,
            summary=document.summary,
            categories=document.categories,
            stored_long_term=True,
            last_access_turn=document.last_access_turn,
            access_count=document.access_count,
            epistemic_status=str(
                document.metadata.get("epistemic_status", "experienced")
            ),
            source=str(document.metadata.get("source", "runtime_memory")),
            supporting_fact_id=document.metadata.get("supporting_fact_id"),
        )

    def forget_expired(self, current_turn: int) -> int:
        retained = [
            memory
            for memory in self.episodic
            if current_turn - memory.turn <= self.short_term_ttl_turns
        ]
        forgotten = len(self.episodic) - len(retained)
        if forgotten:
            self.episodic = retained
            self._forgotten_short_term += forgotten
        return forgotten

    def seed_identity_memories(
        self,
        identity: Any,
        *,
        formative_fact_ids: dict[str, str] | None = None,
        lived_fact_ids: dict[str, str] | None = None,
    ) -> None:
        """Index canonical identity memories without turning them into scripted behavior."""

        identity_scope = str(identity.character_card_id or identity.name)

        def stable_id(kind: str, title: str) -> str:
            value = f"{self.owner_id}|{identity_scope}|{kind}|{title}"
            return f"autobio-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"

        background_content = " ".join(
            (
                str(identity.background),
                f"Aspiration: {identity.aspiration}",
                f"Core wound: {identity.core_wound}",
                f"Values: {', '.join(identity.values)}",
            )
        )
        self.store.upsert(
            MemoryDocument(
                owner_id=self.owner_id,
                id=stable_id("background", "identity_background"),
                turn=0,
                kind=MemoryKind.SEMANTIC.value,
                summary=_memory_summary(background_content, "", ()),
                content=background_content,
                action="",
                outcome_events=(),
                score_delta=0.0,
                importance=0.86,
                emotional_valence=0.0,
                surprise=0.0,
                categories=("autobiographical", "identity"),
                tags=("background", *identity.values),
                metadata={
                    "source": "canonical_identity_seed",
                    "epistemic_status": "canonical_identity",
                    "immutable": True,
                    "memory_role": "background",
                },
            )
        )
        for memory_role, memories, fact_ids in (
            (
                "formative",
                identity.formative_memories,
                formative_fact_ids or {},
            ),
            ("lived", identity.lived_memories, lived_fact_ids or {}),
        ):
            for memory in memories:
                content = " ".join(
                    part
                    for part in (
                        f"{memory.title}: {memory.recollection}",
                        f"Lesson: {memory.lesson}",
                        f"People: {', '.join(memory.people)}" if memory.people else "",
                        f"Place: {memory.place}" if memory.place else "",
                        f"Time: {memory.time_period}" if memory.time_period else "",
                    )
                    if part
                )
                tags = tuple(
                    dict.fromkeys(
                        (
                            memory.title,
                            *memory.people,
                            *memory.themes,
                            *((memory.place,) if memory.place else ()),
                        )
                    )
                )
                categories = tuple(
                    sorted(
                        set(_memory_categories(content, "", (), tags))
                        | {"autobiographical", "identity", f"{memory_role}_memory"}
                    )
                )
                self.store.upsert(
                    MemoryDocument(
                        owner_id=self.owner_id,
                        id=stable_id(memory_role, memory.title),
                        turn=0,
                        kind=MemoryKind.EPISODIC.value,
                        summary=_memory_summary(content, "", ()),
                        content=content,
                        action="",
                        outcome_events=(),
                        score_delta=0.0,
                        importance=0.9 if memory_role == "formative" else 0.76,
                        emotional_valence=memory.emotional_valence,
                        surprise=0.0,
                        categories=categories,
                        tags=tags,
                        metadata={
                            "source": "canonical_identity_seed",
                            "epistemic_status": "canonical_autobiographical_memory",
                            "immutable": True,
                            "memory_role": memory_role,
                            "supporting_fact_id": fact_ids.get(memory.title),
                        },
                    )
                )

    def record(
        self,
        *,
        turn: int,
        content: str,
        action: str,
        outcome_events: list[str],
        score_delta: float,
        surprise: float,
        emotional_intensity: float,
        tags: tuple[str, ...] = (),
        kind: MemoryKind = MemoryKind.EPISODIC,
        extra_categories: tuple[str, ...] = (),
        force_long_term: bool = False,
        track_procedural: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveMemory:
        importance = _clamp(
            0.18
            + 0.28 * min(1.0, abs(score_delta))
            + 0.30 * surprise
            + 0.24 * emotional_intensity
        )
        valence = max(-1.0, min(1.0, score_delta - 0.25 * surprise))
        event_tuple = tuple(outcome_events)
        summary = _memory_summary(content, action, event_tuple)
        categories = tuple(
            sorted(
                set(_memory_categories(content, action, event_tuple, tags))
                | set(extra_categories)
            )
        )
        self.forget_expired(turn)
        memory = CognitiveMemory(
            id=self.store.allocate_id(self.owner_id),
            turn=turn,
            kind=kind,
            content=content,
            action=action,
            outcome_events=event_tuple,
            score_delta=score_delta,
            importance=importance,
            emotional_valence=valence,
            surprise=surprise,
            tags=tags,
            summary=summary,
            categories=categories,
            epistemic_status=str((metadata or {}).get("epistemic_status", "experienced")),
            source=str((metadata or {}).get("source", "runtime_memory")),
            supporting_fact_id=(metadata or {}).get("supporting_fact_id"),
        )
        self.episodic.append(memory)
        if len(self.episodic) > self.short_term_limit:
            self._forgotten_short_term += len(self.episodic) - self.short_term_limit
            self.episodic = self.episodic[-self.short_term_limit :]
        promote = (
            force_long_term
            or importance >= self.long_term_promotion_threshold
            or surprise >= 0.8
            or bool({"identity", "relationship"} & set(categories))
        )
        if promote:
            memory.stored_long_term = True
            self.store.upsert(
                MemoryDocument(
                    owner_id=self.owner_id,
                    id=memory.id,
                    turn=memory.turn,
                    kind=memory.kind.value,
                    summary=memory.summary,
                    content=memory.content,
                    action=memory.action,
                    outcome_events=memory.outcome_events,
                    score_delta=memory.score_delta,
                    importance=memory.importance,
                    emotional_valence=memory.emotional_valence,
                    surprise=memory.surprise,
                    categories=memory.categories,
                    tags=memory.tags,
                    metadata={"source": "agent_experience", **(metadata or {})},
                )
            )
        if track_procedural and action:
            self.procedural.setdefault(action, []).append(score_delta - 0.15 * surprise)
            self.procedural[action] = self.procedural[action][-20:]
        self._unreflected_importance += importance
        return memory

    def retrieve(
        self,
        query: str,
        *,
        current_turn: int,
        mood_valence: float = 0.0,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        self.forget_expired(current_turn)
        query_tokens = _tokens(query)
        long_term_documents = self.store.query(
            self.owner_id,
            terms=query_tokens,
            recent_limit=max(12, limit * 4),
            candidate_limit=max(48, limit * 16),
        )
        long_term_documents = [
            document
            for document in long_term_documents
            if document.metadata.get("source") != "canonical_identity_seed"
            or bool(query_tokens & document.terms)
        ]
        long_term_ids = {document.id for document in long_term_documents}
        latest_order = max(
            (document.created_order for document in long_term_documents), default=0
        )
        candidates = {
            document.id: self._from_document(document)
            for document in long_term_documents
        }
        for document in long_term_documents:
            candidates[document.id].recency_distance = max(
                0, latest_order - document.created_order
            )
        candidates.update({memory.id: memory for memory in self.episodic})
        short_term_ids = {memory.id for memory in self.episodic}
        ranked: list[tuple[float, CognitiveMemory]] = []
        for memory in candidates.values():
            if memory.id in short_term_ids:
                age = max(0, current_turn - memory.turn)
                recency = math.exp(-age / 8)
            else:
                recency = math.exp(-memory.recency_distance / 8)
            memory_tokens = _tokens(
                " ".join(
                    (
                        memory.summary,
                        memory.content,
                        memory.action,
                        *memory.categories,
                        *memory.tags,
                    )
                )
            )
            relevance = (
                len(query_tokens & memory_tokens) / max(1, len(query_tokens | memory_tokens))
            )
            mood_match = 1 - min(1.0, abs(mood_valence - memory.emotional_valence) / 2)
            score = 0.34 * recency + 0.34 * memory.importance + 0.24 * relevance + 0.08 * mood_match
            ranked.append((score, memory))
        selected = sorted(ranked, key=lambda item: (-item[0], item[1].id))[:limit]
        for _score, memory in selected:
            memory.last_access_turn = current_turn
            memory.access_count += 1
            if memory.id in long_term_ids:
                self.store.touch(self.owner_id, memory.id, current_turn)
        return [memory.public_view(score) for score, memory in selected]

    def maybe_reflect(self, turn: int) -> Reflection | None:
        self.forget_expired(turn)
        available = list(reversed(self.episodic))
        evidence_ids = {memory.id for memory in available}
        for document in self.store.list_by_category(self.owner_id, "experience", limit=12):
            if document.id not in evidence_ids:
                available.append(self._from_document(document))
                evidence_ids.add(document.id)
        if self._unreflected_importance < self.reflection_threshold or len(available) < 2:
            return None
        evidence = sorted(available[:6], key=lambda item: item.importance, reverse=True)[:3]
        action_counts = Counter(item.action for item in evidence)
        action = action_counts.most_common(1)[0][0]
        mean_delta = sum(item.score_delta for item in evidence) / len(evidence)
        direction = (
            "helps"
            if mean_delta > 0.08
            else "hurts"
            if mean_delta < -0.08
            else "is unreliable"
        )
        previous = self.reflections[-1] if self.reflections else None
        reflection = Reflection(
            id=self.store.allocate_id(self.owner_id, "reflection"),
            turn=turn,
            belief=(
                f"Under recent conditions, {action} {direction}; "
                "treat this as a revisable belief."
            ),
            confidence=_clamp(0.42 + 0.12 * len(evidence) + 0.18 * abs(mean_delta)),
            evidence_memory_ids=tuple(item.id for item in evidence),
            revision_of=previous.id if previous and action in previous.belief else None,
        )
        self.reflections.append(reflection)
        self.reflections = self.reflections[-12:]
        self.store.upsert(
            MemoryDocument(
                owner_id=self.owner_id,
                id=reflection.id,
                turn=reflection.turn,
                kind=MemoryKind.SEMANTIC.value,
                summary=reflection.belief,
                content=reflection.belief,
                action=action,
                outcome_events=(),
                score_delta=mean_delta,
                importance=reflection.confidence,
                emotional_valence=max(-1.0, min(1.0, mean_delta)),
                surprise=0.0,
                categories=("reflection", "strategy"),
                tags=(action,),
                metadata={
                    "confidence": reflection.confidence,
                    "evidence_memory_ids": list(reflection.evidence_memory_ids),
                    "revision_of": reflection.revision_of,
                },
            )
        )
        self._unreflected_importance = 0.0
        return reflection

    def procedural_values(self) -> dict[str, float]:
        short_term_values = {
            action: sum(values) / len(values)
            for action, values in self.procedural.items()
            if values
        }
        durable_values = self.store.action_values(self.owner_id)
        for action, value in short_term_values.items():
            durable_values.setdefault(action, value)
        return durable_values

    def public_view(self) -> dict[str, Any]:
        return {
            "working_memory_policy": "current observation plus retrieved items",
            "short_term": {
                "active_count": len(self.episodic),
                "ttl_turns": self.short_term_ttl_turns,
                "capacity": self.short_term_limit,
                "forgotten_count": self._forgotten_short_term,
            },
            "long_term": self.store.diagnostics(self.owner_id),
            "episodic_count": len(self.episodic),
            "semantic_reflections": [item.public_view() for item in self.reflections[-4:]],
            "procedural_actions": sorted(self.procedural_values()),
            "retrieval_policy": "inverted-index candidates then salience reranking",
        }


@dataclass(frozen=True)
class AppraisalState:
    novelty: float
    goal_congruence: float
    controllability: float
    other_agency: float
    norm_compatibility: float
    social_threat: float
    coping: str
    reappraisal_target: str

    def public_view(self) -> dict[str, Any]:
        return {
            "novelty": round(self.novelty, 3),
            "goal_congruence": round(self.goal_congruence, 3),
            "controllability": round(self.controllability, 3),
            "other_agency": round(self.other_agency, 3),
            "norm_compatibility": round(self.norm_compatibility, 3),
            "social_threat": round(self.social_threat, 3),
            "coping": self.coping,
            "reappraisal_target": self.reappraisal_target,
        }


def appraise(
    *,
    score_delta: float,
    margin: float,
    surprise: float,
    mod_signals: dict[str, float],
    hostile_severity: float,
    uncertainty: float,
) -> AppraisalState:
    negative_mod = max(
        0.0,
        float(mod_signals.get("stress", 0.0)),
        float(mod_signals.get("frustration", 0.0)),
        float(mod_signals.get("anger", 0.0)),
        float(mod_signals.get("fear", 0.0)),
    )
    incongruence = _clamp(0.45 - 0.35 * score_delta - 0.12 * margin + 0.25 * negative_mod)
    controllability = _clamp(0.72 - 0.42 * uncertainty - 0.20 * surprise)
    social_threat = _clamp(0.72 * hostile_severity + 0.28 * negative_mod)
    if social_threat > 0.62 and controllability > 0.45:
        coping = "assert_boundary"
    elif social_threat > 0.55:
        coping = "protect_self"
    elif incongruence > 0.62 and controllability > 0.5:
        coping = "problem_solve"
    elif incongruence > 0.62:
        coping = "seek_information"
    else:
        coping = "maintain_course"
    return AppraisalState(
        novelty=_clamp(0.65 * surprise + 0.35 * uncertainty),
        goal_congruence=1 - incongruence,
        controllability=controllability,
        other_agency=_clamp(hostile_severity),
        norm_compatibility=1 - _clamp(0.8 * hostile_severity),
        social_threat=social_threat,
        coping=coping,
        reappraisal_target=(
            "separate criticism from identity"
            if social_threat > 0.5
            else "update plan from evidence"
        ),
    )


@dataclass
class SocialBelief:
    trust: float = 0.5
    respect: float = 0.5
    familiarity: float = 0.0
    perceived_hostility: float = 0.0
    perceived_sincerity: float = 0.5
    predicted_intent: str = "unknown"
    confidence: float = 0.0

    def update(self, *, hostile_severity: float, cooperative_signal: float, observed: bool) -> None:
        if not observed:
            return
        self.familiarity = _clamp(self.familiarity + 0.12)
        self.perceived_hostility = _clamp(
            0.62 * self.perceived_hostility + 0.38 * hostile_severity
        )
        self.trust = _clamp(
            0.78 * self.trust + 0.22 * (0.55 * cooperative_signal + 0.45 * (1 - hostile_severity))
        )
        self.respect = _clamp(0.85 * self.respect + 0.15 * (1 - 0.55 * hostile_severity))
        self.perceived_sincerity = _clamp(
            0.76 * self.perceived_sincerity + 0.24 * (1 - hostile_severity * 0.65)
        )
        self.predicted_intent = (
            "escalate" if self.perceived_hostility > 0.62 else "test_boundaries"
            if self.perceived_hostility > 0.35 else "cooperate_or_probe"
        )
        self.confidence = _clamp(self.familiarity * 0.9)

    def public_view(self) -> dict[str, Any]:
        return {
            "trust": round(self.trust, 3),
            "respect": round(self.respect, 3),
            "familiarity": round(self.familiarity, 3),
            "perceived_hostility": round(self.perceived_hostility, 3),
            "perceived_sincerity": round(self.perceived_sincerity, 3),
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PlanState:
    goal: str = "assess_situation"
    subgoal: str = "gather_evidence"
    commitment: float = 0.45
    age: int = 0
    revision: int = 0
    last_replan_reason: str = "initial"

    def update(
        self,
        *,
        desired_goal: str,
        surprise: float,
        stress: float,
        goal_congruence: float,
    ) -> bool:
        reason = ""
        if self.goal == "assess_situation" and desired_goal != self.goal:
            self.goal = desired_goal
            self.subgoal = "gather_evidence"
            self.age = 0
            self.last_replan_reason = "initial_commitment"
            return True
        if desired_goal != self.goal and self.age >= 1 and goal_congruence < 0.48:
            reason = "goal_incongruence"
        elif surprise > 0.72:
            reason = "prediction_failure"
        elif stress > 0.82:
            reason = "coping_overload"
        if reason:
            self.goal = desired_goal
            self.subgoal = "reduce_uncertainty" if surprise > 0.55 else "restore_control"
            self.commitment = _clamp(0.42 + 0.35 * goal_congruence)
            self.age = 0
            self.revision += 1
            self.last_replan_reason = reason
            return True
        self.age += 1
        self.commitment = _clamp(self.commitment + 0.04 * (goal_congruence - 0.45))
        self.last_replan_reason = "persisted"
        return False

    def public_view(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subgoal": self.subgoal,
            "commitment": round(self.commitment, 3),
            "age": self.age,
            "revision": self.revision,
            "last_replan_reason": self.last_replan_reason,
        }


def select_decision_mode(*, stress: float, uncertainty: float, stakes: float) -> DecisionMode:
    deliberation_need = 0.40 * uncertainty + 0.35 * stakes + 0.25 * (1 - stress)
    if stress > 0.82 or deliberation_need < 0.43:
        return DecisionMode.HABITUAL
    return DecisionMode.DELIBERATIVE
