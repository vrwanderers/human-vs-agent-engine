from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Protocol


def tokenize_memory_text(value: str) -> set[str]:
    """Tokenize Latin text and Chinese text without requiring an external segmenter."""

    tokens: set[str] = set()
    for segment in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value.lower()):
        if "\u4e00" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.add(segment)
            else:
                tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
                if len(segment) <= 8:
                    tokens.add(segment)
        elif len(segment) > 1:
            tokens.add(segment)
    return tokens


@dataclass
class MemoryDocument:
    owner_id: str
    id: str
    turn: int
    kind: str
    summary: str
    content: str
    action: str
    outcome_events: tuple[str, ...]
    score_delta: float
    importance: float
    emotional_valence: float
    surprise: float
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    last_access_turn: int = 0
    access_count: int = 0
    created_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def terms(self) -> set[str]:
        return tokenize_memory_text(
            " ".join(
                (
                    self.summary,
                    self.content,
                    self.action,
                    *self.outcome_events,
                    *self.categories,
                    *self.tags,
                )
            )
        )


class LongTermMemoryStore(Protocol):
    backend_name: str

    def allocate_id(self, owner_id: str, prefix: str = "memory") -> str: ...

    def upsert(self, document: MemoryDocument) -> None: ...

    def query(
        self,
        owner_id: str,
        *,
        terms: set[str],
        categories: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        recent_limit: int = 16,
        candidate_limit: int = 96,
    ) -> list[MemoryDocument]: ...

    def list_by_category(
        self, owner_id: str, category: str, *, limit: int = 16
    ) -> list[MemoryDocument]: ...

    def touch(self, owner_id: str, memory_id: str, current_turn: int) -> None: ...

    def count(self, owner_id: str) -> int: ...

    def action_values(self, owner_id: str) -> dict[str, float]: ...

    def diagnostics(self, owner_id: str) -> dict[str, Any]: ...


def _copy_document(document: MemoryDocument) -> MemoryDocument:
    return replace(document, metadata=dict(document.metadata))


class InMemoryIndexedMemoryStore:
    """Owner-scoped inverted index used by tests and ephemeral engine instances."""

    backend_name = "memory_index"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, MemoryDocument]] = defaultdict(dict)
        self._terms: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._categories: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._tags: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._recent_ids: dict[str, list[str]] = defaultdict(list)
        self._document_order: dict[str, dict[str, int]] = defaultdict(dict)
        self._next_document_order: dict[str, int] = defaultdict(int)
        self._action_stats: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
        self._sequences: dict[tuple[str, str], int] = defaultdict(int)
        self._last_query: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def allocate_id(self, owner_id: str, prefix: str = "memory") -> str:
        with self._lock:
            key = (owner_id, prefix)
            self._sequences[key] += 1
            return f"{prefix}-{self._sequences[key]:06d}"

    def _remove_indexes(self, document: MemoryDocument) -> None:
        owner_id = document.owner_id
        for term in document.terms:
            self._terms[owner_id][term].discard(document.id)
        for category in document.categories:
            self._categories[owner_id][category].discard(document.id)
        for tag in document.tags:
            self._tags[owner_id][tag].discard(document.id)
        if document.id in self._recent_ids[owner_id]:
            self._recent_ids[owner_id].remove(document.id)
        if document.action and document.kind == "episodic":
            total, count = self._action_stats[owner_id].get(document.action, (0.0, 0))
            value = document.score_delta - 0.15 * document.surprise
            if count <= 1:
                self._action_stats[owner_id].pop(document.action, None)
            else:
                self._action_stats[owner_id][document.action] = (total - value, count - 1)

    def upsert(self, document: MemoryDocument) -> None:
        with self._lock:
            previous = self._records[document.owner_id].get(document.id)
            if previous is not None:
                self._remove_indexes(previous)
            stored = _copy_document(document)
            if previous is None:
                self._next_document_order[stored.owner_id] += 1
                stored.created_order = self._next_document_order[stored.owner_id]
                self._document_order[stored.owner_id][stored.id] = stored.created_order
            else:
                stored.created_order = previous.created_order
            self._records[document.owner_id][document.id] = stored
            for term in stored.terms:
                self._terms[stored.owner_id][term].add(stored.id)
            for category in stored.categories:
                self._categories[stored.owner_id][category].add(stored.id)
            for tag in stored.tags:
                self._tags[stored.owner_id][tag].add(stored.id)
            self._recent_ids[stored.owner_id].append(stored.id)
            self._recent_ids[stored.owner_id].sort(
                key=self._document_order[stored.owner_id].__getitem__
            )
            if stored.action and stored.kind == "episodic":
                total, count = self._action_stats[stored.owner_id].get(
                    stored.action, (0.0, 0)
                )
                value = stored.score_delta - 0.15 * stored.surprise
                self._action_stats[stored.owner_id][stored.action] = (
                    total + value,
                    count + 1,
                )

    def query(
        self,
        owner_id: str,
        *,
        terms: set[str],
        categories: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        recent_limit: int = 16,
        candidate_limit: int = 96,
    ) -> list[MemoryDocument]:
        with self._lock:
            records = self._records.get(owner_id, {})
            match_counts: dict[str, int] = defaultdict(int)
            for term in terms:
                for memory_id in self._terms.get(owner_id, {}).get(term, ()):
                    match_counts[memory_id] += 1
            for category in categories:
                for memory_id in self._categories.get(owner_id, {}).get(category, ()):
                    match_counts[memory_id] += 2
            for tag in tags:
                for memory_id in self._tags.get(owner_id, {}).get(tag, ()):
                    match_counts[memory_id] += 2
            recent_ids = (
                list(reversed(self._recent_ids.get(owner_id, ())[-recent_limit:]))
                if recent_limit > 0
                else []
            )
            candidate_ids = set(match_counts) | set(recent_ids)
            ordered_ids = sorted(
                candidate_ids,
                key=lambda memory_id: (
                    -match_counts.get(memory_id, 0),
                    -records[memory_id].importance,
                    -records[memory_id].created_order,
                    memory_id,
                ),
            )[:candidate_limit]
            self._last_query[owner_id] = {
                "query_terms": len(terms),
                "indexed_matches": len(match_counts),
                "candidates_loaded": len(ordered_ids),
                "total_records": len(records),
                "full_scan": False,
            }
            return [_copy_document(records[memory_id]) for memory_id in ordered_ids]

    def list_by_category(
        self, owner_id: str, category: str, *, limit: int = 16
    ) -> list[MemoryDocument]:
        with self._lock:
            ids = self._categories.get(owner_id, {}).get(category, set())
            records = self._records.get(owner_id, {})
            selected = sorted(
                (records[memory_id] for memory_id in ids),
                key=lambda item: (-item.created_order, item.id),
            )[:limit]
            return [_copy_document(document) for document in selected]

    def touch(self, owner_id: str, memory_id: str, current_turn: int) -> None:
        with self._lock:
            document = self._records.get(owner_id, {}).get(memory_id)
            if document is None:
                return
            document.last_access_turn = current_turn
            document.access_count += 1

    def count(self, owner_id: str) -> int:
        with self._lock:
            return len(self._records.get(owner_id, {}))

    def action_values(self, owner_id: str) -> dict[str, float]:
        with self._lock:
            return {
                action: total / count
                for action, (total, count) in self._action_stats.get(owner_id, {}).items()
                if count
            }

    def diagnostics(self, owner_id: str) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend_name,
                "owner_scoped": True,
                "records": len(self._records.get(owner_id, {})),
                "term_index_size": len(self._terms.get(owner_id, {})),
                "category_index_size": len(self._categories.get(owner_id, {})),
                "tag_index_size": len(self._tags.get(owner_id, {})),
                "procedural_action_count": len(self._action_stats.get(owner_id, {})),
                "last_query": dict(self._last_query.get(owner_id, {})),
            }


class SQLiteIndexedMemoryStore:
    """Durable normalized memory store; content and indexes are queried independently."""

    backend_name = "sqlite_index"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._last_query: dict[str, dict[str, Any]] = {}
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memory_sequences (
                    owner_id TEXT NOT NULL,
                    prefix TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, prefix)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    owner_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome_events TEXT NOT NULL,
                    score_delta REAL NOT NULL,
                    importance REAL NOT NULL,
                    emotional_valence REAL NOT NULL,
                    surprise REAL NOT NULL,
                    last_access_turn INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    created_order INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    PRIMARY KEY (owner_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_owner_turn
                    ON memories (owner_id, turn DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_owner_action
                    ON memories (owner_id, action);
                CREATE TABLE IF NOT EXISTS memory_terms (
                    owner_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (owner_id, term, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_terms_lookup
                    ON memory_terms (owner_id, term);
                CREATE TABLE IF NOT EXISTS memory_categories (
                    owner_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (owner_id, category, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_categories_lookup
                    ON memory_categories (owner_id, category);
                CREATE TABLE IF NOT EXISTS memory_tags (
                    owner_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (owner_id, tag, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_tags_lookup
                    ON memory_tags (owner_id, tag);
                CREATE TABLE IF NOT EXISTS memory_action_stats (
                    owner_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, action)
                );
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "created_order" not in columns:
                self._connection.execute(
                    "ALTER TABLE memories ADD COLUMN created_order INTEGER NOT NULL DEFAULT 0"
                )
                self._connection.execute(
                    """
                    UPDATE memories SET created_order = rowid
                    WHERE created_order = 0
                    """
                )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_owner_created_order
                ON memories (owner_id, created_order DESC)
                """
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO memory_action_stats (
                    owner_id, action, total_value, sample_count
                )
                SELECT owner_id, action, SUM(score_delta - 0.15 * surprise), COUNT(*)
                FROM memories
                WHERE action != '' AND kind = 'episodic'
                GROUP BY owner_id, action
                """
            )

    def allocate_id(self, owner_id: str, prefix: str = "memory") -> str:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT value FROM memory_sequences WHERE owner_id = ? AND prefix = ?",
                (owner_id, prefix),
            ).fetchone()
            value = int(row["value"]) + 1 if row else 1
            self._connection.execute(
                """
                INSERT INTO memory_sequences (owner_id, prefix, value) VALUES (?, ?, ?)
                ON CONFLICT(owner_id, prefix) DO UPDATE SET value = excluded.value
                """,
                (owner_id, prefix, value),
            )
            return f"{prefix}-{value:06d}"

    def upsert(self, document: MemoryDocument) -> None:
        with self._lock, self._connection:
            values = (document.owner_id, document.id)
            existing = self._connection.execute(
                """
                SELECT created_order, kind, action, score_delta, surprise
                FROM memories WHERE owner_id = ? AND id = ?
                """,
                values,
            ).fetchone()
            if existing is None:
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(created_order), 0) + 1 AS value "
                    "FROM memories WHERE owner_id = ?",
                    (document.owner_id,),
                ).fetchone()
                created_order = int(row["value"])
            else:
                created_order = int(existing["created_order"])
                if str(existing["kind"]) == "episodic" and str(existing["action"]):
                    old_value = float(existing["score_delta"]) - 0.15 * float(
                        existing["surprise"]
                    )
                    self._connection.execute(
                        """
                        UPDATE memory_action_stats
                        SET total_value = total_value - ?, sample_count = sample_count - 1
                        WHERE owner_id = ? AND action = ?
                        """,
                        (old_value, document.owner_id, str(existing["action"])),
                    )
                    self._connection.execute(
                        "DELETE FROM memory_action_stats "
                        "WHERE owner_id = ? AND action = ? AND sample_count <= 0",
                        (document.owner_id, str(existing["action"])),
                    )
            self._connection.execute(
                "DELETE FROM memory_terms WHERE owner_id = ? AND memory_id = ?", values
            )
            self._connection.execute(
                "DELETE FROM memory_categories WHERE owner_id = ? AND memory_id = ?", values
            )
            self._connection.execute(
                "DELETE FROM memory_tags WHERE owner_id = ? AND memory_id = ?", values
            )
            self._connection.execute(
                """
                INSERT INTO memories (
                    owner_id, id, turn, kind, summary, content, action, outcome_events,
                    score_delta, importance, emotional_valence, surprise,
                    last_access_turn, access_count, created_order, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, id) DO UPDATE SET
                    turn=excluded.turn, kind=excluded.kind, summary=excluded.summary,
                    content=excluded.content, action=excluded.action,
                    outcome_events=excluded.outcome_events, score_delta=excluded.score_delta,
                    importance=excluded.importance,
                    emotional_valence=excluded.emotional_valence,
                    surprise=excluded.surprise, last_access_turn=excluded.last_access_turn,
                    access_count=excluded.access_count, metadata=excluded.metadata
                """,
                (
                    document.owner_id,
                    document.id,
                    document.turn,
                    document.kind,
                    document.summary,
                    document.content,
                    document.action,
                    json.dumps(document.outcome_events, ensure_ascii=False),
                    document.score_delta,
                    document.importance,
                    document.emotional_valence,
                    document.surprise,
                    document.last_access_turn,
                    document.access_count,
                    created_order,
                    json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._connection.executemany(
                "INSERT OR IGNORE INTO memory_terms VALUES (?, ?, ?)",
                ((document.owner_id, term, document.id) for term in document.terms),
            )
            self._connection.executemany(
                "INSERT OR IGNORE INTO memory_categories VALUES (?, ?, ?)",
                (
                    (document.owner_id, category, document.id)
                    for category in document.categories
                ),
            )
            self._connection.executemany(
                "INSERT OR IGNORE INTO memory_tags VALUES (?, ?, ?)",
                ((document.owner_id, tag, document.id) for tag in document.tags),
            )
            if document.kind == "episodic" and document.action:
                action_value = document.score_delta - 0.15 * document.surprise
                self._connection.execute(
                    """
                    INSERT INTO memory_action_stats (
                        owner_id, action, total_value, sample_count
                    ) VALUES (?, ?, ?, 1)
                    ON CONFLICT(owner_id, action) DO UPDATE SET
                        total_value = total_value + excluded.total_value,
                        sample_count = sample_count + 1
                    """,
                    (document.owner_id, document.action, action_value),
                )

    @staticmethod
    def _placeholders(values: set[str] | tuple[str, ...]) -> str:
        return ",".join("?" for _ in values)

    def _indexed_ids(
        self,
        table: str,
        column: str,
        owner_id: str,
        values: set[str] | tuple[str, ...],
    ) -> list[str]:
        if not values:
            return []
        ordered = sorted(values)
        rows = self._connection.execute(
            f"SELECT memory_id FROM {table} "  # noqa: S608 - table and column are internal
            f"WHERE owner_id = ? AND {column} IN ({self._placeholders(values)})",
            (owner_id, *ordered),
        ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def _documents(self, owner_id: str, memory_ids: list[str]) -> list[MemoryDocument]:
        if not memory_ids:
            return []
        rows = self._connection.execute(
            f"SELECT * FROM memories WHERE owner_id = ? "  # noqa: S608
            f"AND id IN ({self._placeholders(tuple(memory_ids))})",
            (owner_id, *memory_ids),
        ).fetchall()
        categories: dict[str, list[str]] = defaultdict(list)
        tags: dict[str, list[str]] = defaultdict(list)
        for row in self._connection.execute(
            f"SELECT memory_id, category FROM memory_categories WHERE owner_id = ? "  # noqa: S608
            f"AND memory_id IN ({self._placeholders(tuple(memory_ids))})",
            (owner_id, *memory_ids),
        ).fetchall():
            categories[str(row["memory_id"])].append(str(row["category"]))
        for row in self._connection.execute(
            f"SELECT memory_id, tag FROM memory_tags WHERE owner_id = ? "  # noqa: S608
            f"AND memory_id IN ({self._placeholders(tuple(memory_ids))})",
            (owner_id, *memory_ids),
        ).fetchall():
            tags[str(row["memory_id"])].append(str(row["tag"]))
        return [
            MemoryDocument(
                owner_id=str(row["owner_id"]),
                id=str(row["id"]),
                turn=int(row["turn"]),
                kind=str(row["kind"]),
                summary=str(row["summary"]),
                content=str(row["content"]),
                action=str(row["action"]),
                outcome_events=tuple(json.loads(str(row["outcome_events"]))),
                score_delta=float(row["score_delta"]),
                importance=float(row["importance"]),
                emotional_valence=float(row["emotional_valence"]),
                surprise=float(row["surprise"]),
                categories=tuple(sorted(categories[str(row["id"])])),
                tags=tuple(sorted(tags[str(row["id"])])),
                last_access_turn=int(row["last_access_turn"]),
                access_count=int(row["access_count"]),
                created_order=int(row["created_order"]),
                metadata=dict(json.loads(str(row["metadata"]))),
            )
            for row in rows
        ]

    def query(
        self,
        owner_id: str,
        *,
        terms: set[str],
        categories: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        recent_limit: int = 16,
        candidate_limit: int = 96,
    ) -> list[MemoryDocument]:
        with self._lock:
            match_counts: dict[str, int] = defaultdict(int)
            for memory_id in self._indexed_ids(
                "memory_terms", "term", owner_id, terms
            ):
                match_counts[memory_id] += 1
            for memory_id in self._indexed_ids(
                "memory_categories", "category", owner_id, categories
            ):
                match_counts[memory_id] += 2
            for memory_id in self._indexed_ids("memory_tags", "tag", owner_id, tags):
                match_counts[memory_id] += 2
            recent_ids = [
                str(row["id"])
                for row in self._connection.execute(
                    "SELECT id FROM memories WHERE owner_id = ? "
                    "ORDER BY created_order DESC LIMIT ?",
                    (owner_id, recent_limit),
                ).fetchall()
            ]
            candidate_ids = list(set(match_counts) | set(recent_ids))
            documents = self._documents(owner_id, candidate_ids)
            documents.sort(
                key=lambda item: (
                    -match_counts.get(item.id, 0),
                    -item.importance,
                    -item.created_order,
                    item.id,
                )
            )
            documents = documents[:candidate_limit]
            total = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM memories WHERE owner_id = ?", (owner_id,)
                ).fetchone()["count"]
            )
            self._last_query[owner_id] = {
                "query_terms": len(terms),
                "indexed_matches": len(match_counts),
                "candidates_loaded": len(documents),
                "total_records": total,
                "full_scan": False,
            }
            return documents

    def list_by_category(
        self, owner_id: str, category: str, *, limit: int = 16
    ) -> list[MemoryDocument]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT category.memory_id FROM memory_categories AS category
                JOIN memories AS memory
                  ON memory.owner_id = category.owner_id AND memory.id = category.memory_id
                WHERE category.owner_id = ? AND category.category = ?
                ORDER BY memory.created_order DESC LIMIT ?
                """,
                (owner_id, category, limit),
            ).fetchall()
            documents = self._documents(
                owner_id, [str(row["memory_id"]) for row in rows]
            )
            return sorted(
                documents, key=lambda item: (-item.created_order, item.id)
            )[:limit]

    def touch(self, owner_id: str, memory_id: str, current_turn: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE memories SET last_access_turn = ?, access_count = access_count + 1
                WHERE owner_id = ? AND id = ?
                """,
                (current_turn, owner_id, memory_id),
            )

    def count(self, owner_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM memories WHERE owner_id = ?", (owner_id,)
            ).fetchone()
            return int(row["count"])

    def action_values(self, owner_id: str) -> dict[str, float]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT action, total_value / sample_count AS value
                FROM memory_action_stats WHERE owner_id = ? AND sample_count > 0
                """,
                (owner_id,),
            ).fetchall()
            return {str(row["action"]): float(row["value"]) for row in rows}

    def diagnostics(self, owner_id: str) -> dict[str, Any]:
        with self._lock:
            def count_table(table: str) -> int:
                row = self._connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE owner_id = ?",  # noqa: S608
                    (owner_id,),
                ).fetchone()
                return int(row["count"])

            return {
                "backend": self.backend_name,
                "owner_scoped": True,
                "records": count_table("memories"),
                "term_index_size": count_table("memory_terms"),
                "category_index_size": count_table("memory_categories"),
                "tag_index_size": count_table("memory_tags"),
                "procedural_action_count": count_table("memory_action_stats"),
                "last_query": dict(self._last_query.get(owner_id, {})),
            }

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def build_memory_store_from_env() -> LongTermMemoryStore:
    backend = os.environ.get("HVA_MEMORY_STORE", "memory").strip().lower()
    if backend in {"memory", "in-memory", "indexed-memory"}:
        return InMemoryIndexedMemoryStore()
    if backend == "sqlite":
        return SQLiteIndexedMemoryStore(
            os.environ.get("HVA_MEMORY_SQLITE_PATH", "data/memory/hva-memory.sqlite3")
        )
    raise ValueError(f"Unsupported HVA_MEMORY_STORE: {backend}")
