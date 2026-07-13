from __future__ import annotations

import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol


@dataclass(frozen=True)
class WorldSnapshot:
    world_id: str
    mod_id: str
    revision: int
    state: dict[str, Any]
    saved_at: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "mod_id": self.mod_id,
            "revision": self.revision,
            "saved_at": self.saved_at,
        }


class WorldStateStore(Protocol):
    backend_name: str

    def save(self, world_id: str, mod_id: str, state: dict[str, Any]) -> WorldSnapshot: ...

    def load(self, world_id: str) -> WorldSnapshot | None: ...

    def metadata(self, world_id: str) -> dict[str, Any] | None: ...


class InMemoryWorldStateStore:
    backend_name = "world_memory"

    def __init__(self) -> None:
        self._snapshots: dict[str, WorldSnapshot] = {}
        self._lock = RLock()

    def save(self, world_id: str, mod_id: str, state: dict[str, Any]) -> WorldSnapshot:
        with self._lock:
            previous = self._snapshots.get(world_id)
            snapshot = WorldSnapshot(
                world_id=world_id,
                mod_id=mod_id,
                revision=previous.revision + 1 if previous else 1,
                state=deepcopy(state),
                saved_at=datetime.now(UTC).isoformat(),
            )
            self._snapshots[world_id] = snapshot
            return deepcopy(snapshot)

    def load(self, world_id: str) -> WorldSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(world_id)
            return deepcopy(snapshot) if snapshot else None

    def metadata(self, world_id: str) -> dict[str, Any] | None:
        snapshot = self.load(world_id)
        return snapshot.public_metadata() if snapshot else None


class SQLiteWorldStateStore:
    backend_name = "world_sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    world_id TEXT PRIMARY KEY,
                    mod_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_world_snapshots_mod
                    ON world_snapshots (mod_id, saved_at DESC);
                """
            )

    def save(self, world_id: str, mod_id: str, state: dict[str, Any]) -> WorldSnapshot:
        saved_at = datetime.now(UTC).isoformat()
        state_json = json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT revision FROM world_snapshots WHERE world_id = ?", (world_id,)
            ).fetchone()
            revision = int(row["revision"]) + 1 if row else 1
            self._connection.execute(
                """
                INSERT INTO world_snapshots (
                    world_id, mod_id, revision, state_json, saved_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(world_id) DO UPDATE SET
                    mod_id=excluded.mod_id,
                    revision=excluded.revision,
                    state_json=excluded.state_json,
                    saved_at=excluded.saved_at
                """,
                (world_id, mod_id, revision, state_json, saved_at),
            )
        return WorldSnapshot(world_id, mod_id, revision, deepcopy(state), saved_at)

    def load(self, world_id: str) -> WorldSnapshot | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM world_snapshots WHERE world_id = ?", (world_id,)
            ).fetchone()
        if row is None:
            return None
        return WorldSnapshot(
            world_id=str(row["world_id"]),
            mod_id=str(row["mod_id"]),
            revision=int(row["revision"]),
            state=dict(json.loads(str(row["state_json"]))),
            saved_at=str(row["saved_at"]),
        )

    def metadata(self, world_id: str) -> dict[str, Any] | None:
        snapshot = self.load(world_id)
        return snapshot.public_metadata() if snapshot else None


def build_world_store_from_env() -> WorldStateStore:
    backend = os.environ.get("HVA_WORLD_STORE", "memory").strip().lower()
    if backend == "memory":
        return InMemoryWorldStateStore()
    if backend == "sqlite":
        return SQLiteWorldStateStore(
            os.environ.get("HVA_WORLD_SQLITE_PATH", "data/worlds/hva-worlds.sqlite3")
        )
    raise ValueError(f"Unsupported HVA_WORLD_STORE backend: {backend}")
