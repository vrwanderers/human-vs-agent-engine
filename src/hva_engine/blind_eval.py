from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from random import Random
from threading import RLock
from typing import Any, Protocol


@dataclass(frozen=True)
class BlindSample:
    condition_id: str
    transcript: tuple[str, ...]
    metadata: dict[str, Any]


class BlindEvaluationStore(Protocol):
    backend_name: str

    def create_trial(
        self, study_id: str, sample_a: BlindSample, sample_b: BlindSample, seed: int
    ) -> dict[str, Any]: ...

    def submit_rating(self, rating: dict[str, Any]) -> dict[str, Any]: ...

    def summary(self, study_id: str) -> dict[str, Any]: ...

    def get_trial(self, trial_id: str) -> dict[str, Any] | None: ...


class InMemoryBlindEvaluationStore:
    backend_name = "blind_eval_memory"

    def __init__(self) -> None:
        self._trials: dict[str, dict[str, Any]] = {}
        self._ratings: list[dict[str, Any]] = []
        self._lock = RLock()

    def create_trial(
        self, study_id: str, sample_a: BlindSample, sample_b: BlindSample, seed: int
    ) -> dict[str, Any]:
        trial = _build_trial(study_id, sample_a, sample_b, seed)
        with self._lock:
            self._trials[trial["trial_id"]] = trial
        return _public_trial(trial)

    def submit_rating(self, rating: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            trial = self._trials.get(str(rating["trial_id"]))
            if trial is None or trial["study_id"] != rating["study_id"]:
                raise ValueError("Unknown blind trial")
            rater_hash = sha256(str(rating["rater_id"]).encode()).hexdigest()[:20]
            if any(
                item["trial_id"] == rating["trial_id"]
                and item["rater_hash"] == rater_hash
                for item in self._ratings
            ):
                raise ValueError("This rater already rated the trial")
            stored = {**rating, "rater_hash": rater_hash}
            stored.pop("rater_id", None)
            self._ratings.append(stored)
            return {"accepted": True, "trial_id": rating["trial_id"]}

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        with self._lock:
            trial = self._trials.get(trial_id)
        return _public_trial(trial) if trial else None

    def summary(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            trials = {
                trial_id: trial
                for trial_id, trial in self._trials.items()
                if trial["study_id"] == study_id
            }
            ratings = [
                dict(rating)
                for rating in self._ratings
                if rating["study_id"] == study_id
                and rating["trial_id"] in trials
            ]
        return _aggregate(study_id, trials, ratings)


class SQLiteBlindEvaluationStore(InMemoryBlindEvaluationStore):
    backend_name = "blind_eval_sqlite"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS blind_trials (
                    trial_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    trial_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blind_ratings (
                    trial_id TEXT NOT NULL,
                    study_id TEXT NOT NULL,
                    rater_hash TEXT NOT NULL,
                    rating_json TEXT NOT NULL,
                    PRIMARY KEY (trial_id, rater_hash)
                );
                """
            )

    def create_trial(
        self, study_id: str, sample_a: BlindSample, sample_b: BlindSample, seed: int
    ) -> dict[str, Any]:
        trial = _build_trial(study_id, sample_a, sample_b, seed)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO blind_trials VALUES (?, ?, ?)",
                (
                    trial["trial_id"],
                    study_id,
                    json.dumps(trial, ensure_ascii=False, sort_keys=True),
                ),
            )
        return _public_trial(trial)

    def submit_rating(self, rating: dict[str, Any]) -> dict[str, Any]:
        trial = self._load_trial(str(rating["trial_id"]))
        if trial is None or trial["study_id"] != rating["study_id"]:
            raise ValueError("Unknown blind trial")
        rater_hash = sha256(str(rating["rater_id"]).encode()).hexdigest()[:20]
        stored = {**rating, "rater_hash": rater_hash}
        stored.pop("rater_id", None)
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    "INSERT INTO blind_ratings VALUES (?, ?, ?, ?)",
                    (
                        rating["trial_id"],
                        rating["study_id"],
                        rater_hash,
                        json.dumps(stored, ensure_ascii=False, sort_keys=True),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("This rater already rated the trial") from exc
        return {"accepted": True, "trial_id": rating["trial_id"]}

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        trial = self._load_trial(trial_id)
        return _public_trial(trial) if trial else None

    def summary(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            trial_rows = self._connection.execute(
                "SELECT trial_json FROM blind_trials WHERE study_id = ?", (study_id,)
            ).fetchall()
            rating_rows = self._connection.execute(
                "SELECT rating_json FROM blind_ratings WHERE study_id = ?", (study_id,)
            ).fetchall()
        trials = {
            trial["trial_id"]: trial
            for row in trial_rows
            if (trial := dict(json.loads(str(row["trial_json"]))))
        }
        ratings = [dict(json.loads(str(row["rating_json"]))) for row in rating_rows]
        return _aggregate(study_id, trials, ratings)

    def _load_trial(self, trial_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT trial_json FROM blind_trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
        return dict(json.loads(str(row["trial_json"]))) if row else None


def _build_trial(
    study_id: str, sample_a: BlindSample, sample_b: BlindSample, seed: int
) -> dict[str, Any]:
    rng = Random(seed)
    ordered = [sample_a, sample_b]
    rng.shuffle(ordered)
    digest = sha256(
        f"{study_id}|{seed}|{sample_a.condition_id}|{sample_b.condition_id}".encode()
    ).hexdigest()[:20]
    return {
        "trial_id": f"trial-{digest}",
        "study_id": study_id,
        "samples": {
            label: {
                "condition_id": sample.condition_id,
                "transcript": list(sample.transcript),
                "metadata": sample.metadata,
            }
            for label, sample in zip(("A", "B"), ordered, strict=True)
        },
        "blinding": "condition_labels_hidden_from_raters",
    }


def _public_trial(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial["trial_id"],
        "study_id": trial["study_id"],
        "samples": {
            label: {
                "transcript": sample["transcript"],
                "metadata": {
                    key: value
                    for key, value in sample.get("metadata", {}).items()
                    if key in {"seed", "rounds", "language", "scenario"}
                },
            }
            for label, sample in trial["samples"].items()
        },
        "blinding": trial["blinding"],
    }


def _aggregate(
    study_id: str,
    trials: dict[str, dict[str, Any]],
    ratings: list[dict[str, Any]],
) -> dict[str, Any]:
    dimensions = ("naturalness", "identity_consistency", "contextual_fit", "dramatic_interest")
    values: dict[str, dict[str, list[float]]] = {}
    preferences: dict[str, int] = {}
    ties = 0
    for rating in ratings:
        trial = trials[rating["trial_id"]]
        for label in ("A", "B"):
            condition = trial["samples"][label]["condition_id"]
            bucket = values.setdefault(condition, {key: [] for key in dimensions})
            for dimension in dimensions:
                bucket[dimension].append(float(rating[f"{label.lower()}_{dimension}"]))
        preferred = rating["preferred"]
        if preferred == "tie":
            ties += 1
        else:
            condition = trial["samples"][preferred]["condition_id"]
            preferences[condition] = preferences.get(condition, 0) + 1
    condition_rows = {
        condition: {
            "ratings": len(next(iter(bucket.values()), [])),
            "dimensions": {
                key: round(sum(items) / len(items), 3) if items else None
                for key, items in bucket.items()
            },
            "preference_wins": preferences.get(condition, 0),
        }
        for condition, bucket in values.items()
    }
    return {
        "study_id": study_id,
        "trials": len(trials),
        "ratings": len(ratings),
        "conditions": condition_rows,
        "ties": ties,
        "calibration_status": "usable" if len(ratings) >= 12 else "insufficient_human_ratings",
        "minimum_recommended_ratings": 12,
        "semantics": "human_blind_judgment_not_engine_proxy",
    }


def build_blind_evaluation_store_from_env() -> BlindEvaluationStore:
    backend = os.environ.get("HVA_BLIND_EVAL_STORE", "memory").lower()
    if backend == "memory":
        return InMemoryBlindEvaluationStore()
    if backend == "sqlite":
        return SQLiteBlindEvaluationStore(
            os.environ.get("HVA_BLIND_EVAL_SQLITE_PATH", "data/evaluation/hva-blind.sqlite3")
        )
    raise ValueError(f"Unsupported HVA_BLIND_EVAL_STORE backend: {backend}")
