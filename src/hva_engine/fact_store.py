from __future__ import annotations

import json
import os
from typing import Any, Protocol


class FactStore(Protocol):
    name: str

    def save_fact(self, owner_id: str, fact: Any) -> None: ...

    def save_rejection(self, owner_id: str, rejection: dict[str, Any]) -> None: ...


class InMemoryFactStore:
    name = "memory"

    def __init__(self) -> None:
        self.facts: dict[str, dict[str, Any]] = {}
        self.rejections: list[dict[str, Any]] = []

    def save_fact(self, owner_id: str, fact: Any) -> None:
        self.facts[f"{owner_id}:{fact.id}"] = fact.view()

    def save_rejection(self, owner_id: str, rejection: dict[str, Any]) -> None:
        self.rejections.append({"owner_id": owner_id, **rejection})


class Neo4jFactStore:
    """Optional durable adapter. Conflict validation stays in AgentFactGraph."""

    name = "neo4j"

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Neo4j storage requires the optional dependency: pip install -e '.[neo4j]'"
            ) from exc
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._ensure_schema()

    @classmethod
    def from_env(cls) -> Neo4jFactStore:
        uri = os.environ.get("HVA_NEO4J_URI")
        user = os.environ.get("HVA_NEO4J_USER")
        password = os.environ.get("HVA_NEO4J_PASSWORD")
        if not uri or not user or not password:
            raise ValueError("HVA_NEO4J_URI, HVA_NEO4J_USER, and HVA_NEO4J_PASSWORD are required")
        return cls(
            uri=uri,
            user=user,
            password=password,
            database=os.environ.get("HVA_NEO4J_DATABASE", "neo4j"),
        )

    def _ensure_schema(self) -> None:
        with self._driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT hva_agent_id IF NOT EXISTS "
                "FOR (a:HVAAgent) REQUIRE a.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT hva_fact_uid IF NOT EXISTS "
                "FOR (f:HVAFact) REQUIRE f.uid IS UNIQUE"
            ).consume()

    def save_fact(self, owner_id: str, fact: Any) -> None:
        uid = f"{owner_id}:{fact.id}"
        properties = {
            "uid": uid,
            "fact_id": fact.id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object_json": json.dumps(fact.object, ensure_ascii=False, default=str),
            "confidence": fact.confidence,
            "source": fact.source,
            "immutable": fact.immutable,
            "visibility": fact.visibility.value,
            "status": fact.status.value,
            "revision": fact.revision,
        }
        with self._driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (a:HVAAgent {id: $owner_id})
                MERGE (f:HVAFact {uid: $uid})
                SET f += $properties
                MERGE (a)-[:OWNS_FACT]->(f)
                """,
                owner_id=owner_id,
                uid=uid,
                properties=properties,
            ).consume()
            if fact.supersedes:
                session.run(
                    """
                    MATCH (new:HVAFact {uid: $new_uid})
                    MATCH (old:HVAFact {uid: $old_uid})
                    MERGE (new)-[:SUPERSEDES]->(old)
                    """,
                    new_uid=uid,
                    old_uid=f"{owner_id}:{fact.supersedes}",
                ).consume()

    def save_rejection(self, owner_id: str, rejection: dict[str, Any]) -> None:
        with self._driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (a:HVAAgent {id: $owner_id})
                CREATE (r:HVAFactRejection {
                    predicate: $predicate,
                    proposal_json: $proposal_json,
                    reason: $reason,
                    created_at: datetime()
                })
                MERGE (a)-[:REJECTED_FACT]->(r)
                """,
                owner_id=owner_id,
                predicate=str(rejection.get("predicate", "")),
                proposal_json=json.dumps(rejection, ensure_ascii=False, default=str),
                reason=str(rejection.get("reason", "")),
            ).consume()

    def close(self) -> None:
        self._driver.close()


def build_fact_store_from_env() -> FactStore:
    backend = os.environ.get("HVA_FACT_STORE", "memory").strip().lower()
    if backend == "memory":
        return InMemoryFactStore()
    if backend == "neo4j":
        return Neo4jFactStore.from_env()
    raise ValueError(f"Unknown HVA_FACT_STORE backend: {backend}")
