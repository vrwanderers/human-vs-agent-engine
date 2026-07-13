from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Any

from hva_engine.models import Action, Player


class GameMod(ABC):
    """A deterministic state machine contract implemented by every MOD."""

    id: str
    display_name: str
    description: str
    tags: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset({"turn_based", "numeric_state"})
    supported_modes: tuple[str, ...] = ("human_vs_agent", "agent_vs_agent")

    @abstractmethod
    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]: ...

    @abstractmethod
    def current_player_id(self, state: dict[str, Any]) -> str | None: ...

    @abstractmethod
    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]: ...

    @abstractmethod
    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...

    @abstractmethod
    def is_terminal(self, state: dict[str, Any]) -> bool: ...

    @abstractmethod
    def scores(self, state: dict[str, Any]) -> dict[str, float]: ...

    def public_state(self, state: dict[str, Any], _viewer_id: str | None = None) -> dict[str, Any]:
        return state

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        """Default baseline policy. MODs override this with domain heuristics."""
        return rng.choice(legal)

    def agent_psychological_signals(self, state: dict[str, Any], actor_id: str) -> dict[str, float]:
        """Optional bounded adjustments applied to the Agent psychological matrix."""
        return {}

    def agent_narrative_affordances(
        self, state: dict[str, Any], actor_id: str, legal: list[Action]
    ) -> dict[str, dict[str, Any]]:
        """Optional value, relationship, and delayed-cost metadata for legal actions."""
        return {}

    def agent_influence_affordances(
        self, state: dict[str, Any], actor_id: str, legal: list[Action]
    ) -> dict[str, dict[str, Any]]:
        """Optional opportunities/risks for continuous, game-world social influence."""
        return {}

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.display_name,
            "description": self.description,
            "tags": list(self.tags),
            "capabilities": sorted(self.capabilities),
            "supported_modes": list(self.supported_modes),
        }
