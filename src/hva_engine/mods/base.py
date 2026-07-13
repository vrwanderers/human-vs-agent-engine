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
    competitive_balance_applicable: bool = True
    score_ceiling: float = 2.0
    cooperation_event_types: tuple[str, ...] = ("coordination_bonus",)
    supports_persistent_world: bool = False

    def agent_count_for_mode(self, mode: str) -> int:
        """Number of independent AgentBrain participants requested by this MOD."""

        return 1 if mode in {"human_vs_agent", "human_agent_coop"} else 2

    def agent_character(
        self,
        state: dict[str, Any],
        player: Player,
        role: str,
        behavior_policy: Any,
        memory_owner_id: str,
        rng: Random,
    ) -> tuple[Any, Any] | None:
        """Optional generated CognitiveProfile/AgentIdentity pair for this world."""

        return None

    def agent_world_model(
        self, state: dict[str, Any], actor_id: str
    ) -> dict[str, Any]:
        """MOD-specific environmental facts visible to one Agent."""

        return {}

    def persistent_world_state(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Return a JSON-serializable public-world snapshot, excluding Agent privacy."""

        return None

    def restore_persistent_world(
        self, state: dict[str, Any], snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a stored world into a fresh match state."""

        return state

    def social_platform_manifest(self) -> list[dict[str, Any]]:
        """Optional MOD configuration for the shared social-media compatibility layer."""

        return []

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

    def public_action(self, action: Action, _actor_id: str) -> Action:
        """Remove engine-only decision annotations from a publicly observable action."""

        payload = {
            key: value
            for key, value in action.payload.items()
            if key != "response_plan" and not key.startswith("_")
        }
        return action.model_copy(update={"payload": payload})

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        """Default baseline policy. MODs override this with domain heuristics."""
        return rng.choice(legal)

    def agent_utterance(
        self, state: dict[str, Any], actor_id: str, action: Action
    ) -> str | None:
        """Optional semantic baseline text; the engine applies the character's speech style."""

        return None

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

    def agent_skill_id(self, action: Action) -> str:
        """Stable procedural skill family; MODs may separate semantically distinct actions."""

        return action.type

    def agent_skill_context(
        self, state: dict[str, Any], actor_id: str, action: Action
    ) -> dict[str, Any]:
        """Stable context used to decide whether a learned skill transfers.

        Volatile values such as turn, score, health, and pressure are deliberately excluded.
        Navigation/work MODs should expose location, route, task, toolset, or environment
        version here so a familiar skill does not become automatic in a new place by accident.
        """

        context: dict[str, Any] = {"mod": self.id, "role": "actor"}
        stable_state_keys = (
            "location",
            "location_id",
            "route",
            "route_id",
            "region",
            "terrain",
            "weather",
            "phase",
            "job",
            "job_id",
            "task",
            "task_id",
            "toolset",
            "vehicle",
            "environment_version",
        )
        stable_payload_keys = (
            "destination",
            "location_id",
            "route_id",
            "job_id",
            "task_id",
            "toolset",
            "vehicle",
        )
        for key in stable_state_keys:
            if key in state and isinstance(state[key], (str, int, float, bool)):
                context[key] = state[key]
        for key in stable_payload_keys:
            if key in action.payload and isinstance(
                action.payload[key], (str, int, float, bool)
            ):
                context[key] = action.payload[key]
        context["actor_role"] = (
            "current_actor" if actor_id == self.current_player_id(state) else "observer"
        )
        return context

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.display_name,
            "description": self.description,
            "tags": list(self.tags),
            "capabilities": sorted(self.capabilities),
            "supported_modes": list(self.supported_modes),
            "agent_count_by_mode": {
                mode: self.agent_count_for_mode(mode) for mode in self.supported_modes
            },
            "cooperation_event_types": list(self.cooperation_event_types),
            "supports_persistent_world": self.supports_persistent_world,
            "social_platforms": self.social_platform_manifest(),
        }
