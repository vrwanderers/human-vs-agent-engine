from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from hva_engine.models import GameEvent
from hva_engine.mods.base import GameMod


@dataclass(frozen=True)
class AgentObservation:
    """A viewer-scoped, detached snapshot passed into one Agent decision."""

    viewer_id: str
    state: dict[str, Any]
    scores: dict[str, float]
    events: list[GameEvent]
    shared_facts: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class ObservationPolicy:
    """Enforces the boundary between authoritative state and Agent observations."""

    def for_agent(
        self,
        *,
        mod: GameMod,
        state: dict[str, Any],
        scores: dict[str, float],
        events: list[GameEvent],
        shared_facts: list[dict[str, Any]],
        agent_id: str,
    ) -> AgentObservation:
        # Both the input and output are detached. A MOD cannot accidentally hand the
        # authoritative dictionary to an Agent or mutate it while constructing a view.
        internal_copy = deepcopy(state)
        visible_state = mod.public_state(internal_copy, agent_id)
        if not isinstance(visible_state, dict):
            raise TypeError("MOD public_state must return a dictionary")
        detached_state = deepcopy(visible_state)
        return AgentObservation(
            viewer_id=agent_id,
            state=detached_state,
            scores=deepcopy(scores),
            events=deepcopy(events),
            shared_facts=deepcopy(shared_facts),
            diagnostics={
                "viewer_scoped": True,
                "viewer_id": agent_id,
                "detached_snapshot": True,
                "visible_state_keys": sorted(detached_state),
            },
        )
