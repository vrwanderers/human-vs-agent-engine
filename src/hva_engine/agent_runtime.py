from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hva_engine.context import ContextComposer, ContextPacket, SharedFact
from hva_engine.models import Action, GameEvent
from hva_engine.mods.base import GameMod


@dataclass
class MemoryItem:
    turn: int
    situation: str
    action: str
    outcome_events: list[str]
    score_after: float


@dataclass
class AgentBrain:
    """Inspectable baseline brain: perception → world model → memory → constrained decision."""

    player_id: str
    role: str
    memory_limit: int = 12
    memory: deque[MemoryItem] = field(default_factory=lambda: deque(maxlen=12))
    world_model: dict[str, Any] = field(default_factory=dict)
    decisions: int = 0
    last_context: ContextPacket | None = None
    _match_id: str = ""
    _shared_facts: list[SharedFact] = field(default_factory=list)

    def observe(
        self,
        mod: GameMod,
        state: dict[str, Any],
        scores: dict[str, float],
        events: list[GameEvent],
        match_id: str,
        shared_facts: list[SharedFact] | None = None,
    ) -> dict[str, Any]:
        turn = int(state.get("turn", len(events)))
        rivals = {pid: score for pid, score in scores.items() if pid != self.player_id}
        self.world_model = {
            "turn": turn,
            "self_score": round(scores.get(self.player_id, 0.0), 3),
            "other_scores": rivals,
            "terminal": mod.is_terminal(state),
            "recent_events": [event.type for event in events[-4:]],
            "objective": "shared_success" if "coop" in self.role else "outperform_opponents",
            "memory_depth": len(self.memory),
            "shared_fact_count": len(shared_facts or []),
        }
        self._match_id = match_id
        self._shared_facts = list(shared_facts or [])
        return self.world_model

    def decide(
        self,
        mod: GameMod,
        state: dict[str, Any],
        legal: list[Action],
        rng: Random,
    ) -> tuple[Action, dict[str, Any]]:
        if not legal:
            raise ValueError("Agent cannot decide without legal actions")
        self.last_context = ContextComposer().compose(
            match_id=self._match_id,
            agent_id=self.player_id,
            role=self.role,
            mod=mod,
            state=state,
            world_model=self.world_model,
            memory=[item.__dict__ for item in self.memory],
            legal_actions=legal,
            shared_facts=self._shared_facts,
        )
        action = mod.agent_action(state, self.player_id, legal, rng)
        memory_used = bool(self.memory)
        confidence = min(0.92, 0.62 + 0.025 * len(self.memory))
        trace = {
            "policy": f"{mod.id}.baseline",
            "rationale": f"Choose {action.type} from {len(legal)} rule-valid options",
            "confidence": round(confidence, 3),
            "memory_used": memory_used,
            "memory_depth": len(self.memory),
            "world_model": self.world_model,
            "predicted_effect": self._prediction(action),
            "prompt_layers": self.last_context.layers,
            "context_policy": self.last_context.diagnostics,
        }
        self.decisions += 1
        return action, trace

    def remember(
        self,
        turn: int,
        action: Action,
        emitted: list[dict[str, Any]],
        score_after: float,
    ) -> None:
        self.memory.append(
            MemoryItem(
                turn=turn,
                situation=f"turn={turn};objective={self.world_model.get('objective')}",
                action=action.type,
                outcome_events=[item["type"] for item in emitted],
                score_after=round(score_after, 3),
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "decisions": self.decisions,
            "world_model": self.world_model,
            "memory_depth": len(self.memory),
            "recent_memory": [item.__dict__ for item in list(self.memory)[-3:]],
            "context_policy": self.last_context.diagnostics if self.last_context else None,
        }

    def _prediction(self, action: Action) -> str:
        return {
            "attack": "reduce opponent capacity",
            "move": "improve future tactical position",
            "accelerate": "gain position at higher resource risk",
            "conserve": "preserve resources for later turns",
            "pit": "trade current progress for restored resources",
            "evidence": "increase credibility and audience support",
            "emotion": "seek immediate audience swing",
            "rebuttal": "counter the opponent's previous argument",
            "coordinate": "increase team synergy for the next action",
            "research": "reduce uncertainty and systemic threat",
            "stabilize": "directly reduce crisis severity",
        }.get(action.type, "improve objective value within the rules")
