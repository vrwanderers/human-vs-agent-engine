from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActorKind(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    AUDIENCE = "audience"


class MatchStatus(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"


class MatchMode(StrEnum):
    HUMAN_VS_AGENT = "human_vs_agent"
    AGENT_VS_AGENT = "agent_vs_agent"
    AGENT_COOP = "agent_coop"
    HUMAN_AGENT_COOP = "human_agent_coop"


class Player(BaseModel):
    id: str
    name: str
    kind: ActorKind


class Action(BaseModel):
    type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class GameEvent(BaseModel):
    seq: int
    type: str
    actor_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MatchView(BaseModel):
    id: str
    mod_id: str
    mode: MatchMode
    status: MatchStatus
    players: list[Player]
    human_player_id: str | None
    current_player_id: str | None
    state: dict[str, Any]
    legal_actions: list[Action]
    scores: dict[str, float]
    events: list[GameEvent]
    agent_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CreateMatchRequest(BaseModel):
    mod_id: str
    human_name: str = Field(default="Human", min_length=1, max_length=40)
    seed: int | None = None
    mode: MatchMode | None = None


class SubmitActionRequest(BaseModel):
    actor_id: str
    action: Action


class DanmakuRequest(BaseModel):
    match_id: str
    user: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)
