from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class ContentMode(StrEnum):
    STANDARD = "standard"
    MATURE_FICTION = "mature_fiction"


class EventVisibility(StrEnum):
    PUBLIC = "public"
    ENGINE_PRIVATE = "engine_private"


class AgentTuning(BaseModel):
    """Per-match cognitive style; engine rules remain authoritative in every mode."""

    realism: float = Field(default=0.7, ge=0.0, le=1.0)
    shadow_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    content_mode: ContentMode = ContentMode.STANDARD


class CharacterTraitProfile(BaseModel):
    """Stable dispositions; these are inputs to cognition, never direct action rules."""

    model_config = ConfigDict(extra="forbid")

    risk_tolerance: float = Field(ge=0.0, le=1.0)
    loss_aversion: float = Field(ge=0.0, le=1.0)
    patience: float = Field(ge=0.0, le=1.0)
    curiosity: float = Field(ge=0.0, le=1.0)
    empathy: float = Field(ge=0.0, le=1.0)
    adaptability: float = Field(ge=0.0, le=1.0)
    machiavellianism: float = Field(ge=0.0, le=1.0)
    decision_noise: float = Field(default=0.14, ge=0.0, le=0.4)
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    coping_style: str = Field(min_length=1, max_length=80)
    display_rule: str = Field(min_length=1, max_length=160)


class CharacterMemorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=100)
    recollection: str = Field(min_length=1, max_length=600)
    emotional_valence: float = Field(ge=-1.0, le=1.0)
    lesson: str = Field(min_length=1, max_length=300)


class CharacterCardSpec(BaseModel):
    """Declarative identity seed. It deliberately contains no action or response table."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    name: str = Field(min_length=1, max_length=80)
    background: str = Field(min_length=1, max_length=800)
    aspiration: str = Field(min_length=1, max_length=400)
    core_wound: str = Field(min_length=1, max_length=400)
    values: list[str] = Field(min_length=2, max_length=8)
    social_style: str = Field(min_length=1, max_length=300)
    formative_memories: list[CharacterMemorySpec] = Field(min_length=3, max_length=3)
    traits: CharacterTraitProfile
    motive_weights: dict[str, float] = Field(min_length=3, max_length=12)
    commitment_weights: dict[str, float] = Field(min_length=1, max_length=12)
    source_work: str | None = Field(default=None, max_length=160)
    source_url: str | None = Field(default=None, max_length=500)
    source_policy: str = Field(default="user_supplied_original", max_length=80)
    original_language: str = Field(default="und", pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$")
    cultural_region: str = Field(default="unspecified", min_length=1, max_length=80)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value or len(value) > 100 for value in cleaned):
            raise ValueError("Character values must be non-empty and at most 100 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Character values must be unique")
        return cleaned

    @field_validator("motive_weights", "commitment_weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        for name, weight in weights.items():
            if not name or len(name) > 80 or not 0.0 <= weight <= 1.0:
                raise ValueError("Character weights need short names and values in [0, 1]")
        return weights

    @field_validator("motive_weights")
    @classmethod
    def validate_motive_vocabulary(cls, weights: dict[str, float]) -> dict[str, float]:
        allowed = {
            "self_preservation",
            "truth",
            "belonging",
            "autonomy",
            "status",
            "duty",
            "redemption",
            "care",
        }
        unsupported = sorted(set(weights) - allowed)
        if unsupported:
            raise ValueError(f"Unsupported abstract motives: {unsupported}")
        return weights

    @field_validator("formative_memories")
    @classmethod
    def validate_memory_titles(
        cls, memories: list[CharacterMemorySpec]
    ) -> list[CharacterMemorySpec]:
        if len({memory.title for memory in memories}) != len(memories):
            raise ValueError("Formative memory titles must be unique")
        return memories


class AgentCharacterSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    custom_card: CharacterCardSpec | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> AgentCharacterSelection:
        if (self.card_id is None) == (self.custom_card is None):
            raise ValueError("Choose exactly one of card_id or custom_card")
        return self


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
    visibility: EventVisibility = EventVisibility.PUBLIC
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
    agent_tuning: AgentTuning = Field(default_factory=AgentTuning)
    agent_characters: list[AgentCharacterSelection] = Field(
        default_factory=list, max_length=2
    )


class SubmitActionRequest(BaseModel):
    actor_id: str
    action: Action


class DanmakuRequest(BaseModel):
    match_id: str
    user: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)
