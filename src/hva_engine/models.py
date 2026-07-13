from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

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
    people: list[str] = Field(default_factory=list, max_length=8)
    themes: list[str] = Field(default_factory=list, max_length=8)
    place: str | None = Field(default=None, max_length=120)
    time_period: str | None = Field(default=None, max_length=120)

    @field_validator("people", "themes")
    @classmethod
    def validate_memory_terms(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value or len(value) > 80 for value in cleaned):
            raise ValueError("Memory people/themes must be non-empty and at most 80 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Memory people/themes must be unique")
        return cleaned


class CharacterSpeechStyleSpec(BaseModel):
    """Surface-language constraints; never an action policy or response script."""

    model_config = ConfigDict(extra="forbid")

    voice_register: Literal[
        "neutral",
        "plain",
        "colloquial",
        "formal",
        "academic",
        "philosophical",
        "technical",
        "aristocratic",
        "confrontational",
    ] = "neutral"
    education_voice: str = Field(default="unspecified", max_length=80)
    vocabulary_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    sentence_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    directness: float = Field(default=0.5, ge=0.0, le=1.0)
    roughness: float = Field(default=0.1, ge=0.0, le=1.0)
    warmth: float = Field(default=0.5, ge=0.0, le=1.0)
    humor: float = Field(default=0.2, ge=0.0, le=1.0)
    philosophical_abstraction: float = Field(default=0.2, ge=0.0, le=1.0)
    technical_jargon: float = Field(default=0.1, ge=0.0, le=1.0)
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0)
    verbal_habits: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("verbal_habits")
    @classmethod
    def validate_verbal_habits(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value or len(value) > 80 for value in cleaned):
            raise ValueError("Verbal habits must be non-empty and at most 80 characters")
        return cleaned


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
    lived_memories: list[CharacterMemorySpec] = Field(default_factory=list, max_length=20)
    speech_style: CharacterSpeechStyleSpec | None = None
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

    @model_validator(mode="after")
    def validate_all_memory_titles(self) -> CharacterCardSpec:
        titles = [
            memory.title
            for memory in (*self.formative_memories, *self.lived_memories)
        ]
        if len(set(titles)) != len(titles):
            raise ValueError("Formative and lived memory titles must be unique")
        return self


class AgentCharacterSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    custom_card: CharacterCardSpec | None = None
    memory_owner_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]+$",
        description="Stable private identity used to restore this character's indexed memory",
    )

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
    world_id: str | None = None
    world_revision: int | None = None
    view_scope: str = "public"
    viewer_id: str | None = None


class CreateMatchRequest(BaseModel):
    mod_id: str
    human_name: str = Field(default="Human", min_length=1, max_length=40)
    human_memory_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]+$",
        description="Stable private identity used by agents to remember this human across matches",
    )
    seed: int | None = None
    mode: MatchMode | None = None
    agent_tuning: AgentTuning = Field(default_factory=AgentTuning)
    agent_characters: list[AgentCharacterSelection] = Field(
        default_factory=list, max_length=8
    )
    agent_memory_owner_ids: list[str] = Field(default_factory=list, max_length=8)
    world_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]+$",
        description="Stable MOD world partition used for durable snapshots",
    )
    resume_world: bool = False

    @field_validator("agent_memory_owner_ids")
    @classmethod
    def validate_agent_memory_owner_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Agent memory owner IDs must be unique")
        for value in values:
            if not 3 <= len(value) <= 128 or not value[0].isalnum():
                raise ValueError("Invalid agent memory owner ID")
            allowed = set(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
            )
            if any(character not in allowed for character in value):
                raise ValueError("Invalid agent memory owner ID")
        return values


class SubmitActionRequest(BaseModel):
    actor_id: str
    action: Action


class PublishStimulusRequest(BaseModel):
    target_agent_id: str = Field(min_length=1, max_length=128)
    modality: Literal[
        "language",
        "vision",
        "audio",
        "touch",
        "interoception",
        "memory",
        "imagination",
        "world_event",
    ]
    semantic_tags: list[str] = Field(default_factory=list, max_length=12)
    source_id: str = Field(default="world", min_length=1, max_length=128)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    reality_status: Literal[
        "observed", "remembered", "imagined", "inferred", "canonical"
    ] | None = None
    privacy: Literal["public", "agent_private"] = "public"
    causal_group: str | None = Field(default=None, max_length=96)

    @field_validator("semantic_tags")
    @classmethod
    def validate_semantic_tags(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split())[:80].lower() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Stimulus semantic tags cannot be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Stimulus semantic tags must be unique")
        return cleaned


class BlindTrialSampleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(min_length=1, max_length=80)
    transcript: list[str] = Field(min_length=1, max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateBlindTrialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(min_length=3, max_length=96, pattern=r"^[A-Za-z0-9_.:-]+$")
    seed: int
    sample_a: BlindTrialSampleSpec
    sample_b: BlindTrialSampleSpec


class BlindRatingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(min_length=3, max_length=96)
    trial_id: str = Field(min_length=8, max_length=96)
    rater_id: str = Field(min_length=3, max_length=128)
    preferred: Literal["A", "B", "tie"]
    a_naturalness: int = Field(ge=1, le=7)
    b_naturalness: int = Field(ge=1, le=7)
    a_identity_consistency: int = Field(ge=1, le=7)
    b_identity_consistency: int = Field(ge=1, le=7)
    a_contextual_fit: int = Field(ge=1, le=7)
    b_contextual_fit: int = Field(ge=1, le=7)
    a_dramatic_interest: int = Field(ge=1, le=7)
    b_dramatic_interest: int = Field(ge=1, le=7)
    notes: str = Field(default="", max_length=1200)

class DanmakuRequest(BaseModel):
    match_id: str
    user: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)
