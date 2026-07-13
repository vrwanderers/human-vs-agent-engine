from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hva_engine.cognition import (
    AgentIdentity,
    AutobiographicalMemory,
    CognitiveProfile,
    RuntimeBehaviorPolicy,
)
from hva_engine.models import AgentCharacterSelection, CharacterCardSpec


class CharacterCardError(ValueError):
    pass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class CharacterCardRegistry:
    """Resolves identity seeds; cards never contain situation-to-action mappings."""

    def __init__(self, cards: list[CharacterCardSpec]) -> None:
        self.cards: dict[str, CharacterCardSpec] = {}
        for card in cards:
            if card.id in self.cards:
                raise CharacterCardError(f"Duplicate character card: {card.id}")
            self.cards[card.id] = card

    @classmethod
    def load_default(cls) -> CharacterCardRegistry:
        path = Path(__file__).with_name("data") / "character_cards_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contains_source_text") is not False:
            raise CharacterCardError("Character cards must not redistribute source text")
        cards = [CharacterCardSpec.model_validate(value) for value in payload["cards"]]
        return cls(cards)

    def resolve(self, selection: AgentCharacterSelection) -> tuple[CharacterCardSpec, str]:
        if selection.custom_card is not None:
            return selection.custom_card, "custom"
        try:
            return self.cards[str(selection.card_id)], "builtin"
        except KeyError as exc:
            raise CharacterCardError(f"Unknown character card: {selection.card_id}") from exc

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "id": card.id,
                "name": card.name,
                "source_work": card.source_work,
                "source_url": card.source_url,
                "source_policy": card.source_policy,
                "original_language": card.original_language,
                "cultural_region": card.cultural_region,
                "background": card.background,
                "aspiration": card.aspiration,
                "values": card.values,
                "social_style": card.social_style,
                "decision_model": "runtime_cognition_not_scripted_actions",
            }
            for card in self.cards.values()
        ]

    def instantiate(
        self,
        card: CharacterCardSpec,
        policy: RuntimeBehaviorPolicy,
        source_kind: str,
    ) -> tuple[CognitiveProfile, AgentIdentity]:
        traits = card.traits
        shadow = policy.effective_shadow_intensity
        profile = CognitiveProfile(
            archetype=f"character_card:{card.id}",
            risk_tolerance=traits.risk_tolerance,
            loss_aversion=traits.loss_aversion,
            patience=traits.patience,
            curiosity=traits.curiosity,
            empathy=round(_clamp(traits.empathy * (1 - 0.45 * shadow)), 3),
            adaptability=traits.adaptability,
            machiavellianism=round(
                _clamp(traits.machiavellianism + 0.45 * shadow), 3
            ),
            decision_noise=round(
                _clamp(traits.decision_noise + 0.05 * policy.realism), 3
            ),
            openness=traits.openness,
            conscientiousness=traits.conscientiousness,
            extraversion=traits.extraversion,
            agreeableness=round(
                _clamp(traits.agreeableness * (1 - 0.25 * shadow)), 3
            ),
            neuroticism=traits.neuroticism,
            coping_style=traits.coping_style,
            display_rule=traits.display_rule,
        )
        identity = AgentIdentity(
            name=card.name,
            background=card.background,
            aspiration=card.aspiration,
            core_wound=card.core_wound,
            values=tuple(card.values),
            social_style=card.social_style,
            formative_memories=tuple(
                AutobiographicalMemory(
                    title=memory.title,
                    recollection=memory.recollection,
                    emotional_valence=memory.emotional_valence,
                    lesson=memory.lesson,
                )
                for memory in card.formative_memories
            ),
            motive_weights=dict(card.motive_weights),
            commitment_weights=dict(card.commitment_weights),
            character_card_id=(
                card.id if source_kind == "builtin" else f"custom:{card.id}"
            ),
        )
        return profile, identity
