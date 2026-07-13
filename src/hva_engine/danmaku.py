from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hva_engine.engine import EngineError, GameEngine
from hva_engine.models import MatchView


class DanmakuAdapter(Protocol):
    """Platform adapters normalize signed live events into a user/message pair."""

    async def connect(self) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class ParsedCommand:
    action_type: str
    argument: str | None = None


ALIASES = {
    "move": "move",
    "移动": "move",
    "attack": "attack",
    "攻击": "attack",
    "charge": "charge",
    "蓄力": "charge",
    "accelerate": "accelerate",
    "加速": "accelerate",
    "conserve": "conserve",
    "保胎": "conserve",
    "pit": "pit",
    "进站": "pit",
    "evidence": "evidence",
    "证据": "evidence",
    "emotion": "emotion",
    "情感": "emotion",
    "rebuttal": "rebuttal",
    "反驳": "rebuttal",
    "ask": "ask",
    "采访": "ask",
    "提问": "ask",
}


def parse_command(message: str) -> ParsedCommand | None:
    text = message.strip()
    if not text.startswith("!"):
        return None
    parts = text[1:].split(maxsplit=1)
    action_type = ALIASES.get(parts[0].lower())
    if not action_type:
        return None
    return ParsedCommand(action_type, parts[1].strip() if len(parts) == 2 else None)


def dispatch_danmaku(
    engine: GameEngine, match_id: str, message: str, user: str = "anonymous"
) -> MatchView:
    match = engine.get(match_id)
    command = parse_command(message)
    if command is None:
        raise EngineError("Danmaku is not a recognized command")
    legal = match.mod.legal_actions(match.state, match.human_player_id)
    candidates = [
        action
        for action in legal
        if action.type == command.action_type
        or (command.action_type == "ask" and action.type.startswith("ask_"))
    ]
    if command.argument:
        candidates = [
            action
            for action in candidates
            if command.argument.lower() in {str(value).lower() for value in action.payload.values()}
        ]
    if not candidates:
        raise EngineError("Command does not map to a legal action in the current state")
    match.add_event("danmaku_received", user=user, user_command=message)
    return engine.submit(match_id, match.human_player_id, candidates[0])
