from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.models import Action, Player
from hva_engine.mods.base import GameMod


class DebateArena(GameMod):
    id = "debate_arena"
    display_name = "辩论擂台"
    description = "在证据、情感和反驳之间博弈，争夺虚拟观众支持。"
    tags = ("debate", "social", "text")
    capabilities = frozenset({"turn_based", "text_state", "stochastic", "audience_input"})
    _actions = ("evidence", "emotion", "rebuttal")

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        order = [p.id for p in players]
        rng.shuffle(order)
        return {
            "turn": 0,
            "max_turns": 10,
            "order": order,
            "initiative": order[0],
            "credibility": {p.id: 5.0 for p in players},
            "support": {
                order[0]: 55.0,
                order[1]: 45.0,
            },
            "last_move": {p.id: None for p in players},
            "winner": None,
            "topic": "AI 是否应当参与公共决策？",
        }

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        return None if self.is_terminal(state) else state["order"][state["turn"] % 2]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        return [
            Action(type=kind, payload={"claim": self._sample_claim(kind)}) for kind in self._actions
        ]

    def _sample_claim(self, kind: str) -> str:
        return {
            "evidence": "引用可验证的数据支持论点",
            "emotion": "用具体故事争取观众共鸣",
            "rebuttal": "指出对方上一轮论证的漏洞",
        }[kind]

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        opponent = next(pid for pid in new["order"] if pid != actor_id)
        previous = new["last_move"][opponent]
        base = {"evidence": 6.0, "emotion": 5.0, "rebuttal": 3.0}[action.type]
        counter_bonus = (
            4.0
            if (
                (action.type, previous)
                in {("rebuttal", "evidence"), ("evidence", "emotion"), ("emotion", "rebuttal")}
            )
            else 0.0
        )
        swing = base + counter_bonus + rng.uniform(-1.5, 1.5)
        if action.type == "emotion":
            new["credibility"][actor_id] = max(0, new["credibility"][actor_id] - 0.3)
        elif action.type == "evidence":
            new["credibility"][actor_id] = min(10, new["credibility"][actor_id] + 0.5)
        swing *= 0.8 + new["credibility"][actor_id] / 25
        new["support"][actor_id] = min(100, new["support"][actor_id] + swing / 2)
        new["support"][opponent] = max(0, new["support"][opponent] - swing / 2)
        new["last_move"][actor_id] = action.type
        new["turn"] += 1
        if new["turn"] >= new["max_turns"]:
            best = max(new["support"].values())
            leaders = [pid for pid in new["order"] if new["support"][pid] == best]
            new["winner"] = leaders[0] if len(leaders) == 1 else "draw"
        return new, [{"type": "audience_shift", "swing": round(swing, 2), "move": action.type}]

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return state["winner"] is not None

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        return {pid: round(score / 50, 3) for pid, score in state["support"].items()}

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        opponent = next(pid for pid in state["order"] if pid != actor_id)
        counter = {"evidence": "rebuttal", "emotion": "evidence", "rebuttal": "emotion"}
        desired = counter.get(state["last_move"][opponent], "evidence")
        return next(action for action in legal if action.type == desired)
