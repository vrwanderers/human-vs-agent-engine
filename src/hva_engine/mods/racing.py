from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.models import Action, Player
from hva_engine.mods.base import GameMod


class RacingStrategy(GameMod):
    id = "racing_strategy"
    display_name = "赛车策略"
    description = "管理速度、燃料和轮胎，在天气变化中率先完成赛程。"
    tags = ("racing", "resource", "risk")
    capabilities = frozenset({"turn_based", "numeric_state", "stochastic", "audience_input"})

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        order = [p.id for p in players]
        rng.shuffle(order)
        return {
            "turn": 0,
            "max_turns": 24,
            "track_length": 36,
            "order": order,
            "initiative": order[0],
            "weather": "dry",
            "finished": False,
            "cars": {p.id: {"position": 0, "speed": 1, "fuel": 18, "tyres": 100} for p in players},
            "winner": None,
        }

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        return None if self.is_terminal(state) else state["order"][state["turn"] % 2]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        car = state["cars"][actor_id]
        actions = [Action(type="conserve")]
        if car["fuel"] >= 2 and car["tyres"] >= 8:
            actions.append(Action(type="accelerate"))
        if car["fuel"] <= 7 or car["tyres"] <= 35:
            actions.append(Action(type="pit"))
        return actions

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        car = new["cars"][actor_id]
        if action.type == "accelerate":
            car["speed"] = min(5, car["speed"] + 1)
            car["fuel"] -= 2
            car["tyres"] -= 9 if new["weather"] == "dry" else 13
        elif action.type == "conserve":
            car["speed"] = max(1, car["speed"] - 1)
            car["fuel"] = max(0, car["fuel"] - 1)
            car["tyres"] -= 3
        elif action.type == "pit":
            car.update({"speed": 1, "fuel": 18, "tyres": 100})
        progress = 0 if action.type == "pit" else car["speed"]
        if new["weather"] == "rain" and car["speed"] >= 4 and rng.random() < 0.3:
            progress = max(0, progress - 3)
        car["position"] += progress
        new["turn"] += 1
        emitted = [{"type": "lap_progress", "distance": progress, "strategy": action.type}]
        if new["turn"] % 6 == 0:
            new["weather"] = "rain" if rng.random() < 0.4 else "dry"
            emitted.append({"type": "weather_changed", "weather": new["weather"]})
        round_finished = new["turn"] % len(new["order"]) == 0
        crossed = any(new["cars"][pid]["position"] >= new["track_length"] for pid in new["order"])
        if round_finished and (crossed or new["turn"] >= new["max_turns"]):
            best = max(new["cars"][pid]["position"] for pid in new["order"])
            leaders = [pid for pid in new["order"] if new["cars"][pid]["position"] == best]
            new["winner"] = leaders[0] if len(leaders) == 1 else None
            new["finished"] = True
        return new, emitted

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return state["finished"]

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        return {
            pid: round(car["position"] / state["track_length"] + (state["winner"] == pid), 3)
            for pid, car in state["cars"].items()
        }

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        car = state["cars"][actor_id]
        by_type = {action.type: action for action in legal}
        if "pit" in by_type and (car["fuel"] <= 4 or car["tyres"] <= 20):
            return by_type["pit"]
        if "accelerate" in by_type and state["weather"] == "dry" and car["fuel"] > 7:
            return by_type["accelerate"]
        return by_type["conserve"]
