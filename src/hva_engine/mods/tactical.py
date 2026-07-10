from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.models import Action, Player
from hva_engine.mods.base import GameMod


class TacticalDuel(GameMod):
    id = "tactical_duel"
    display_name = "战术对决"
    description = "在 5×5 战场上移动、蓄力并攻击对手。"
    tags = ("tactics", "combat", "spatial")
    capabilities = frozenset({"turn_based", "numeric_state", "spatial", "audience_input"})

    _directions = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        order = [p.id for p in players]
        rng.shuffle(order)
        return {
            "turn": 0,
            "max_turns": 30,
            "order": order,
            "initiative": order[0],
            "units": {
                players[0].id: {"x": 0, "y": 2, "hp": 10, "energy": 2},
                players[1].id: {"x": 4, "y": 2, "hp": 10, "energy": 2},
            },
            "winner": None,
        }

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        if self.is_terminal(state):
            return None
        return state["order"][state["turn"] % len(state["order"])]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        unit = state["units"][actor_id]
        occupied = {(u["x"], u["y"]) for pid, u in state["units"].items() if pid != actor_id}
        actions: list[Action] = []
        for name, (dx, dy) in self._directions.items():
            target = (unit["x"] + dx, unit["y"] + dy)
            if 0 <= target[0] < 5 and 0 <= target[1] < 5 and target not in occupied:
                actions.append(Action(type="move", payload={"direction": name}))
        enemies = [
            pid for pid in state["order"] if pid != actor_id and state["units"][pid]["hp"] > 0
        ]
        for enemy_id in enemies:
            enemy = state["units"][enemy_id]
            distance = abs(unit["x"] - enemy["x"]) + abs(unit["y"] - enemy["y"])
            if distance <= 2 and unit["energy"] >= 1:
                actions.append(Action(type="attack", payload={"target_id": enemy_id}))
        actions.append(Action(type="charge"))
        return actions

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        unit = new["units"][actor_id]
        emitted: list[dict[str, Any]] = []
        if action.type == "move":
            dx, dy = self._directions[action.payload["direction"]]
            unit["x"], unit["y"] = unit["x"] + dx, unit["y"] + dy
            unit["energy"] = min(4, unit["energy"] + 1)
            emitted.append({"type": "unit_moved", "position": [unit["x"], unit["y"]]})
        elif action.type == "attack":
            target_id = action.payload["target_id"]
            damage = 1 + int(unit["energy"] >= 3) + rng.randint(0, 1)
            new["units"][target_id]["hp"] = max(0, new["units"][target_id]["hp"] - damage)
            unit["energy"] -= 1
            emitted.append({"type": "unit_attacked", "target_id": target_id, "damage": damage})
            if new["units"][target_id]["hp"] == 0:
                new["winner"] = actor_id
        elif action.type == "charge":
            unit["energy"] = min(5, unit["energy"] + 2)
            emitted.append({"type": "unit_charged", "energy": unit["energy"]})
        new["turn"] += 1
        if new["turn"] >= new["max_turns"] and not new["winner"]:
            best_hp = max(new["units"][pid]["hp"] for pid in new["order"])
            leaders = [pid for pid in new["order"] if new["units"][pid]["hp"] == best_hp]
            new["winner"] = leaders[0] if len(leaders) == 1 else None
        return new, emitted

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return state["winner"] is not None or state["turn"] >= state["max_turns"]

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        return {
            pid: round(unit["hp"] / 10 + (1.0 if state["winner"] == pid else 0.0), 3)
            for pid, unit in state["units"].items()
        }

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        attacks = [action for action in legal if action.type == "attack"]
        if attacks:
            return attacks[0]
        me = state["units"][actor_id]
        enemy_id = next(pid for pid in state["order"] if pid != actor_id)
        enemy = state["units"][enemy_id]
        moves = [action for action in legal if action.type == "move"]
        if moves:

            def distance_after(action: Action) -> int:
                dx, dy = self._directions[action.payload["direction"]]
                return abs(me["x"] + dx - enemy["x"]) + abs(me["y"] + dy - enemy["y"])

            return min(moves, key=distance_after)
        return legal[0]
