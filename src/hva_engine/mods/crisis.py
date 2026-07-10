from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.models import Action, Player
from hva_engine.mods.base import GameMod


class CrisisCoop(GameMod):
    id = "crisis_coop"
    display_name = "危机联合作战"
    description = "两名 Agent 共享情报与资源，在连锁危机失控前协同稳定局势。"
    tags = ("cooperation", "crisis", "shared-objective")
    capabilities = frozenset({"turn_based", "numeric_state", "stochastic", "shared_objective"})
    supported_modes = ("agent_coop", "human_agent_coop")

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        order = [p.id for p in players]
        rng.shuffle(order)
        return {
            "turn": 0,
            "max_turns": 16,
            "order": order,
            "initiative": order[0],
            "threat": round(rng.uniform(76, 90), 2),
            "supplies": round(rng.uniform(44, 54), 2),
            "intel": 10.0,
            "trust": 45.0,
            "synergy": 0.0,
            "last_actions": {p.id: None for p in players},
            "success": None,
        }

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        return None if self.is_terminal(state) else state["order"][state["turn"] % 2]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        actions = [Action(type="coordinate"), Action(type="conserve")]
        if state["supplies"] >= 8:
            actions.append(Action(type="stabilize"))
        if state["supplies"] >= 5:
            actions.append(Action(type="research"))
        return actions

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        emitted: list[dict[str, Any]] = []
        bonus = new["synergy"]
        if action.type == "coordinate":
            new["trust"] = min(100, new["trust"] + 10)
            new["synergy"] = 4 + new["trust"] / 25
            emitted.append({"type": "coordination_bonus", "synergy": round(new["synergy"], 2)})
        elif action.type == "stabilize":
            reduction = 10 + bonus
            new["threat"] = max(0, new["threat"] - reduction)
            new["supplies"] -= 8
            new["synergy"] = 0
            emitted.append({"type": "threat_reduced", "amount": round(reduction, 2)})
        elif action.type == "research":
            new["intel"] = min(100, new["intel"] + 14 + bonus)
            new["threat"] = max(0, new["threat"] - 4 - bonus / 2)
            new["supplies"] -= 5
            new["synergy"] = 0
            emitted.append({"type": "intel_gained", "intel": round(new["intel"], 2)})
        elif action.type == "conserve":
            new["supplies"] += 4
            new["threat"] += 3
            emitted.append({"type": "resources_recovered", "supplies": new["supplies"]})
        new["last_actions"][actor_id] = action.type
        new["turn"] += 1
        if new["turn"] % 3 == 0 and rng.random() < 0.55:
            surge = max(3, 10 - new["intel"] / 18)
            new["threat"] += surge
            emitted.append({"type": "crisis_surge", "amount": round(surge, 2)})
        if new["threat"] <= 0:
            new["success"] = True
        elif new["turn"] >= new["max_turns"] or new["threat"] >= 110:
            new["success"] = new["threat"] < 30
        return new, emitted

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return state["success"] is not None

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        team_score = max(0.0, 1.5 - state["threat"] / 86)
        if state["success"] is True:
            team_score += 0.5
        return {pid: round(team_score, 3) for pid in state["order"]}

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        by_type = {action.type: action for action in legal}
        partner = next(pid for pid in state["order"] if pid != actor_id)
        if state["synergy"] > 0 and "stabilize" in by_type:
            return by_type["stabilize"]
        if state["last_actions"][partner] != "coordinate" and state["trust"] < 70:
            return by_type["coordinate"]
        if state["intel"] < 38 and "research" in by_type:
            return by_type["research"]
        if "stabilize" in by_type:
            return by_type["stabilize"]
        return by_type["conserve"]
