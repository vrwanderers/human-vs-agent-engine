from __future__ import annotations

from collections import Counter
from typing import Any

from hva_engine.models import ActorKind, GameEvent, MatchMode, Player
from hva_engine.mods.base import GameMod

WEIGHTS = {
    "player_engagement": 0.28,
    "engine_generality": 0.20,
    "dynamism": 0.18,
    "virtual_player_rating": 0.14,
    "ai_opponent_intelligence": 0.20,
}
ENGINE_CAPABILITIES = {
    "turn_based",
    "numeric_state",
    "text_state",
    "spatial",
    "stochastic",
    "audience_input",
}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


class MatchEvaluator:
    """Calculates comparable, explainable 0..1 signals from the canonical event stream."""

    def evaluate(
        self,
        mod: GameMod,
        players: list[Player],
        events: list[GameEvent],
        scores: dict[str, float],
        finished: bool,
        mode: MatchMode,
    ) -> dict[str, Any]:
        action_events = [event for event in events if event.type == "action_applied"]
        humans = {p.id for p in players if p.kind == ActorKind.HUMAN}
        agents = {p.id for p in players if p.kind == ActorKind.AGENT}
        human_actions = [event for event in action_events if event.actor_id in humans]
        agent_actions = [event for event in action_events if event.actor_id in agents]

        if humans:
            human_share = len(human_actions) / max(1, len(action_events) / 2)
            action_types = {event.payload.get("action_type") for event in human_actions}
            diversity = len(action_types) / max(1, min(3, len(human_actions)))
            engagement = _clamp(0.65 * min(1, human_share) + 0.35 * diversity)
        else:
            engagement = 0.5  # neutral in Agent-only calibration matches

        contract_coverage = len(mod.capabilities & ENGINE_CAPABILITIES) / len(ENGINE_CAPABILITIES)
        required_contract = 1.0 if {"turn_based", "numeric_state"} & mod.capabilities else 0.8
        generality = _clamp(0.65 * required_contract + 0.35 * contract_coverage)

        domain_events = [
            event.type for event in events if event.type not in {"match_created", "action_applied"}
        ]
        type_diversity = len(set(domain_events)) / max(1, min(5, len(domain_events)))
        payload_changes = sum(bool(event.payload) for event in events) / max(1, len(events))
        dynamism = _clamp(0.55 * type_diversity + 0.45 * payload_changes)

        agent_score = sum(scores.get(pid, 0.0) for pid in agents) / max(1, len(agents))
        max_score = max([1.0, *scores.values()])
        virtual_rating = _clamp(agent_score / max_score)

        decisions = [event for event in events if event.type == "agent_decision"]
        legal_rate = _clamp(len(agent_actions) / max(1, len(decisions)))
        world_model_rate = _clamp(
            sum(bool(event.payload.get("world_model")) for event in decisions)
            / max(1, len(decisions))
        )
        memory_expected = max(1, len(decisions) - len(agents))
        memory_rate = _clamp(
            sum(bool(event.payload.get("memory_used")) for event in decisions) / memory_expected
        )
        planning_rate = _clamp(
            sum(bool(event.payload.get("predicted_effect")) for event in decisions)
            / max(1, len(decisions))
        )
        agent_types = Counter(event.payload.get("action_type") for event in agent_actions)
        policy_diversity = len(agent_types) / max(1, min(3, len(agent_actions)))
        competitiveness = 0.5
        if finished and humans and agents:
            human_score = sum(scores.get(pid, 0.0) for pid in humans) / len(humans)
            competitiveness = 1 - min(1, abs(agent_score - human_score) / max_score)
        coordination_events = sum(event.type == "coordination_bonus" for event in events)
        cooperation = (
            _clamp(
                0.45 * min(1, coordination_events / 2)
                + 0.55 * (sum(scores.values()) / max(1, len(scores) * max_score))
            )
            if "coop" in mode.value
            else None
        )
        task_performance = cooperation if cooperation is not None else competitiveness
        intelligence = _clamp(
            0.25 * legal_rate
            + 0.15 * world_model_rate
            + 0.15 * memory_rate
            + 0.15 * planning_rate
            + 0.1 * policy_diversity
            + 0.2 * task_performance
        )

        ai_capability_profile = {
            "rules_compliance": legal_rate,
            "world_model_grounding": world_model_rate,
            "memory_utilization": memory_rate,
            "decision_planning": planning_rate,
            "adversarial_competitiveness": _clamp(competitiveness)
            if mode in {MatchMode.HUMAN_VS_AGENT, MatchMode.AGENT_VS_AGENT}
            else None,
            "cooperation_quality": cooperation,
        }

        dimensions = {
            "player_engagement": engagement,
            "engine_generality": generality,
            "dynamism": dynamism,
            "virtual_player_rating": virtual_rating,
            "ai_opponent_intelligence": intelligence,
        }
        composite = _clamp(sum(dimensions[key] * weight for key, weight in WEIGHTS.items()))
        return {
            "version": "mvp-1",
            "composite_score": composite,
            "dimensions": dimensions,
            "ai_capability_profile": ai_capability_profile,
            "applicability": {
                "player_engagement": bool(humans),
                "adversarial_competitiveness": mode
                in {MatchMode.HUMAN_VS_AGENT, MatchMode.AGENT_VS_AGENT},
                "cooperation_quality": "coop" in mode.value,
            },
            "sample": {
                "actions": len(action_events),
                "human_actions": len(human_actions),
                "agent_actions": len(agent_actions),
                "finished": finished,
                "mode": mode.value,
                "agent_decisions": len(decisions),
            },
            "diagnostics": self._diagnostics(dimensions),
        }

    def _diagnostics(self, dimensions: dict[str, float]) -> list[str]:
        labels = {
            "player_engagement": "增加玩家可感知的决策频率与动作差异",
            "engine_generality": "用新 MOD 验证隐藏信息、并行回合或文本状态接口",
            "dynamism": "增加状态反转、环境事件与可追踪的局势变化",
            "virtual_player_rating": "调整虚拟玩家策略，使其保持可信且有角色感",
            "ai_opponent_intelligence": "增加前瞻搜索、对手建模与难度校准",
        }
        return [labels[key] for key, value in dimensions.items() if value < 0.65]
