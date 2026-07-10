from __future__ import annotations

from collections import Counter
from typing import Any

from hva_engine.models import ActorKind, GameEvent, MatchMode, Player
from hva_engine.mods.base import GameMod

MODE_WEIGHTS = {
    MatchMode.HUMAN_VS_AGENT: {
        "player_engagement": 0.25,
        "engine_generality": 0.15,
        "dynamism": 0.20,
        "virtual_player_rating": 0.15,
        "ai_opponent_intelligence": 0.25,
    },
    MatchMode.AGENT_VS_AGENT: {
        "engine_generality": 0.18,
        "dynamism": 0.28,
        "virtual_player_rating": 0.18,
        "ai_opponent_intelligence": 0.36,
    },
    MatchMode.AGENT_COOP: {
        "engine_generality": 0.16,
        "dynamism": 0.26,
        "virtual_player_rating": 0.18,
        "ai_opponent_intelligence": 0.40,
    },
    MatchMode.HUMAN_AGENT_COOP: {
        "player_engagement": 0.20,
        "engine_generality": 0.15,
        "dynamism": 0.20,
        "virtual_player_rating": 0.15,
        "ai_opponent_intelligence": 0.30,
    },
}
ENGINE_CAPABILITIES = {
    "turn_based",
    "numeric_state",
    "text_state",
    "spatial",
    "stochastic",
    "audience_input",
    "shared_objective",
}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


class MatchEvaluator:
    """MVP-2 evaluator: applicability-aware, rule-gated, and outcome-sensitive."""

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
        decisions = [event for event in events if event.type == "agent_decision"]
        humans = {p.id for p in players if p.kind == ActorKind.HUMAN}
        agents = {p.id for p in players if p.kind == ActorKind.AGENT}
        human_actions = [event for event in action_events if event.actor_id in humans]
        agent_actions = [event for event in action_events if event.actor_id in agents]

        engagement: float | None = None
        if humans:
            expected_human_share = len(humans) / len(players)
            actual_share = len(human_actions) / max(1, len(action_events))
            participation = min(1.0, actual_share / expected_human_share)
            action_types = {event.payload.get("action_type") for event in human_actions}
            diversity = len(action_types) / max(1, min(4, len(human_actions)))
            # Until latency/retention telemetry exists, engagement is a discounted proxy.
            engagement = _clamp(0.7 * (0.65 * participation + 0.35 * diversity))

        coverage = len(mod.capabilities & ENGINE_CAPABILITIES) / len(ENGINE_CAPABILITIES)
        contract = 1.0 if "turn_based" in mod.capabilities else 0.7
        generality = _clamp(0.7 * contract + 0.3 * coverage)

        domain_events = [
            event.type
            for event in events
            if event.type not in {"match_created", "action_applied", "agent_decision"}
        ]
        event_diversity = len(set(domain_events)) / max(1, min(6, len(domain_events)))
        action_diversity = len({event.payload.get("action_type") for event in action_events}) / max(
            1, min(5, len(action_events))
        )
        signals = self._score_signals(action_events, players, "coop" in mode.value)
        signal_changes = sum(
            abs(current - previous) > 0.03
            for previous, current in zip(signals, signals[1:], strict=False)
        ) / max(1, len(signals) - 1)
        directions = [
            1 if current > previous else -1
            for previous, current in zip(signals, signals[1:], strict=False)
            if abs(current - previous) > 0.03
        ]
        reversals = sum(
            current != previous
            for previous, current in zip(directions, directions[1:], strict=False)
        )
        reversal_rate = min(1.0, reversals / 2)
        dynamism = _clamp(
            0.2 * event_diversity
            + 0.2 * action_diversity
            + 0.35 * signal_changes
            + 0.25 * reversal_rate
        )

        max_score = max([1.0, *scores.values()])
        competitiveness = 0.5
        if finished and mode in {MatchMode.HUMAN_VS_AGENT, MatchMode.AGENT_VS_AGENT}:
            values = [scores.get(player.id, 0.0) for player in players]
            competitiveness = 1 - min(1.0, abs(values[0] - values[1]) / max_score)
        team_performance = _clamp(sum(scores.values()) / max(1, len(scores) * 1.5))
        virtual_rating = (
            _clamp(team_performance) if "coop" in mode.value else _clamp(competitiveness)
        )

        legal_rate = _clamp(len(agent_actions) / len(decisions)) if decisions else 1.0
        rules_valid = not decisions or (legal_rate == 1.0 and len(agent_actions) == len(decisions))
        world_model_rate = _clamp(
            sum(bool(event.payload.get("world_model")) for event in decisions)
            / max(1, len(decisions))
        )
        prediction_accuracy = _clamp(
            sum(event.payload.get("prediction_verified") is True for event in decisions)
            / max(1, len(decisions))
        )
        memory_expected = max(1, len(decisions) - 3 * max(1, len(agents)))
        memory_rate = _clamp(
            sum(bool(event.payload.get("memory_used")) for event in decisions) / memory_expected
        )
        memory_influence = _clamp(
            sum(bool(event.payload.get("memory_influenced")) for event in decisions)
            / memory_expected
        )
        planning_rate = _clamp(
            sum(bool(event.payload.get("prediction")) for event in decisions)
            / max(1, len(decisions))
        )
        agent_types = Counter(event.payload.get("action_type") for event in agent_actions)
        policy_diversity = _clamp(len(agent_types) / max(1, min(4, len(agent_actions))))
        coordination_events = sum(event.type == "coordination_bonus" for event in events)
        cooperation = (
            _clamp(
                0.45 * min(1.0, coordination_events / max(1, len(action_events) / 5))
                + 0.55 * team_performance
            )
            if "coop" in mode.value
            else None
        )
        task_performance = cooperation if cooperation is not None else competitiveness
        intelligence = _clamp(
            0.15 * world_model_rate
            + 0.20 * prediction_accuracy
            + 0.10 * memory_influence
            + 0.10 * planning_rate
            + 0.15 * policy_diversity
            + 0.30 * task_performance
        )
        if not rules_valid:
            intelligence = 0.0

        dimensions: dict[str, float | None] = {
            "player_engagement": engagement,
            "engine_generality": generality,
            "dynamism": dynamism,
            "virtual_player_rating": virtual_rating,
            "ai_opponent_intelligence": intelligence,
        }
        weights = MODE_WEIGHTS[mode]
        composite = _clamp(sum(float(dimensions[key]) * weight for key, weight in weights.items()))
        if not rules_valid:
            composite = 0.0

        ai_capability_profile = {
            "rules_compliance": legal_rate,
            "world_model_grounding": world_model_rate,
            "prediction_accuracy": prediction_accuracy,
            "memory_utilization": memory_rate,
            "memory_influence_rate": memory_influence,
            "decision_planning": planning_rate,
            "policy_diversity": policy_diversity,
            "adversarial_competitiveness": _clamp(competitiveness)
            if mode in {MatchMode.HUMAN_VS_AGENT, MatchMode.AGENT_VS_AGENT}
            else None,
            "cooperation_quality": cooperation,
        }
        return {
            "version": "mvp-2",
            "valid_for_comparison": rules_valid,
            "composite_score": composite,
            "weights": weights,
            "dimensions": dimensions,
            "ai_capability_profile": ai_capability_profile,
            "evidence": {
                "player_engagement": "proxy" if humans else "not_applicable",
                "memory_effectiveness": "influence_proxy; ablation_required",
                "dynamism": "within_match_trajectory",
            },
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
            "diagnostics": self._diagnostics(dimensions, rules_valid),
        }

    def _score_signals(
        self, action_events: list[GameEvent], players: list[Player], cooperative: bool
    ) -> list[float]:
        signals: list[float] = []
        for event in action_events:
            scores = event.payload.get("scores_after", {})
            if cooperative:
                signals.append(sum(scores.values()) / max(1, len(scores)))
            elif len(players) >= 2:
                signals.append(scores.get(players[0].id, 0.0) - scores.get(players[1].id, 0.0))
        return signals

    def _diagnostics(self, dimensions: dict[str, float | None], rules_valid: bool) -> list[str]:
        labels = {
            "player_engagement": "采集真实玩家留存、响应时间和主观娱乐性",
            "engine_generality": "用隐藏信息、并行回合和多人阵营验证契约",
            "dynamism": "增加领先反转、可控随机性和差异化状态轨迹",
            "virtual_player_rating": "将默认难度校准到公平且有挑战性的区间",
            "ai_opponent_intelligence": "验证预测、记忆增益和策略多样性",
        }
        diagnostics = [
            labels[key] for key, value in dimensions.items() if value is not None and value < 0.65
        ]
        if not rules_valid:
            diagnostics.insert(0, "规则遵守未通过，当前对局不得进入能力比较")
        return diagnostics
