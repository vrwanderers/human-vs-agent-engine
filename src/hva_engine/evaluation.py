from __future__ import annotations

from collections import Counter
from typing import Any

from hva_engine.models import ActorKind, GameEvent, MatchMode, Player
from hva_engine.mods.base import GameMod

MODE_WEIGHTS = {
    MatchMode.HUMAN_VS_AGENT: {
        "player_engagement": 0.23,
        "engine_generality": 0.13,
        "dynamism": 0.17,
        "virtual_player_rating": 0.12,
        "ai_opponent_intelligence": 0.20,
        "ai_human_likeness": 0.15,
    },
    MatchMode.AGENT_VS_AGENT: {
        "engine_generality": 0.15,
        "dynamism": 0.23,
        "virtual_player_rating": 0.15,
        "ai_opponent_intelligence": 0.32,
        "ai_human_likeness": 0.15,
    },
    MatchMode.AGENT_COOP: {
        "engine_generality": 0.14,
        "dynamism": 0.22,
        "virtual_player_rating": 0.15,
        "ai_opponent_intelligence": 0.34,
        "ai_human_likeness": 0.15,
    },
    MatchMode.HUMAN_AGENT_COOP: {
        "player_engagement": 0.18,
        "engine_generality": 0.13,
        "dynamism": 0.17,
        "virtual_player_rating": 0.12,
        "ai_opponent_intelligence": 0.25,
        "ai_human_likeness": 0.15,
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
    """MVP-3 evaluator: adds human-likeness and narrative/psychology evidence."""

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
        fact_graph_grounding = _clamp(
            sum(
                event.payload.get("context_policy", {}).get("fact_graph_layered") is True
                for event in decisions
            )
            / max(1, len(decisions))
        )
        story_reveals = [event for event in events if event.type == "story_reveal"]
        story_fact_provenance = _clamp(
            sum(bool(event.payload.get("supporting_fact_ids")) for event in story_reveals)
            / max(1, len(story_reveals))
        )
        agent_types = Counter(event.payload.get("action_type") for event in agent_actions)
        policy_diversity = _clamp(len(agent_types) / max(1, min(4, len(agent_actions))))
        human_likeness, human_likeness_profile = self._human_likeness(decisions, events, agents)
        mod_specific_profile: dict[str, Any] | None = None
        if mod.id == "adversarial_interview":
            mod_specific_profile = self._interview_assessment(decisions, events)
            human_likeness = _clamp(
                0.55 * human_likeness + 0.45 * mod_specific_profile["composite"]
            )
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
            0.07 * world_model_rate
            + 0.03 * fact_graph_grounding
            + 0.15 * prediction_accuracy
            + 0.10 * memory_influence
            + 0.10 * planning_rate
            + 0.10 * policy_diversity
            + 0.15 * human_likeness
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
            "ai_human_likeness": human_likeness,
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
            "fact_graph_grounding": fact_graph_grounding,
            "story_fact_provenance": story_fact_provenance,
            "human_likeness": human_likeness,
            "human_likeness_components": human_likeness_profile,
            "interview_assessment": mod_specific_profile,
        }
        return {
            "version": "mvp-3",
            "valid_for_comparison": rules_valid,
            "composite_score": composite,
            "weights": weights,
            "dimensions": dimensions,
            "ai_capability_profile": ai_capability_profile,
            "mod_specific_profile": mod_specific_profile,
            "evidence": {
                "player_engagement": "proxy" if humans else "not_applicable",
                "memory_effectiveness": "influence_proxy; ablation_required",
                "dynamism": "within_match_trajectory",
                "ai_human_likeness": "behavioral_proxy; human panel calibration required",
            },
            "applicability": {
                "player_engagement": bool(humans),
                "adversarial_competitiveness": mode
                in {MatchMode.HUMAN_VS_AGENT, MatchMode.AGENT_VS_AGENT},
                "cooperation_quality": "coop" in mode.value,
                "ai_human_likeness": bool(decisions),
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

    def _human_likeness(
        self, decisions: list[GameEvent], events: list[GameEvent], agents: set[str]
    ) -> tuple[float, dict[str, float]]:
        if not decisions:
            empty = {
                "persona_stability": 0.0,
                "identity_continuity": 0.0,
                "psychological_modeling": 0.0,
                "psychological_dynamics": 0.0,
                "opponent_modeling": 0.0,
                "intention_persistence": 0.0,
                "bounded_rationality": 0.0,
                "narrative_revelation": 0.0,
            }
            return 0.0, empty

        def stable_field(field: str, nested: str) -> float:
            stable = 0
            observed = 0
            for agent_id in agents:
                values = {
                    event.payload.get(field, {}).get(nested)
                    for event in decisions
                    if event.actor_id == agent_id and event.payload.get(field, {}).get(nested)
                }
                if values:
                    observed += 1
                    stable += len(values) == 1
            return stable / max(1, observed)

        persona_stability = stable_field("persona", "archetype")
        identity_continuity = stable_field("identity", "name")
        psychological_modeling = sum(
            bool(event.payload.get("psychological_matrix")) for event in decisions
        ) / len(decisions)
        opponent_modeling = sum(
            bool(event.payload.get("opponent_model")) for event in decisions
        ) / len(decisions)

        psychology_deltas: list[float] = []
        intention_pairs = 0
        intention_same = 0
        for agent_id in agents:
            agent_decisions = [event for event in decisions if event.actor_id == agent_id]
            for previous, current in zip(agent_decisions, agent_decisions[1:], strict=False):
                previous_matrix = previous.payload.get("psychological_matrix", {})
                current_matrix = current.payload.get("psychological_matrix", {})
                shared_keys = set(previous_matrix) & set(current_matrix)
                if shared_keys:
                    psychology_deltas.append(
                        sum(
                            abs(float(current_matrix[key]) - float(previous_matrix[key]))
                            for key in shared_keys
                        )
                        / len(shared_keys)
                    )
                previous_intention = previous.payload.get("intention")
                current_intention = current.payload.get("intention")
                if previous_intention and current_intention:
                    intention_pairs += 1
                    intention_same += previous_intention == current_intention
        psychological_dynamics = _clamp(
            0.45 if not psychology_deltas else sum(psychology_deltas) / len(psychology_deltas) * 6
        )
        persistence_rate = intention_same / max(1, intention_pairs)
        intention_persistence = _clamp(
            0.5 if not intention_pairs else 1 - abs(persistence_rate - 0.65) / 0.65
        )
        nonmax_rate = sum(
            event.payload.get("deliberation_summary", {}).get("chosen_was_utility_max") is False
            for event in decisions
        ) / len(decisions)
        bounded_rationality = _clamp(1 - abs(nonmax_rate - 0.18) / 0.42)
        reveals = [event for event in events if event.type == "story_reveal"]
        narrative_revelation = _clamp(len(reveals) / max(1, len(agents) * 3))

        profile = {
            "persona_stability": _clamp(persona_stability),
            "identity_continuity": _clamp(identity_continuity),
            "psychological_modeling": _clamp(psychological_modeling),
            "psychological_dynamics": psychological_dynamics,
            "opponent_modeling": _clamp(opponent_modeling),
            "intention_persistence": intention_persistence,
            "bounded_rationality": bounded_rationality,
            "narrative_revelation": narrative_revelation,
        }
        score = _clamp(
            0.12 * profile["persona_stability"]
            + 0.12 * profile["identity_continuity"]
            + 0.16 * profile["psychological_modeling"]
            + 0.12 * profile["psychological_dynamics"]
            + 0.10 * profile["opponent_modeling"]
            + 0.10 * profile["intention_persistence"]
            + 0.10 * profile["bounded_rationality"]
            + 0.18 * profile["narrative_revelation"]
        )
        return score, profile

    def _interview_assessment(
        self, decisions: list[GameEvent], events: list[GameEvent]
    ) -> dict[str, Any]:
        questions = [event for event in events if event.type == "interview_question"]
        responses = [event for event in events if event.type == "interview_response"]
        arc_shifts = [event for event in events if event.type == "character_arc_shift"]
        story_reveals = [event for event in events if event.type == "story_reveal"]
        strategies = [str(event.payload.get("strategy")) for event in responses]
        response_diversity = _clamp(len(set(strategies)) / max(1, min(5, len(strategies))))

        matrices = [
            event.payload.get("psychological_matrix", {})
            for event in decisions
            if event.payload.get("psychological_matrix")
        ]
        matrix_keys = ("stress", "frustration", "anger", "fear", "confidence", "morale")
        ranges = (
            [
                max(float(matrix.get(key, 0.0)) for matrix in matrices)
                - min(float(matrix.get(key, 0.0)) for matrix in matrices)
                for key in matrix_keys
            ]
            if matrices
            else []
        )
        average_range = sum(ranges) / max(1, len(ranges))
        # A believable arc changes under pressure without becoming pure random volatility.
        psychological_reactivity = _clamp(1 - abs(average_range - 0.22) / 0.18)
        pressure_signal_grounding = _clamp(
            sum(
                bool(event.payload.get("world_model", {}).get("mod_psychological_signals"))
                for event in decisions
            )
            / max(1, len(decisions))
        )

        identity_strategies = {
            "answer_honestly",
            "admit_uncertainty",
            "reframe",
            "invoke_memory",
        }
        identity_explanation = _clamp(
            0.55
            * sum(strategy in identity_strategies for strategy in strategies)
            / max(1, len(strategies))
            + 0.25
            * sum(event.type == "identity_memory_invoked" for event in events)
            / max(1, len(responses) / 3)
            + 0.20 * min(1.0, len(story_reveals) / 3)
        )
        provenance = _clamp(
            sum(bool(event.payload.get("supporting_fact_ids")) for event in story_reveals)
            / max(1, len(story_reveals))
        )

        final_arc = (
            str(responses[-1].payload.get("arc_stage", "guarded")) if responses else "guarded"
        )
        resolution = {
            "integrated": 1.0,
            "defiant": 0.82,
            "fractured": 0.68,
            "opening_up": 0.72,
            "hardened": 0.58,
            "unresolved": 0.46,
            "guarded": 0.25,
        }.get(final_arc, 0.3)
        character_arc = _clamp(0.55 * min(1.0, len(arc_shifts) / 2) + 0.45 * resolution)
        metrics = responses[-1].payload.get("metrics_after", {}) if responses else {}
        resilience = _clamp(
            0.28 * float(metrics.get("composure", 0.0)) / 100
            + 0.26 * float(metrics.get("authenticity", 0.0)) / 100
            + 0.24 * float(metrics.get("coherence", 0.0)) / 100
            + 0.22 * float(metrics.get("trust", 0.0)) / 100
        )
        question_coverage = _clamp(
            len({event.payload.get("theme") for event in questions})
            / max(1, min(6, len(questions)))
        )
        components = {
            "question_coverage": question_coverage,
            "psychological_reactivity": psychological_reactivity,
            "pressure_signal_grounding": pressure_signal_grounding,
            "identity_explanation": identity_explanation,
            "fact_provenance": provenance,
            "response_strategy_diversity": response_diversity,
            "character_arc": character_arc,
            "resilience": resilience,
        }
        composite = _clamp(
            0.08 * question_coverage
            + 0.16 * psychological_reactivity
            + 0.10 * pressure_signal_grounding
            + 0.16 * identity_explanation
            + 0.12 * provenance
            + 0.12 * response_diversity
            + 0.16 * character_arc
            + 0.10 * resilience
        )
        return {
            "composite": composite,
            "components": components,
            "final_arc": final_arc,
            "questions": len(questions),
            "responses": len(responses),
            "arc_shifts": len(arc_shifts),
            "story_reveals": len(story_reveals),
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
            "ai_human_likeness": "采集真人对角色可信度、情绪连贯性和故事吸引力的盲评",
        }
        diagnostics = [
            labels[key] for key, value in dimensions.items() if value is not None and value < 0.65
        ]
        if not rules_valid:
            diagnostics.insert(0, "规则遵守未通过，当前对局不得进入能力比较")
        return diagnostics
