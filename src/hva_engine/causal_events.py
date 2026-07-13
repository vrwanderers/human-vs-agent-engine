from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import Any


@dataclass(frozen=True)
class CausalRule:
    id: str
    event_kind: str
    conditions: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    once: bool = True
    cooldown_minutes: int = 0
    retry_minutes: int = 60
    max_occurrences: int | None = None


class CausalRuleEngine:
    """Serializable rule-state evaluator; MODs provide event templates, not fixed schedules."""

    @staticmethod
    def initial_state(facts: dict[str, float]) -> dict[str, Any]:
        return {
            "facts": {key: round(float(value), 4) for key, value in facts.items()},
            "rule_history": [],
            "event_causes": [],
            "next_attempt_at": {},
            "occurrences": {},
            "revision": 0,
        }

    def evaluate(
        self,
        *,
        rules: tuple[CausalRule, ...],
        causal_state: dict[str, Any],
        world: dict[str, Any],
        day: int,
        minute: int,
        rng: Random,
        event_factory: Callable[[CausalRule, list[str], int], dict[str, Any]],
        max_events: int = 4,
    ) -> list[dict[str, Any]]:
        absolute_minute = (day - 1) * 24 * 60 + minute
        emitted: list[dict[str, Any]] = []
        for _pass in range(max_events):
            selected: tuple[CausalRule, list[str]] | None = None
            for rule in rules:
                causes = self._eligible(
                    rule, causal_state, world, day, minute, absolute_minute
                )
                if causes is not None:
                    selected = (rule, causes)
                    break
            if selected is None:
                break
            rule, causes = selected
            causal_state["next_attempt_at"][rule.id] = (
                absolute_minute + max(1, rule.retry_minutes)
            )
            if rng.random() > rule.probability:
                continue
            occurrence = int(causal_state["occurrences"].get(rule.id, 0)) + 1
            event = event_factory(rule, causes, occurrence)
            causal_state["occurrences"][rule.id] = occurrence
            causal_state["revision"] += 1
            causal_state["rule_history"].append(
                {
                    "rule_id": rule.id,
                    "event_id": event["id"],
                    "event_kind": rule.event_kind,
                    "day": day,
                    "minute": minute,
                    "absolute_minute": absolute_minute,
                    "causes": list(causes),
                }
            )
            causal_state["event_causes"].extend(
                {
                    "cause_id": cause,
                    "effect_event_id": event["id"],
                    "rule_id": rule.id,
                }
                for cause in causes
            )
            emitted.append(event)
        return emitted

    def _eligible(
        self,
        rule: CausalRule,
        causal_state: dict[str, Any],
        world: dict[str, Any],
        day: int,
        minute: int,
        absolute_minute: int,
    ) -> list[str] | None:
        occurrence = int(causal_state["occurrences"].get(rule.id, 0))
        if rule.once and occurrence:
            return None
        if rule.max_occurrences is not None and occurrence >= rule.max_occurrences:
            return None
        if absolute_minute < int(causal_state["next_attempt_at"].get(rule.id, 0)):
            return None
        conditions = rule.conditions
        if day < int(conditions.get("min_day", 1)):
            return None
        if max_day := conditions.get("max_day"):
            if day > int(max_day):
                return None
        window = conditions.get("minute_window")
        if window and not int(window[0]) <= minute <= int(window[1]):
            return None
        if conditions.get("once_per_day"):
            if any(
                item["rule_id"] == rule.id and int(item["day"]) == day
                for item in causal_state["rule_history"]
            ):
                return None
        facts = causal_state["facts"]
        for key, threshold in conditions.get("fact_min", {}).items():
            if float(facts.get(key, 0.0)) < float(threshold):
                return None
        for key, threshold in conditions.get("fact_max", {}).items():
            if float(facts.get(key, 0.0)) > float(threshold):
                return None
        history = causal_state["rule_history"]
        latest_by_rule = {
            item["rule_id"]: item for item in history
        }
        required_rules = list(conditions.get("requires_rules", []))
        if any(rule_id not in latest_by_rule for rule_id in required_rules):
            return None
        any_rules = list(conditions.get("requires_any_rules", []))
        if any_rules and not any(rule_id in latest_by_rule for rule_id in any_rules):
            return None
        age_requirements = conditions.get("rule_age_minutes", {})
        for rule_id, minimum_age in age_requirements.items():
            item = latest_by_rule.get(rule_id)
            if item is None or absolute_minute - int(item["absolute_minute"]) < int(minimum_age):
                return None
        if conditions.get("active_incident_min") is not None:
            threshold = float(conditions["active_incident_min"])
            if not any(
                incident["status"] == "active"
                and float(incident["severity"]) >= threshold
                for incident in world.get("active_incidents", [])
            ):
                return None
        if required_incident := conditions.get("requires_active_incident_rule"):
            source = latest_by_rule.get(str(required_incident))
            if source is None or not any(
                incident["event_id"] == source["event_id"]
                and incident["status"] == "active"
                for incident in world.get("active_incidents", [])
            ):
                return None
        if resolved_rule := conditions.get("requires_resolved_incident_rule"):
            source = latest_by_rule.get(str(resolved_rule))
            if source is None or not any(
                incident["event_id"] == source["event_id"]
                and incident["status"] == "resolved"
                for incident in world.get("active_incidents", [])
            ):
                return None
        if rule.cooldown_minutes and history:
            prior = [item for item in history if item["rule_id"] == rule.id]
            if (
                prior
                and absolute_minute - int(prior[-1]["absolute_minute"])
                < rule.cooldown_minutes
            ):
                return None
        causes = [latest_by_rule[rule_id]["event_id"] for rule_id in required_rules]
        if any_rules:
            causes.extend(
                latest_by_rule[rule_id]["event_id"]
                for rule_id in any_rules
                if rule_id in latest_by_rule
            )
        causes.extend(f"fact:{key}" for key in conditions.get("fact_min", {}))
        return list(dict.fromkeys(causes))
