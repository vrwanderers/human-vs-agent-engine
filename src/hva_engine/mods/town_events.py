from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.causal_events import CausalRule, CausalRuleEngine


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return round(max(lower, min(upper, value)), 2)


def _unit(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


class TownEventDirector:
    """Causal authority for evolving weather, civic policy, incidents, and news."""

    RULES = (
        CausalRule(
            "daily_briefing",
            "mayor_speech",
            {"minute_window": [8 * 60, 9 * 60], "once_per_day": True},
            once=False,
            cooldown_minutes=20 * 60,
        ),
        CausalRule(
            "water_policy",
            "policy",
            {
                "requires_rules": ["daily_briefing"],
                "fact_min": {"water_pressure": 0.60},
                "minute_window": [9 * 60, 16 * 60],
            },
        ),
        CausalRule(
            "regional_supply_news",
            "world_news",
            {"min_day": 1, "minute_window": [11 * 60, 15 * 60]},
            probability=0.55,
            retry_minutes=60,
        ),
        CausalRule(
            "workshop_accident",
            "local_accident",
            {
                "fact_min": {"maintenance_risk": 0.62},
                "minute_window": [9 * 60, 19 * 60],
            },
            probability=0.28,
            once=False,
            cooldown_minutes=20 * 60,
            max_occurrences=2,
        ),
        CausalRule(
            "weather_alert",
            "weather_alert",
            {
                "fact_min": {"storm_pressure": 0.68},
                "minute_window": [15 * 60, 20 * 60],
            },
            probability=0.82,
            retry_minutes=60,
        ),
        CausalRule(
            "storm_arrival",
            "weather_change",
            {
                "requires_rules": ["weather_alert"],
                "rule_age_minutes": {"weather_alert": 360},
                "minute_window": [6 * 60, 11 * 60],
            },
        ),
        CausalRule(
            "flash_flood",
            "natural_disaster",
            {
                "requires_rules": ["storm_arrival"],
                "rule_age_minutes": {"storm_arrival": 60},
                "fact_min": {"drainage_stress": 0.65},
                "minute_window": [8 * 60, 18 * 60],
            },
        ),
        CausalRule(
            "shelter_notice",
            "town_announcement",
            {
                "requires_rules": ["flash_flood"],
                "requires_active_incident_rule": "flash_flood",
                "rule_age_minutes": {"flash_flood": 30},
            },
        ),
        CausalRule(
            "emergency_aid_policy",
            "policy",
            {
                "requires_rules": ["flash_flood"],
                "active_incident_min": 0.70,
                "rule_age_minutes": {"flash_flood": 90},
            },
        ),
        CausalRule(
            "storm_power_outage",
            "unexpected_event",
            {
                "requires_rules": ["storm_arrival"],
                "fact_min": {"grid_stress": 0.62},
                "minute_window": [10 * 60, 21 * 60],
            },
            probability=0.42,
            once=False,
            cooldown_minutes=18 * 60,
            max_occurrences=2,
        ),
        CausalRule(
            "recovery_news",
            "news",
            {
                "requires_rules": ["flash_flood"],
                "requires_resolved_incident_rule": "flash_flood",
                "minute_window": [7 * 60, 13 * 60],
            },
        ),
        CausalRule(
            "recovery_accountability",
            "mayor_speech",
            {
                "requires_rules": ["recovery_news"],
                "rule_age_minutes": {"recovery_news": 60},
            },
        ),
        CausalRule(
            "recovery_meeting",
            "town_announcement",
            {
                "requires_rules": ["recovery_accountability"],
                "rule_age_minutes": {"recovery_accountability": 120},
                "minute_window": [12 * 60, 20 * 60],
            },
        ),
    )

    def __init__(self) -> None:
        self.rule_engine = CausalRuleEngine()

    @staticmethod
    def initial_world(rng: Random, weather: str) -> dict[str, Any]:
        temperature = {
            "晴朗": rng.uniform(24, 29),
            "多云": rng.uniform(20, 25),
            "微雨": rng.uniform(17, 22),
        }[weather]
        return {
            "season": "夏末",
            "weather": weather,
            "temperature_c": round(temperature, 1),
            "wind": "东南风 2 级",
            "forecast": ["午后云量增加", "未来两日存在对流天气不确定性"],
            "risk_level": 0.08,
            "event_history": [],
            "news": [],
            "announcements": [],
            "policies": [],
            "active_incidents": [],
            "reactions": [],
            "causal_edges": [],
            "mayor": {
                "name": "顾岚",
                "role": "风铃镇镇长",
                "public_priorities": ["防灾", "供水", "公开信息"],
            },
            "bulletin_revision": 0,
        }

    @staticmethod
    def initial_causal_state(rng: Random) -> dict[str, Any]:
        return CausalRuleEngine.initial_state(
            {
                "public_readiness": rng.uniform(0.34, 0.48),
                "water_pressure": rng.uniform(0.60, 0.73),
                "supply_pressure": rng.uniform(0.48, 0.62),
                "maintenance_risk": rng.uniform(0.43, 0.51),
                "storm_pressure": rng.uniform(0.53, 0.61),
                "drainage_stress": rng.uniform(0.58, 0.67),
                "grid_stress": rng.uniform(0.42, 0.50),
                "response_capacity": rng.uniform(0.28, 0.39),
                "verification_capacity": rng.uniform(0.18, 0.31),
                "rumor_pressure": 0.12,
            }
        )

    def observe_action(
        self,
        state: dict[str, Any],
        *,
        action_type: str,
        job_id: str,
        location_id: str,
    ) -> None:
        facts = state["_causal_state"]["facts"]

        def adjust(key: str, amount: float) -> None:
            facts[key] = _unit(float(facts.get(key, 0.0)) + amount)

        if action_type == "work":
            adjust("public_readiness", 0.018)
            if job_id == "mechanic":
                adjust("maintenance_risk", -0.09)
                adjust("grid_stress", -0.045)
            elif job_id == "gardener":
                adjust("water_pressure", -0.035)
                adjust("drainage_stress", -0.025)
            elif job_id == "archivist":
                adjust("verification_capacity", 0.04)
        elif action_type == "respond_incident":
            adjust("response_capacity", 0.07)
            adjust("public_readiness", 0.03)
            if location_id in {"farm", "lake"}:
                adjust("drainage_stress", -0.045)
            if location_id in {"workshop", "library"}:
                adjust("grid_stress", -0.04)
        elif action_type in {"check_bulletin", "verify_claim", "investigate_claim"}:
            adjust("verification_capacity", 0.055)
            adjust("rumor_pressure", -0.065 if action_type == "investigate_claim" else -0.04)
        elif action_type in {"reshare_post", "publish_post"}:
            adjust("rumor_pressure", 0.025)

    def trigger_due(self, state: dict[str, Any], rng: Random) -> list[dict[str, Any]]:
        facts = state["_causal_state"]["facts"]
        facts["storm_pressure"] = _unit(float(facts["storm_pressure"]) + 0.009)
        facts["maintenance_risk"] = _unit(float(facts["maintenance_risk"]) + 0.007)
        if state["world"]["weather"] in {"暴雨前夕", "暴雨"}:
            facts["grid_stress"] = _unit(float(facts["grid_stress"]) + 0.018)
            facts["drainage_stress"] = _unit(float(facts["drainage_stress"]) + 0.016)

        def factory(rule: CausalRule, causes: list[str], occurrence: int) -> dict[str, Any]:
            event_id = f"world-{int(state['_causal_state']['revision']) + 1:05d}"
            return self._event_template(
                rule, event_id, causes, occurrence, state, rng
            )

        selected = self.rule_engine.evaluate(
            rules=self.RULES,
            causal_state=state["_causal_state"],
            world=state["world"],
            day=state["day"],
            minute=state["minute_of_day"],
            rng=rng,
            event_factory=factory,
        )
        emitted = [self._activate(state, event, rng) for event in selected]
        self._refresh_risk(state)
        return emitted

    def _event_template(
        self,
        rule: CausalRule,
        event_id: str,
        causes: list[str],
        occurrence: int,
        state: dict[str, Any],
        rng: Random,
    ) -> dict[str, Any]:
        facts = state["_causal_state"]["facts"]
        templates: dict[str, dict[str, Any]] = {
            "daily_briefing": {
                "source_id": "mayor-gu-lan",
                "title": "镇长晨间简报" if occurrence == 1 else f"镇长第{state['day']}日简报",
                "summary": "顾岚公布供水、气象与公共设施风险，并承诺保留公告修订记录。",
                "category": "mayor_speech",
                "severity": 0.24 + 0.25 * float(facts["public_readiness"]),
                "delivery": "broadcast",
                "modality": "language",
                "tags": ["mayor", "preparedness", "public_information"],
            },
            "water_policy": {
                "source_id": "town-hall",
                "title": "弹性节水与公共维护政策",
                "summary": "用水登记和公共维护补贴将随供水压力动态调整。",
                "category": "policy",
                "severity": float(facts["water_pressure"]),
                "delivery": "broadcast",
                "modality": "language",
                "tags": ["policy", "water", "work", "duty"],
                "policy": {
                    "policy_id": "water-accounting",
                    "effect": "public_work_bonus",
                    "active_until_day": state["day"] + 3,
                },
            },
            "regional_supply_news": {
                "source_id": "regional-news",
                "title": "区域铁路劳资谈判影响物资运输",
                "summary": "零件与邮件存在延迟风险，具体范围尚待铁路公司确认。",
                "category": "world_news",
                "severity": float(facts["supply_pressure"]),
                "delivery": "public_bulletin",
                "modality": "language",
                "tags": ["news", "railway", "supply", "uncertainty"],
            },
            "workshop_accident": {
                "source_id": "workshop-alarm",
                "title": "工坊传动系统故障",
                "summary": "维护风险累积导致设备停机，现场有碎片和二次故障风险。",
                "category": "accident",
                "severity": min(0.82, float(facts["maintenance_risk"]) + 0.08),
                "delivery": "local",
                "modality": "audio",
                "affected_locations": ["workshop"],
                "tags": ["accident", "machinery", "alarm", "injury_risk"],
                "incident_effort": 11.0 + 7.0 * float(facts["maintenance_risk"]),
            },
            "weather_alert": {
                "source_id": "weather-service",
                "title": "强对流与持续降雨预警",
                "summary": "气象压力持续上升，镜湖和农场低地存在积水风险。",
                "category": "weather",
                "severity": float(facts["storm_pressure"]),
                "delivery": "broadcast",
                "modality": "language",
                "affected_locations": ["lake", "farm"],
                "tags": ["weather", "heavy_rain", "flood", "warning"],
                "weather_update": {
                    "weather": "暴雨前夕",
                    "temperature_c": 18.0,
                    "wind": "东北风 5 级",
                },
            },
            "storm_arrival": {
                "source_id": "world-weather",
                "title": "暴雨抵达风铃镇",
                "summary": "预警中的雨带抵达，能见度和道路通行条件持续下降。",
                "category": "weather",
                "severity": min(0.9, float(facts["storm_pressure"]) + 0.08),
                "delivery": "environment",
                "modality": "world_event",
                "affected_locations": ["farm", "lake", "square"],
                "tags": ["weather", "storm", "visibility", "route_change"],
                "weather_update": {
                    "weather": "暴雨",
                    "temperature_c": 16.5,
                    "wind": "东北风 6 级",
                },
            },
            "flash_flood": {
                "source_id": "town-siren",
                "title": "镜湖支流突发洪水",
                "summary": "持续暴雨叠加排水压力，农场低地进水，道路需要加固。",
                "category": "natural_disaster",
                "severity": min(0.94, float(facts["drainage_stress"]) + 0.15),
                "delivery": "broadcast",
                "modality": "audio",
                "affected_locations": ["farm", "lake"],
                "tags": ["disaster", "flood", "family", "mutual_aid", "danger"],
                "incident_effort": 24.0 + 18.0 * float(facts["drainage_stress"]),
            },
            "shelter_notice": {
                "source_id": "town-hall",
                "title": "小镇避险公告",
                "summary": "旅店开放为避险点，非救援人员应远离湖岸并核对官方更新。",
                "category": "town_announcement",
                "severity": 0.78,
                "delivery": "broadcast",
                "modality": "language",
                "affected_locations": ["lake", "inn"],
                "tags": ["announcement", "shelter", "flood", "safety"],
            },
            "emergency_aid_policy": {
                "source_id": "mayor-gu-lan",
                "title": "紧急互助政策",
                "summary": "救援、维修和照顾邻居获得补贴；物资去向需公开登记。",
                "category": "policy",
                "severity": 0.68,
                "delivery": "broadcast",
                "modality": "language",
                "tags": ["policy", "mutual_aid", "relief", "registration"],
                "policy": {
                    "policy_id": "emergency-mutual-aid",
                    "effect": "incident_response_bonus",
                    "active_until_day": state["day"] + 2,
                },
            },
            "storm_power_outage": {
                "source_id": "power-grid",
                "title": "东区电网故障",
                "summary": "暴雨和电网压力导致工坊、图书馆部分停电。",
                "category": "unexpected_event",
                "severity": min(0.78, float(facts["grid_stress"]) + 0.08),
                "delivery": "local",
                "modality": "world_event",
                "affected_locations": ["workshop", "library"],
                "tags": ["power_outage", "unexpected", "darkness", "equipment"],
                "incident_effort": 12.0 + 10.0 * float(facts["grid_stress"]),
            },
            "recovery_news": {
                "source_id": "willow-gazette",
                "title": "洪水处置进入恢复阶段",
                "summary": "水位回落但道路仍需检查；网络流传的水坝垮塌说法缺乏证据。",
                "category": "news",
                "severity": 0.36,
                "delivery": "public_bulletin",
                "modality": "language",
                "tags": ["news", "recovery", "misinformation", "evidence"],
            },
            "recovery_accountability": {
                "source_id": "mayor-gu-lan",
                "title": "镇长灾后说明",
                "summary": "顾岚公布物资和道路进度，并承认预警覆盖与信息澄清仍有缺口。",
                "category": "mayor_speech",
                "severity": 0.44,
                "delivery": "broadcast",
                "modality": "language",
                "tags": ["mayor", "accountability", "recovery", "public_records"],
            },
            "recovery_meeting": {
                "source_id": "town-hall",
                "title": "灾后公共会议",
                "summary": "广场会议将讨论防洪、平台谣言治理和长期公共预算。",
                "category": "town_announcement",
                "severity": 0.30,
                "delivery": "broadcast",
                "modality": "language",
                "affected_locations": ["square"],
                "tags": ["announcement", "meeting", "policy", "community"],
            },
        }
        event = {
            "id": event_id,
            "rule_id": rule.id,
            "kind": rule.event_kind,
            "causes": causes,
            **templates[rule.id],
        }
        event["severity"] = round(float(event["severity"]), 3)
        if "incident_effort" in event:
            event["incident_effort"] = round(float(event["incident_effort"]), 2)
        return event

    def _activate(
        self, state: dict[str, Any], event: dict[str, Any], rng: Random
    ) -> dict[str, Any]:
        visible_to = self._recipients(state, event, rng)
        record = {
            key: deepcopy(value)
            for key, value in event.items()
            if key not in {"incident_effort", "policy", "weather_update"}
        }
        record.update(
            {
                "happened_day": state["day"],
                "happened_time": state["time"],
                "discoverable": event["delivery"] != "local",
            }
        )
        state["world"]["event_history"].append(record)
        state["world"]["causal_edges"].extend(
            {
                "cause_id": cause,
                "effect_event_id": event["id"],
                "rule_id": event["rule_id"],
            }
            for cause in event["causes"]
        )
        if record["discoverable"]:
            state["world"]["bulletin_revision"] += 1
        for actor_id in visible_to:
            known = state["_knowledge"].setdefault(actor_id, [])
            if event["id"] not in known:
                known.append(event["id"])
        if event["category"] in {"news", "world_news"}:
            state["world"]["news"].append(record)
        if event["category"] in {"mayor_speech", "town_announcement", "weather"}:
            state["world"]["announcements"].append(record)
        if policy := event.get("policy"):
            state["world"]["policies"].append(
                {"event_id": event["id"], "title": event["title"], **deepcopy(policy)}
            )
        if update := event.get("weather_update"):
            state["world"].update(deepcopy(update))
            state["weather"] = update["weather"]
        if effort := float(event.get("incident_effort", 0.0)):
            incident = {
                "incident_id": event["id"],
                "event_id": event["id"],
                "rule_id": event["rule_id"],
                "title": event["title"],
                "category": event["category"],
                "severity": event["severity"],
                "affected_locations": list(event.get("affected_locations", [])),
                "remaining_effort": effort,
                "initial_effort": effort,
                "status": "active",
            }
            state["world"]["active_incidents"].append(incident)
            state["town_progress"] = max(
                0.0, round(state["town_progress"] - 4.5 * float(event["severity"]), 2)
            )
            for resident in state["residents"].values():
                if resident["location"] in incident["affected_locations"]:
                    resident["mood"] = _clamp(
                        resident["mood"] - 9 * float(event["severity"])
                    )
                    resident["energy"] = _clamp(
                        resident["energy"] - 4 * float(event["severity"])
                    )
        return {
            "type": f"town_world_{event['kind']}",
            "_actor_id": event["source_id"],
            "_visible_to": visible_to,
            "event_id": event["id"],
            "rule_id": event["rule_id"],
            "causes": list(event["causes"]),
            "headline": event["title"],
            "summary": event["summary"],
            "category": event["category"],
            "severity": event["severity"],
            "affected_locations": list(event.get("affected_locations", [])),
            "delivery": event["delivery"],
            "stimulus": {
                "source_id": event["source_id"],
                "modality": event["modality"],
                "semantic_tags": list(event["tags"]),
                "intensity": event["severity"],
                "valence": -event["severity"]
                if event["category"]
                in {"accident", "natural_disaster", "unexpected_event", "weather"}
                else 0.05,
                "urgency": min(1.0, float(event["severity"]) + 0.12),
                "novelty": 0.76,
                "uncertainty": 0.12 if event["source_id"] != "regional-news" else 0.38,
                "reality_status": "canonical",
                "causal_group": event["id"],
            },
        }

    @staticmethod
    def _recipients(
        state: dict[str, Any], event: dict[str, Any], rng: Random
    ) -> list[str]:
        actor_ids = list(state["residents"])
        if event["delivery"] in {"broadcast", "environment"}:
            return actor_ids
        affected = set(event.get("affected_locations", []))
        if event["delivery"] == "local":
            return [
                actor_id
                for actor_id, resident in state["residents"].items()
                if resident["location"] in affected
            ]
        bulletin_locations = {"square", "library", "inn"}
        recipients = [
            actor_id
            for actor_id, resident in state["residents"].items()
            if resident["location"] in bulletin_locations
        ]
        if not recipients and actor_ids:
            recipients.append(rng.choice(actor_ids))
        return recipients

    def check_bulletin(self, state: dict[str, Any], actor_id: str) -> list[str]:
        learned: list[str] = []
        known = state["_knowledge"].setdefault(actor_id, [])
        for event in state["world"]["event_history"]:
            if event.get("discoverable") and event["id"] not in known:
                known.append(event["id"])
                learned.append(event["id"])
        return learned

    def observe_location(
        self, state: dict[str, Any], actor_id: str, location_id: str
    ) -> list[dict[str, Any]]:
        known = state["_knowledge"].setdefault(actor_id, [])
        records = {record["id"]: record for record in state["world"]["event_history"]}
        discovered: list[dict[str, Any]] = []
        for incident in state["world"]["active_incidents"]:
            event_id = str(incident["event_id"])
            if (
                incident["status"] != "active"
                or location_id not in incident["affected_locations"]
                or event_id in known
            ):
                continue
            known.append(event_id)
            record = records[event_id]
            discovered.append(
                {
                    "type": "town_incident_discovered",
                    "_actor_id": record["source_id"],
                    "_visible_to": [actor_id],
                    "event_id": event_id,
                    "incident_id": incident["incident_id"],
                    "headline": record["title"],
                    "summary": record["summary"],
                    "category": record["category"],
                    "severity": record["severity"],
                    "affected_locations": list(incident["affected_locations"]),
                    "delivery": "location_observation",
                    "stimulus": {
                        "source_id": record["source_id"],
                        "modality": "world_event",
                        "semantic_tags": [
                            "incident",
                            "direct_observation",
                            str(record["category"]),
                        ],
                        "intensity": record["severity"],
                        "valence": -float(record["severity"]),
                        "urgency": min(1.0, float(record["severity"]) + 0.12),
                        "novelty": 0.68,
                        "uncertainty": 0.08,
                        "reality_status": "canonical",
                        "causal_group": event_id,
                    },
                }
            )
        return discovered

    def respond_incident(
        self,
        state: dict[str, Any],
        actor_id: str,
        incident_id: str,
        effort: float,
    ) -> dict[str, Any]:
        incident = next(
            item
            for item in state["world"]["active_incidents"]
            if item["incident_id"] == incident_id and item["status"] == "active"
        )
        before = float(incident["remaining_effort"])
        incident["remaining_effort"] = round(max(0.0, before - effort), 2)
        if incident["remaining_effort"] == 0:
            incident["status"] = "resolved"
            state["community_warmth"] = _clamp(state["community_warmth"] + 4.5)
        reaction = {
            "actor_id": actor_id,
            "incident_id": incident_id,
            "effort": round(effort, 2),
            "resolved": incident["status"] == "resolved",
            "turn": state["turn"],
        }
        state["world"]["reactions"].append(reaction)
        self._refresh_risk(state)
        return reaction

    @staticmethod
    def visible_world(state: dict[str, Any], actor_id: str | None) -> dict[str, Any]:
        world = deepcopy(state["world"])
        if actor_id is None or state["residents"].get(actor_id, {}).get("kind") == "human":
            return world
        known = set(state["_knowledge"].get(actor_id, []))
        world["event_history"] = [
            event for event in world["event_history"] if event["id"] in known
        ]
        world["news"] = [event for event in world["news"] if event["id"] in known]
        world["announcements"] = [
            event for event in world["announcements"] if event["id"] in known
        ]
        world["policies"] = [
            policy for policy in world["policies"] if policy["event_id"] in known
        ]
        world["active_incidents"] = [
            incident
            for incident in world["active_incidents"]
            if incident["event_id"] in known
        ]
        world["causal_edges"] = [
            edge
            for edge in world["causal_edges"]
            if edge["effect_event_id"] in known
            and (str(edge["cause_id"]).startswith("fact:") or edge["cause_id"] in known)
        ]
        world["bulletin_revision"] = sum(
            bool(event.get("discoverable")) for event in world["event_history"]
        )
        incident_risk = max(
            (
                float(incident["severity"])
                for incident in world["active_incidents"]
                if incident["status"] == "active"
            ),
            default=0.0,
        )
        weather_risk = 0.72 if world["weather"] == "暴雨" else 0.12
        world["risk_level"] = round(max(incident_risk, weather_risk), 3)
        return world

    @staticmethod
    def _refresh_risk(state: dict[str, Any]) -> None:
        active = [
            item for item in state["world"]["active_incidents"] if item["status"] == "active"
        ]
        incident_risk = max((float(item["severity"]) for item in active), default=0.0)
        weather_risk = 0.72 if state["world"]["weather"] == "暴雨" else 0.12
        state["world"]["risk_level"] = round(max(incident_risk, weather_risk), 3)
