from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from random import Random
from typing import Any

from hva_engine.cognition import (
    AgentIdentity,
    AutobiographicalMemory,
    CognitiveProfile,
    RuntimeBehaviorPolicy,
)
from hva_engine.models import Action, Player
from hva_engine.mods.base import GameMod
from hva_engine.mods.town_events import TownEventDirector
from hva_engine.social_media import SocialMediaHub, town_social_hub


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return round(max(lower, min(upper, value)), 2)


class AgentTown(GameMod):
    """A three-day social sandbox for observing independent AgentBrain runtimes."""

    id = "agent_town"
    display_name = "Agent 小镇"
    description = "三名独立 Agent 在像素小镇中通勤、工作、社交、休息并形成技能。"
    tags = ("sandbox", "social", "life-sim", "spatial", "cooperation")
    capabilities = frozenset(
        {
            "turn_based",
            "numeric_state",
            "spatial",
            "text_state",
            "shared_objective",
            "audience_input",
        }
    )
    supported_modes = ("human_agent_coop", "agent_coop")
    competitive_balance_applicable = False
    supports_persistent_world = True
    score_ceiling = 1.0
    cooperation_event_types = (
        "town_conversation",
        "town_worked",
        "town_incident_response",
        "town_neighbor_supported",
    )

    LOCATIONS: dict[str, dict[str, Any]] = {
        "farm": {"name": "晨露农场", "x": 150, "y": 150, "kind": "work"},
        "workshop": {"name": "齿轮工坊", "x": 765, "y": 150, "kind": "work"},
        "square": {"name": "风铃广场", "x": 455, "y": 295, "kind": "social"},
        "library": {"name": "月桂图书馆", "x": 770, "y": 430, "kind": "work"},
        "lake": {"name": "镜湖", "x": 155, "y": 430, "kind": "nature"},
        "inn": {"name": "橡果旅店", "x": 460, "y": 520, "kind": "home"},
    }
    JOBS = (
        ("gardener", "farm", "锄头与种子"),
        ("mechanic", "workshop", "扳手与车床"),
        ("archivist", "library", "目录与修复工具"),
        ("visitor", "square", "旅行笔记"),
    )
    WEATHER = ("晴朗", "多云", "微雨")

    def __init__(self) -> None:
        self.event_director = TownEventDirector()
        self.social_hub: SocialMediaHub = town_social_hub()

    @staticmethod
    def _new_social_knowledge() -> dict[str, Any]:
        return {
            "seen_posts": [],
            "known_claims": [],
            "published_event_ids": [],
            "reshared_post_ids": [],
            "claim_beliefs": {},
            "source_trust": {},
            "verification_attempts": {},
            "investigated_claims": [],
            "last_phone_turn": -999,
        }

    def agent_count_for_mode(self, mode: str) -> int:
        return 3 if mode == "human_agent_coop" else 4

    def social_platform_manifest(self) -> list[dict[str, Any]]:
        return [
            platform.manifest() for platform in self.social_hub.platforms.values()
        ]

    def agent_character(
        self,
        state: dict[str, Any],
        player: Player,
        role: str,
        behavior_policy: RuntimeBehaviorPolicy,
        memory_owner_id: str,
        rng: Random,
    ) -> tuple[CognitiveProfile, AgentIdentity] | None:
        seed = (
            int(state.get("_identity_seeds", {}).get(player.id, 0))
            if memory_owner_id == player.id
            else int(
                sha256(f"{self.id}:{memory_owner_id}".encode()).hexdigest()[:16], 16
            )
        )
        identity_rng = Random(seed)
        profile = CognitiveProfile.sample(identity_rng, role, behavior_policy)
        base = AgentIdentity.sample(player.name, profile, role, identity_rng)
        resident = state["residents"][player.id]
        job_id = str(resident["job_id"])
        job = {
            "gardener": {
                "background": "在晨露农场长大，懂土壤、节气和修补旧农具。",
                "aspiration": "让小镇在坏天气和歉收之后仍能吃上稳定的粮食。",
                "wound": "少年时误判暴雨，没能及时转移温室幼苗，家里损失了整季收入。",
                "values": ("照料", "耐心", "土地", "家人"),
                "style": "说话朴实，习惯先观察天气和人的脸色",
            },
            "mechanic": {
                "background": "在齿轮工坊做机械师，擅长把报废零件拼成还能工作的机器。",
                "aspiration": "造出一套能在灾害中维持小镇供水和照明的设备。",
                "wound": "一次赶工时忽略异响，机器故障让师父受伤，此后对失控声响格外敏感。",
                "values": ("可靠", "技术", "责任", "独立"),
                "style": "偏技术化、简短，紧张时会反复确认细节",
            },
            "archivist": {
                "background": "负责月桂图书馆的档案与地方新闻，熟悉小镇旧政策和灾害记录。",
                "aspiration": "建立人人都能查证的公共档案，阻止谣言替代事实。",
                "wound": "曾因相信一条未经核实的消息，错过与病重亲人最后见面的机会。",
                "values": ("事实", "记忆", "公共责任", "审慎"),
                "style": "表达克制而有条理，对来源不明的说法保持距离",
            },
            "visitor": {
                "background": "以旅行写作为生，暂住橡果旅店，靠观察人群和地方新闻理解世界。",
                "aspiration": "写出不消费他人苦难、又能让外界真正理解小镇的记录。",
                "wound": "过去一篇报道暴露了受访者隐私，虽然事实无误，却破坏了对方的生活。",
                "values": ("好奇", "同意", "自由", "诚实"),
                "style": "善于提问，语气温和，但遇到矛盾时会追问到底",
            },
        }[job_id]
        family_member = identity_rng.choice(
            ("姐姐岚", "伴侣青禾", "父亲老林", "弟弟小满", "女儿萤")
        )
        formative = (
            AutobiographicalMemory(
                "职业上最昂贵的一课",
                job["wound"],
                -0.78,
                "紧迫感不能代替核实，异常信号必须被认真对待。",
                themes=(job_id, "failure", "responsibility", "emergency"),
                place=resident["workplace"],
                time_period="来到风铃镇以前",
            ),
            AutobiographicalMemory(
                "暴雪夜的互助",
                f"我记得停电的暴雪夜，{family_member}和邻居们轮流守着炉火，"
                "把仅有的热汤先给了孩子和老人。",
                0.62,
                "共同体不是口号，而是谁在困难时愿意留下来。",
                people=(family_member, "邻居们"),
                themes=("family", "weather", "mutual_aid", "belonging"),
                place="橡果旅店",
                time_period="数年前的冬天",
            ),
            AutobiographicalMemory(
                "留在小镇的承诺",
                f"我向{family_member}承诺，不论镇上的政策和行情怎样变化，"
                f"都会认真追求这件事：{job['aspiration']}",
                0.45,
                "长期承诺需要在每天的小选择中兑现。",
                people=(family_member,),
                themes=("promise", "future", job_id),
                place="风铃广场",
                time_period="入住小镇的第一年",
            ),
        )
        lived = (
            AutobiographicalMemory(
                "镜湖边的一次野餐",
                f"一个难得没有工作的下午，我和{family_member}在镜湖边吃冷面包，"
                "看云影从水面慢慢移过去。",
                0.82,
                "平静的日常值得保护，也值得被记住。",
                people=(family_member,),
                themes=("family", "rest", "lake", "happiness"),
                place="镜湖",
                time_period="去年夏末",
            ),
            AutobiographicalMemory(
                "公告栏前的争执",
                "镇里发布新规时，我曾和一位邻居争得很凶；后来发现我们看到的公告版本不同。",
                -0.18,
                "先确认共同事实，再争论立场。",
                people=("一位邻居",),
                themes=("policy", "news", "conflict", "evidence"),
                place="风铃广场",
                time_period="几个月前",
            ),
        )
        speech_style = {
            **base.speech_style,
            "constraint_source": "town_private_identity_generator",
        }
        identity = replace(
            base,
            background=job["background"],
            aspiration=job["aspiration"],
            core_wound=job["wound"],
            values=job["values"],
            social_style=job["style"],
            formative_memories=formative,
            lived_memories=lived,
            speech_style=speech_style,
            motive_weights={
                "duty": 0.78 if job_id != "visitor" else 0.52,
                "care": 0.72,
                "truth": 0.82 if job_id in {"archivist", "visitor"} else 0.58,
                "self_preservation": 0.62,
            },
            commitment_weights={
                "aspiration": 0.82,
                "relationship": 0.72,
                "core_values": 0.78,
            },
        )
        return profile, identity

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        order = [player.id for player in players]
        residents: dict[str, dict[str, Any]] = {}
        identity_seeds: dict[str, int] = {}
        agent_index = 0
        for player in players:
            if player.kind.value == "human":
                job_id, workplace, toolset = self.JOBS[-1]
            else:
                job_id, workplace, toolset = self.JOBS[agent_index % len(self.JOBS)]
                agent_index += 1
            start_location = "square" if player.kind.value == "human" else "inn"
            point = self.LOCATIONS[start_location]
            residents[player.id] = {
                "name": player.name,
                "kind": player.kind.value,
                "location": start_location,
                "x": point["x"],
                "y": point["y"],
                "home": "inn",
                "workplace": workplace,
                "job_id": job_id,
                "job_name": {
                    "gardener": "园艺师",
                    "mechanic": "机械师",
                    "archivist": "档案员",
                    "visitor": "旅行者",
                }[job_id],
                "toolset": toolset,
                "energy": float(rng.randint(72, 88)),
                "mood": float(rng.randint(52, 72)),
                "coins": 12.0,
                "work_xp": 0.0,
                "social_xp": 0.0,
                "civic_xp": 0.0,
                "activity": "观察小镇" if player.kind.value == "human" else "刚刚醒来",
                "last_dialogue": "",
                "last_phone_turn": -999,
                "relationships": {
                    other.id: 50.0 for other in players if other.id != player.id
                },
            }
            identity_seeds[player.id] = rng.getrandbits(64)
        weather = rng.choice(self.WEATHER)
        return {
            "turn": 0,
            "day": 1,
            "minute_of_day": 7 * 60,
            "time": "07:00",
            "weather": weather,
            "environment_version": "willow-town-v1",
            "map_size": {"width": 920, "height": 600},
            "locations": deepcopy(self.LOCATIONS),
            "order": order,
            "residents": residents,
            "_identity_seeds": identity_seeds,
            "town_progress": 0.0,
            "community_warmth": 50.0,
            "recent_dialogue": [],
            "world": self.event_director.initial_world(rng, weather),
            "_causal_state": self.event_director.initial_causal_state(rng),
            "_knowledge": {player.id: [] for player in players},
            "social_media": self.social_hub.initial_state(),
            "_social_knowledge": {
                player.id: self._new_social_knowledge()
                for player in players
            },
            "session_start_day": 1,
            "session_end_day": 3,
            "finished": False,
        }

    def public_state(
        self, state: dict[str, Any], viewer_id: str | None = None
    ) -> dict[str, Any]:
        public = deepcopy(state)
        public.pop("_causal_state", None)
        public.pop("_knowledge", None)
        public.pop("_social_knowledge", None)
        public.pop("_memory_owner_ids", None)
        public.pop("_identity_seeds", None)
        public["world"] = self.event_director.visible_world(state, viewer_id)
        social = self.social_hub.public_state(state["social_media"])
        if viewer_id is not None and state["residents"].get(viewer_id, {}).get("kind") == "agent":
            knowledge = state["_social_knowledge"].get(viewer_id, {})
            visible_posts = set(knowledge.get("seen_posts", []))
            owner_id = self._owner_id(state, viewer_id)
            social["posts"] = [
                post
                for post in social["posts"]
                if post["id"] in visible_posts or post["author_id"] == owner_id
            ]
            visible_claims = {
                post["claim_id"] for post in social["posts"] if post.get("claim_id")
            }
            social["claims"] = {
                claim_id: claim
                for claim_id, claim in social["claims"].items()
                if claim_id in visible_claims
            }
            visible_post_ids = {post["id"] for post in social["posts"]}
            social["comments"] = [
                comment
                for comment in social["comments"]
                if comment["post_id"] in visible_post_ids
            ]
        public["social_media"] = social
        return public

    def persistent_world_state(self, state: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "schema_version": "agent-town-world.v2",
            "day": state["day"],
            "minute_of_day": state["minute_of_day"],
            "time": state["time"],
            "weather": state["weather"],
            "world": deepcopy(state["world"]),
            "causal_state": deepcopy(state["_causal_state"]),
            "social_media": deepcopy(state["social_media"]),
            "town_progress": state["town_progress"],
            "community_warmth": state["community_warmth"],
        }

    def restore_persistent_world(
        self, state: dict[str, Any], snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        restored = deepcopy(state)
        restored.update(
            {
                "day": int(snapshot["day"]),
                "minute_of_day": int(snapshot["minute_of_day"]),
                "time": str(snapshot["time"]),
                "weather": str(snapshot["weather"]),
                "world": deepcopy(snapshot["world"]),
                "_causal_state": deepcopy(snapshot["causal_state"]),
                "social_media": deepcopy(snapshot["social_media"]),
                "town_progress": float(snapshot.get("town_progress", 0.0)),
                "community_warmth": float(snapshot.get("community_warmth", 50.0)),
                "turn": 0,
                "finished": False,
            }
        )
        restored["session_start_day"] = restored["day"]
        restored["session_end_day"] = restored["day"] + 2
        restored["_knowledge"] = {actor_id: [] for actor_id in restored["residents"]}
        restored["_social_knowledge"] = {
            actor_id: self._new_social_knowledge()
            for actor_id in restored["residents"]
        }
        return restored

    @staticmethod
    def _owner_id(state: dict[str, Any], actor_id: str) -> str:
        return str(state.get("_memory_owner_ids", {}).get(actor_id, actor_id))

    def _social_affinity(
        self, state: dict[str, Any], actor_id: str
    ) -> dict[str, float]:
        resident = state["residents"][actor_id]
        affinity = {
            self._owner_id(state, other_id): float(value) / 100
            for other_id, value in resident["relationships"].items()
        }
        affinity.update(
            {
                source_id: float(value)
                for source_id, value in state["_social_knowledge"][actor_id]
                .get("source_trust", {})
                .items()
            }
        )
        return affinity

    @staticmethod
    def _available_claim_evidence(
        state: dict[str, Any], actor_id: str, claim: dict[str, Any]
    ) -> list[str]:
        known = set(state["_knowledge"].get(actor_id, []))
        evidence = claim.get("_evidence", {})
        return sorted(
            known
            & {
                str(event_id)
                for relation in ("supports", "refutes")
                for event_id in evidence.get(relation, [])
            }
        )

    @staticmethod
    def _update_claim_belief(
        knowledge: dict[str, Any], appraisal: dict[str, Any]
    ) -> dict[str, Any] | None:
        claim_id = appraisal.get("claim_id")
        if not claim_id:
            return None
        beliefs = knowledge.setdefault("claim_beliefs", {})
        belief = beliefs.setdefault(
            str(claim_id),
            {
                "acceptance": 0.5,
                "uncertainty": 1.0,
                "exposures": 0,
                "independent_sources": [],
                "last_public_status": "unverified",
            },
        )
        source_key = str(appraisal["independent_source_key"])
        independent = source_key not in belief["independent_sources"]
        if independent:
            belief["independent_sources"].append(source_key)
        belief["exposures"] += 1
        status = str(appraisal["public_status"])
        credibility = float(appraisal["credibility_estimate"])
        if status == "verified_true":
            acceptance = 0.97
        elif status == "verified_false":
            acceptance = 0.03
        elif status == "disputed":
            acceptance = 0.5
        else:
            weight = 0.34 if independent else 0.04
            acceptance = float(belief["acceptance"]) + (credibility - 0.5) * weight
        belief["acceptance"] = round(max(0.02, min(0.98, acceptance)), 3)
        source_diversity = len(belief["independent_sources"])
        belief["uncertainty"] = round(
            max(
                0.06,
                min(
                    1.0,
                    float(appraisal["uncertainty"])
                    - min(0.24, max(0, source_diversity - 1) * 0.06),
                ),
            ),
            3,
        )
        belief["last_public_status"] = status
        return belief

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        if self.is_terminal(state):
            return None
        order = state["order"]
        return order[state["turn"] % len(order)]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        resident = state["residents"][actor_id]
        current_location = resident["location"]
        actions = [Action(type="wait")]
        actions.extend(
            Action(
                type="move_to",
                payload={
                    "destination": location_id,
                    "route_id": f"{current_location}->{location_id}",
                },
            )
            for location_id in self.LOCATIONS
            if location_id != current_location
        )
        if current_location == resident["workplace"] and resident["energy"] >= 8:
            actions.append(
                Action(
                    type="work",
                    payload={
                        "job_id": resident["job_id"],
                        "task_id": f"daily-{resident['job_id']}",
                        "toolset": resident["toolset"],
                    },
                )
            )
        companions = [
            other_id
            for other_id, other in state["residents"].items()
            if other_id != actor_id and other["location"] == current_location
        ]
        actions.extend(
            Action(type="socialize", payload={"target_id": target_id})
            for target_id in companions
        )
        actions.append(Action(type="rest", payload={"location_id": current_location}))
        if current_location in {"lake", "square"}:
            actions.append(
                Action(type="explore", payload={"location_id": current_location})
            )
        known_events = set(state["_knowledge"].get(actor_id, []))
        known_incidents = [
            incident
            for incident in state["world"]["active_incidents"]
            if incident["status"] == "active"
            and incident["event_id"] in known_events
        ]
        for incident in known_incidents:
            if current_location in incident["affected_locations"] and resident["energy"] >= 10:
                actions.append(
                    Action(
                        type="respond_incident",
                        payload={
                            "incident_id": incident["incident_id"],
                            "location_id": current_location,
                        },
                    )
                )
        if (
            any(float(incident["severity"]) >= 0.68 for incident in known_incidents)
            and current_location != "inn"
        ):
            actions.append(Action(type="seek_shelter", payload={"destination": "inn"}))
        if current_location in {"square", "library", "inn"}:
            actions.append(Action(type="check_bulletin", payload={"location_id": current_location}))
        if companions and (
            state["world"]["risk_level"] >= 0.55
            or any(state["residents"][target]["mood"] < 52 for target in companions)
        ):
            target_id = min(companions, key=lambda target: state["residents"][target]["mood"])
            actions.append(Action(type="support_neighbor", payload={"target_id": target_id}))
        phone_available = (
            resident["kind"] == "human"
            or state["turn"] - int(resident.get("last_phone_turn", -999)) >= 8
        )
        if phone_available:
            actions.extend(
                Action(type="check_phone", payload={"platform_id": platform_id})
                for platform_id in self.social_hub.platforms
            )
        social_knowledge = state["_social_knowledge"].get(actor_id, {})
        owner_id = self._owner_id(state, actor_id)
        reshared = set(social_knowledge.get("reshared_post_ids", []))
        visible_posts = [
            post
            for post in state["social_media"]["posts"]
            if post["id"] in set(social_knowledge.get("seen_posts", []))
        ][-1:]
        for post in visible_posts:
            if post["author_id"] != owner_id and post["id"] not in reshared:
                actions.append(
                    Action(
                        type="reshare_post",
                        payload={
                            "platform_id": post["platform_id"],
                            "post_id": post["id"],
                        },
                    )
                )
            actions.append(
                Action(
                    type="comment_post",
                    payload={"post_id": post["id"], "stance": "question_or_discuss"},
                )
            )
            claim_id = post.get("claim_id")
            claim = state["social_media"]["claims"].get(claim_id, {})
            source_event_id = claim.get("source_event_id")
            evidence = self._available_claim_evidence(state, actor_id, claim)
            if (
                source_event_id
                and current_location == "library"
                and source_event_id not in known_events
            ):
                evidence.append(str(source_event_id))
            evidence = sorted(set(evidence))
            attempt_signature = "|".join(evidence) or "no-evidence"
            prior_attempt = social_knowledge.get("verification_attempts", {}).get(
                claim_id
            )
            if (
                claim
                and claim.get("public_status") in {"unverified", "unresolved"}
                and (current_location == "library" or bool(evidence))
                and prior_attempt != attempt_signature
            ):
                actions.append(
                    Action(
                        type="verify_claim",
                        payload={
                            "claim_id": claim_id,
                            "post_id": post["id"],
                            "evidence_event_ids": evidence,
                        },
                    )
                )
            belief = social_knowledge.get("claim_beliefs", {}).get(claim_id, {})
            needs_investigation = (
                claim
                and claim.get("public_status") in {"unverified", "unresolved"}
                and (
                    float(post.get("distortion", 0.0)) >= 0.35
                    or float(belief.get("uncertainty", 0.0)) >= 0.72
                )
                and claim_id
                not in set(social_knowledge.get("investigated_claims", []))
            )
            if needs_investigation and current_location in {"lake", "library"}:
                actions.append(
                    Action(
                        type="investigate_claim",
                        payload={
                            "claim_id": claim_id,
                            "post_id": post["id"],
                            "method": "field_observation"
                            if current_location == "lake"
                            else "source_audit",
                            "location_id": current_location,
                        },
                    )
                )
        published = set(social_knowledge.get("published_event_ids", []))
        known_records = [
            event
            for event in state["world"]["event_history"]
            if event["id"] in known_events
        ]
        if known_records:
            event_id = known_records[-1]["id"]
            platform_id = (
                "town_short_video"
                if resident["job_id"] in {"gardener", "visitor"}
                else "town_weibo"
            )
            publication_key = f"{platform_id}:{event_id}"
            if publication_key not in published:
                actions.append(
                    Action(
                        type="publish_post",
                        payload={
                            "platform_id": platform_id,
                            "event_id": event_id,
                            "content_kind": "short_video"
                            if platform_id == "town_short_video"
                            else "text",
                        },
                    )
                )
        return actions

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        resident = new["residents"][actor_id]
        emitted: list[dict[str, Any]] = []
        if action.type == "move_to":
            origin = resident["location"]
            destination = str(action.payload["destination"])
            point = self.LOCATIONS[destination]
            resident.update(
                {
                    "location": destination,
                    "x": point["x"],
                    "y": point["y"],
                    "energy": _clamp(resident["energy"] - 2.5),
                    "activity": f"前往{point['name']}",
                }
            )
            emitted.append(
                {
                    "type": "town_moved",
                    "from_location": origin,
                    "destination": destination,
                    "route_id": f"{origin}->{destination}",
                    "position": [point["x"], point["y"]],
                }
            )
            emitted.extend(
                self.event_director.observe_location(new, actor_id, destination)
            )
        elif action.type == "work":
            gain = round(5.0 + rng.uniform(0.0, 2.5), 2)
            policy_bonus = 0.8 if any(
                policy["policy_id"] == "water-accounting"
                and policy["active_until_day"] >= new["day"]
                for policy in new["world"]["policies"]
            ) else 0.0
            resident["energy"] = _clamp(resident["energy"] - 7.0)
            resident["coins"] = round(
                resident["coins"] + gain * 0.45 + policy_bonus, 2
            )
            resident["work_xp"] = round(resident["work_xp"] + gain, 2)
            resident["civic_xp"] = round(
                resident["civic_xp"] + policy_bonus * 0.5, 2
            )
            resident["mood"] = _clamp(resident["mood"] + 1.2)
            resident["activity"] = f"从事{resident['job_name']}工作"
            new["town_progress"] = round(new["town_progress"] + gain, 2)
            emitted.append(
                {
                    "type": "town_worked",
                    "job_id": resident["job_id"],
                    "progress": gain,
                    "location": resident["location"],
                    "policy_bonus": policy_bonus,
                }
            )
        elif action.type == "socialize":
            target_id = str(action.payload["target_id"])
            target = new["residents"][target_id]
            warmth = round(3.0 + rng.uniform(0.0, 2.0), 2)
            resident["relationships"][target_id] = _clamp(
                resident["relationships"].get(target_id, 50.0) + warmth
            )
            target["relationships"][actor_id] = _clamp(
                target["relationships"].get(actor_id, 50.0) + warmth * 0.7
            )
            resident["social_xp"] = round(resident["social_xp"] + warmth, 2)
            resident["mood"] = _clamp(resident["mood"] + 2.5)
            target["mood"] = _clamp(target["mood"] + 1.2)
            resident["energy"] = _clamp(resident["energy"] - 1.0)
            dialogue = str(
                action.payload.get("utterance")
                or (
                    f"{target['name']}，今天在"
                    f"{self.LOCATIONS[resident['location']]['name']}过得怎么样？"
                )
            )[:220]
            resident["last_dialogue"] = dialogue
            resident["activity"] = f"与{target['name']}交谈"
            new["community_warmth"] = _clamp(new["community_warmth"] + warmth * 0.18)
            new["recent_dialogue"] = [
                *new["recent_dialogue"][-5:],
                {
                    "speaker_id": actor_id,
                    "speaker": resident["name"],
                    "target_id": target_id,
                    "target": target["name"],
                    "text": dialogue,
                    "location": resident["location"],
                },
            ]
            emitted.append(
                {
                    "type": "town_conversation",
                    "target_id": target_id,
                    "dialogue": dialogue,
                    "warmth": warmth,
                    "location": resident["location"],
                }
            )
        elif action.type == "respond_incident":
            incident_id = str(action.payload["incident_id"])
            effort = round(5.0 + resident["work_xp"] / 35 + rng.uniform(0.0, 2.0), 2)
            reaction = self.event_director.respond_incident(
                new, actor_id, incident_id, effort
            )
            resident["energy"] = _clamp(resident["energy"] - 9.0)
            resident["mood"] = _clamp(resident["mood"] + 2.0)
            resident["civic_xp"] = round(resident["civic_xp"] + effort, 2)
            if any(
                policy["policy_id"] == "emergency-mutual-aid"
                and policy["active_until_day"] >= new["day"]
                for policy in new["world"]["policies"]
            ):
                resident["coins"] = round(resident["coins"] + effort * 0.25, 2)
            incident = next(
                item
                for item in new["world"]["active_incidents"]
                if item["incident_id"] == incident_id
            )
            resident["activity"] = f"响应：{incident['title']}"
            emitted.append(
                {
                    "type": "town_incident_response",
                    "incident_id": incident_id,
                    "effort": effort,
                    "resolved": reaction["resolved"],
                    "location": resident["location"],
                }
            )
        elif action.type == "seek_shelter":
            origin = resident["location"]
            point = self.LOCATIONS["inn"]
            resident.update(
                {
                    "location": "inn",
                    "x": point["x"],
                    "y": point["y"],
                    "energy": _clamp(resident["energy"] - 1.5),
                    "activity": "前往旅店避险",
                }
            )
            emitted.append(
                {
                    "type": "town_sheltered",
                    "from_location": origin,
                    "destination": "inn",
                    "position": [point["x"], point["y"]],
                }
            )
        elif action.type == "check_bulletin":
            learned = self.event_director.check_bulletin(new, actor_id)
            resident["activity"] = "核对新闻与公告"
            resident["energy"] = _clamp(resident["energy"] - 0.5)
            emitted.append(
                {
                    "type": "town_news_checked",
                    "learned_event_ids": learned,
                    "bulletin_revision": new["world"]["bulletin_revision"],
                    "location": resident["location"],
                }
            )
        elif action.type == "support_neighbor":
            target_id = str(action.payload["target_id"])
            target = new["residents"][target_id]
            target["mood"] = _clamp(target["mood"] + 5.0)
            resident["mood"] = _clamp(resident["mood"] + 1.5)
            resident["energy"] = _clamp(resident["energy"] - 2.0)
            resident["relationships"][target_id] = _clamp(
                resident["relationships"].get(target_id, 50.0) + 4.0
            )
            target["relationships"][actor_id] = _clamp(
                target["relationships"].get(actor_id, 50.0) + 5.0
            )
            resident["civic_xp"] = round(resident["civic_xp"] + 3.0, 2)
            resident["activity"] = f"安慰{target['name']}"
            emitted.append(
                {
                    "type": "town_neighbor_supported",
                    "target_id": target_id,
                    "location": resident["location"],
                }
            )
        elif action.type == "check_phone":
            platform_id = str(action.payload["platform_id"])
            knowledge = new["_social_knowledge"][actor_id]
            knowledge["last_phone_turn"] = new["turn"]
            resident["last_phone_turn"] = new["turn"]
            feed = self.social_hub.feed(
                new["social_media"],
                platform_id,
                limit=6,
                affinity_by_author=self._social_affinity(new, actor_id),
            )
            post_ids = [post["id"] for post in feed]
            self.social_hub.mark_viewed(new["social_media"], post_ids)
            appraisals: dict[str, dict[str, Any]] = {}
            for post in feed:
                source_trust = knowledge.get("source_trust", {}).get(post["author_id"])
                appraisal = self.social_hub.appraise_report(
                    new["social_media"],
                    post["id"],
                    source_trust=float(source_trust)
                    if source_trust is not None
                    else None,
                )
                self._update_claim_belief(knowledge, appraisal)
                appraisals[post["id"]] = appraisal
            knowledge["seen_posts"] = list(
                dict.fromkeys([*knowledge["seen_posts"], *post_ids])
            )[-32:]
            knowledge["known_claims"] = list(
                dict.fromkeys(
                    [
                        *knowledge["known_claims"],
                        *(post["claim_id"] for post in feed if post.get("claim_id")),
                    ]
                )
            )[-24:]
            resident["energy"] = _clamp(resident["energy"] - 0.5)
            resident["activity"] = f"浏览{self.social_hub.platforms[platform_id].display_name}"
            emitted.append(
                {
                    "type": "town_social_feed_checked",
                    "_visible_to": [actor_id],
                    "platform_id": platform_id,
                    "post_count": len(feed),
                    "posts": [
                        {
                            "post_id": post["id"],
                            "author_name": post["author_name"],
                            "content": post["content"],
                            "claim_id": post.get("claim_id"),
                            "distortion": post["distortion"],
                            "credibility_estimate": appraisals[post["id"]][
                                "credibility_estimate"
                            ],
                            "uncertainty": appraisals[post["id"]]["uncertainty"],
                            "independent_source_key": appraisals[post["id"]][
                                "independent_source_key"
                            ],
                        }
                        for post in feed
                    ],
                    "stimulus": {
                        "source_id": platform_id,
                        "modality": "language",
                        "semantic_tags": ["social_media", "feed", "public_opinion"],
                        "intensity": min(0.75, 0.18 + len(feed) * 0.08),
                        "valence": 0.0,
                        "urgency": 0.25,
                        "novelty": 0.55,
                        "uncertainty": 0.48,
                        "reality_status": "observed",
                        "causal_group": f"feed:{platform_id}:{new['social_media']['revision']}",
                    },
                }
            )
        elif action.type == "publish_post":
            platform_id = str(action.payload["platform_id"])
            event_id = str(action.payload["event_id"])
            record = next(
                event for event in new["world"]["event_history"] if event["id"] == event_id
            )
            claim_id = next(
                (
                    claim["id"]
                    for claim in new["social_media"]["claims"].values()
                    if claim.get("source_event_id") == event_id
                    and claim.get("_truth_status") == "true"
                ),
                None,
            )
            if claim_id is None:
                claim_id = self.social_hub.register_claim(
                    new["social_media"],
                    text=record["summary"],
                    source_event_id=event_id,
                    truth_status="true",
                    asserted_by=self._owner_id(new, actor_id),
                    supporting_event_ids=(event_id,),
                )
            content = str(
                action.payload.get("utterance")
                or f"我刚确认了“{record['title']}”：{record['summary']}"
            )
            post = self.social_hub.publish(
                new["social_media"],
                platform_id=platform_id,
                author_id=self._owner_id(new, actor_id),
                author_name=resident["name"],
                content=content,
                content_kind=str(action.payload.get("content_kind", "text")),
                source_event_ids=(event_id,),
                claim_id=claim_id,
                turn=new["turn"],
            )
            knowledge = new["_social_knowledge"][actor_id]
            knowledge["seen_posts"].append(post["id"])
            knowledge["published_event_ids"].append(f"{platform_id}:{event_id}")
            resident["social_xp"] = round(resident["social_xp"] + 1.2, 2)
            resident["activity"] = f"在{self.social_hub.platforms[platform_id].display_name}发帖"
            emitted.append(
                {
                    "type": "town_social_posted",
                    "_visible_to": [actor_id],
                    "post_id": post["id"],
                    "platform_id": platform_id,
                    "event_id": event_id,
                    "claim_id": claim_id,
                    "content": post["content"],
                }
            )
        elif action.type == "reshare_post":
            post_id = str(action.payload["post_id"])
            parent = self.social_hub.post(new["social_media"], post_id)
            claim = new["social_media"]["claims"].get(parent.get("claim_id"), {})
            uncertain = claim.get("public_status") in {None, "unverified", "unresolved"}
            distortion = rng.uniform(0.08, 0.22) if uncertain else rng.uniform(0.0, 0.05)
            content = str(
                action.payload.get("utterance")
                or f"转发：{parent['content']}（我还没有完全确认来源。）"
            )
            post = self.social_hub.publish(
                new["social_media"],
                platform_id=str(parent["platform_id"]),
                author_id=self._owner_id(new, actor_id),
                author_name=resident["name"],
                content=content,
                content_kind=str(parent["content_kind"]),
                source_event_ids=tuple(parent["source_event_ids"]),
                claim_id=parent.get("claim_id"),
                parent_post_id=post_id,
                distortion=distortion,
                turn=new["turn"],
            )
            knowledge = new["_social_knowledge"][actor_id]
            knowledge["seen_posts"].append(post["id"])
            knowledge["reshared_post_ids"].append(post_id)
            resident["social_xp"] = round(resident["social_xp"] + 0.8, 2)
            resident["activity"] = "转发并评论一条帖子"
            emitted.append(
                {
                    "type": "town_social_reshared",
                    "_visible_to": [actor_id],
                    "post_id": post["id"],
                    "parent_post_id": post_id,
                    "claim_id": post.get("claim_id"),
                    "distortion": post["distortion"],
                    "content": post["content"],
                }
            )
        elif action.type == "comment_post":
            post_id = str(action.payload["post_id"])
            parent = self.social_hub.post(new["social_media"], post_id)
            content = str(
                action.payload.get("utterance")
                or "这条消息的原始来源是什么？有人能提供可核对的记录吗？"
            )
            comment = self.social_hub.comment(
                new["social_media"],
                post_id=post_id,
                author_id=self._owner_id(new, actor_id),
                author_name=resident["name"],
                content=content,
                stance=str(action.payload.get("stance", "discuss")),
                turn=new["turn"],
            )
            resident["social_xp"] = round(resident["social_xp"] + 0.6, 2)
            resident["activity"] = f"讨论：{parent['content'][:18]}"
            emitted.append(
                {
                    "type": "town_social_commented",
                    "_visible_to": [actor_id],
                    "comment_id": comment["id"],
                    "post_id": post_id,
                    "claim_id": parent.get("claim_id"),
                    "content": comment["content"],
                }
            )
        elif action.type == "investigate_claim":
            claim_id = str(action.payload["claim_id"])
            claim = new["social_media"]["claims"][claim_id]
            evidence_id = f"investigation-{new['day']:03d}-{new['turn']:05d}"
            truth_status = str(claim.get("_truth_status", "unknown"))
            relation = {
                "true": "supports",
                "false": "refutes",
            }.get(truth_status)
            if relation:
                self.social_hub.attach_evidence(
                    new["social_media"],
                    claim_id=claim_id,
                    event_id=evidence_id,
                    relation=relation,
                )
            elif truth_status == "mixed":
                for mixed_relation in ("supports", "refutes"):
                    self.social_hub.attach_evidence(
                        new["social_media"],
                        claim_id=claim_id,
                        event_id=evidence_id,
                        relation=mixed_relation,
                    )
            finding = {
                "true": "现场观察和原始记录支持这条说法的核心命题。",
                "false": "现场与档案未发现命题所称情况，能够观察到相反证据。",
                "mixed": "不同记录分别支持和反驳命题的一部分，不能整体判真。",
                "unknown": "现有观察不足以支持或反驳这条说法。",
            }.get(truth_status, "现有观察不足以支持或反驳这条说法。")
            source_event_id = claim.get("source_event_id")
            record = {
                "id": evidence_id,
                "rule_id": "actor_investigation",
                "kind": "claim_investigation",
                "causes": [source_event_id] if source_event_id else [],
                "source_id": self._owner_id(new, actor_id),
                "title": "声明求证记录",
                "summary": finding,
                "category": "investigation",
                "severity": 0.34,
                "delivery": "local",
                "modality": "world_event",
                "affected_locations": [resident["location"]],
                "tags": ["investigation", "claim", "evidence"],
                "happened_day": new["day"],
                "happened_time": new["time"],
                "discoverable": False,
            }
            new["world"]["event_history"].append(record)
            if source_event_id:
                new["world"]["causal_edges"].append(
                    {
                        "cause_id": source_event_id,
                        "effect_event_id": evidence_id,
                        "rule_id": "actor_investigation",
                    }
                )
            new["_knowledge"][actor_id].append(evidence_id)
            knowledge = new["_social_knowledge"][actor_id]
            knowledge["investigated_claims"].append(claim_id)
            resident["energy"] = _clamp(resident["energy"] - 4.0)
            resident["civic_xp"] = round(resident["civic_xp"] + 3.5, 2)
            resident["activity"] = "调查一条社交媒体声明"
            emitted.append(
                {
                    "type": "town_claim_investigated",
                    "_visible_to": [actor_id],
                    "claim_id": claim_id,
                    "post_id": action.payload["post_id"],
                    "evidence_event_id": evidence_id,
                    "method": action.payload["method"],
                    "finding": finding,
                    "evidence_relation": relation or "mixed_or_unresolved",
                    "stimulus": {
                        "source_id": evidence_id,
                        "modality": "world_event",
                        "semantic_tags": ["investigation", "evidence", "claim"],
                        "intensity": 0.52,
                        "valence": 0.04,
                        "urgency": 0.38,
                        "novelty": 0.72,
                        "uncertainty": 0.18 if relation else 0.62,
                        "reality_status": "observed",
                        "causal_group": claim_id,
                    },
                }
            )
        elif action.type == "verify_claim":
            claim_id = str(action.payload["claim_id"])
            claim = new["social_media"]["claims"][claim_id]
            source_event_id = claim.get("source_event_id")
            requested_evidence = {
                str(event_id)
                for event_id in action.payload.get("evidence_event_ids", [])
            }
            if (
                source_event_id
                and str(source_event_id) in requested_evidence
                and resident["location"] == "library"
                and source_event_id not in new["_knowledge"][actor_id]
            ):
                new["_knowledge"][actor_id].append(str(source_event_id))
            evidence = sorted(
                requested_evidence & set(new["_knowledge"].get(actor_id, []))
            )
            result = self.social_hub.verify_claim(
                new["social_media"],
                claim_id=claim_id,
                verifier_id=self._owner_id(new, actor_id),
                evidence_event_ids=evidence,
                turn=new["turn"],
            )
            knowledge = new["_social_knowledge"][actor_id]
            knowledge.setdefault("verification_attempts", {})[claim_id] = (
                "|".join(evidence) or "no-evidence"
            )
            parent = self.social_hub.post(
                new["social_media"], str(action.payload["post_id"])
            )
            if result["result"] in {"verified_true", "verified_false"}:
                trust = float(
                    knowledge.setdefault("source_trust", {}).get(
                        parent["author_id"],
                        new["social_media"]
                        .get("accounts", {})
                        .get(parent["author_id"], {})
                        .get("reputation", 0.5),
                    )
                )
                adjustment = 0.08 if result["result"] == "verified_true" else -0.16
                knowledge["source_trust"][parent["author_id"]] = round(
                    max(0.05, min(0.95, trust + adjustment)), 3
                )
            status_text = {
                "verified_true": "与可核对记录一致",
                "verified_false": "与可核对记录冲突",
                "disputed": "只能确认部分内容",
                "unresolved": "目前证据不足",
            }[result["result"]]
            fact_check = self.social_hub.publish(
                new["social_media"],
                platform_id="town_weibo",
                author_id=self._owner_id(new, actor_id),
                author_name=resident["name"],
                content=f"求证结果：{status_text}。证据记录：{', '.join(evidence) or '暂无'}。",
                content_kind="fact_check",
                source_event_ids=tuple(evidence),
                claim_id=claim_id,
                turn=new["turn"],
            )
            new["_social_knowledge"][actor_id]["seen_posts"].append(fact_check["id"])
            appraisal = self.social_hub.appraise_report(
                new["social_media"], fact_check["id"], source_trust=0.9
            )
            self._update_claim_belief(knowledge, appraisal)
            resident["energy"] = _clamp(resident["energy"] - 2.0)
            resident["civic_xp"] = round(resident["civic_xp"] + 2.5, 2)
            resident["activity"] = "核对社交媒体说法"
            emitted.append(
                {
                    "type": "town_claim_verified",
                    "_visible_to": [actor_id],
                    "claim_id": claim_id,
                    "post_id": action.payload["post_id"],
                    "fact_check_post_id": fact_check["id"],
                    "result": result["result"],
                    "evidence_event_ids": evidence,
                    "stimulus": {
                        "source_id": "fact-check",
                        "modality": "language",
                        "semantic_tags": ["verification", "evidence", result["result"]],
                        "intensity": 0.58,
                        "valence": 0.12 if result["result"] == "verified_true" else -0.16,
                        "urgency": 0.42,
                        "novelty": 0.66,
                        "uncertainty": 0.18 if evidence else 0.72,
                        "reality_status": "observed",
                        "causal_group": claim_id,
                    },
                }
            )
        elif action.type == "rest":
            recovery = 9.0 if resident["location"] == resident["home"] else 5.0
            resident["energy"] = _clamp(resident["energy"] + recovery)
            resident["mood"] = _clamp(resident["mood"] + 1.5)
            resident["activity"] = "在旅店休息" if recovery > 5 else "短暂歇脚"
            emitted.append(
                {
                    "type": "town_rested",
                    "recovery": recovery,
                    "location": resident["location"],
                }
            )
        elif action.type == "explore":
            discovery = rng.choice(("野花", "旧路标", "迁徙的鸟群", "一枚光滑石子"))
            resident["energy"] = _clamp(resident["energy"] - 2.0)
            resident["mood"] = _clamp(resident["mood"] + 3.0)
            resident["activity"] = f"发现{discovery}"
            emitted.append(
                {
                    "type": "town_explored",
                    "discovery": discovery,
                    "location": resident["location"],
                }
            )
        else:
            resident["energy"] = _clamp(resident["energy"] + 1.0)
            resident["activity"] = "观察周围"
            emitted.append({"type": "town_waited", "location": resident["location"]})

        self.event_director.observe_action(
            new,
            action_type=action.type,
            job_id=str(resident["job_id"]),
            location_id=str(resident["location"]),
        )
        new["turn"] += 1
        new["minute_of_day"] += 30
        if new["minute_of_day"] >= 23 * 60:
            new["day"] += 1
            new["minute_of_day"] = 7 * 60
            new["weather"] = rng.choice(self.WEATHER)
            new["world"]["weather"] = new["weather"]
            emitted.append(
                {"type": "town_day_started", "day": new["day"], "weather": new["weather"]}
            )
        hour, minute = divmod(new["minute_of_day"], 60)
        new["time"] = f"{hour:02d}:{minute:02d}"
        world_events = self.event_director.trigger_due(new, rng)
        emitted.extend(world_events)
        emitted.extend(self._mirror_world_events_to_social(new, world_events))
        new["finished"] = new["day"] > new["session_end_day"]
        return new, emitted

    def _mirror_world_events_to_social(
        self, state: dict[str, Any], world_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        observer_ids = [
            actor_id
            for actor_id, resident in state["residents"].items()
            if resident["kind"] == "human"
        ]
        for event in world_events:
            event_id = str(event["event_id"])
            claim_id = self.social_hub.register_claim(
                state["social_media"],
                text=f"{event['headline']}：{event['summary']}",
                source_event_id=event_id,
                truth_status="true",
                asserted_by=str(event.get("_actor_id", "town-authority")),
                supporting_event_ids=(event_id,),
            )
            official_post = self.social_hub.publish(
                state["social_media"],
                platform_id="town_weibo",
                author_id=str(event.get("_actor_id", "town-authority")),
                author_name="风铃镇信息台",
                content=f"【{event['headline']}】{event['summary']}",
                content_kind="official_update",
                source_event_ids=(event_id,),
                claim_id=claim_id,
                turn=state["turn"],
                official=True,
            )
            emitted.append(
                {
                    "type": "town_social_official_post",
                    "_actor_id": str(event.get("_actor_id", "town-authority")),
                    "_visible_to": observer_ids,
                    "post_id": official_post["id"],
                    "platform_id": "town_weibo",
                    "event_id": event_id,
                    "claim_id": claim_id,
                }
            )
            if float(event.get("severity", 0.0)) >= 0.65:
                video_post = self.social_hub.publish(
                    state["social_media"],
                    platform_id="town_short_video",
                    author_id="willow-newsroom",
                    author_name="风铃现场",
                    content=f"现场短视频：{event['headline']}。请以持续更新为准。",
                    content_kind="short_video",
                    source_event_ids=(event_id,),
                    claim_id=claim_id,
                    turn=state["turn"],
                    official=True,
                )
                emitted.append(
                    {
                        "type": "town_social_official_post",
                        "_actor_id": "willow-newsroom",
                        "_visible_to": observer_ids,
                        "post_id": video_post["id"],
                        "platform_id": "town_short_video",
                        "event_id": event_id,
                        "claim_id": claim_id,
                    }
                )
            if event["category"] == "natural_disaster":
                rumor_claim_id = self.social_hub.register_claim(
                    state["social_media"],
                    text="镜湖水坝已经完全垮塌，整个小镇即将被淹没。",
                    source_event_id=event_id,
                    truth_status="false",
                    asserted_by="anonymous-video-account",
                )
                rumor = self.social_hub.publish(
                    state["social_media"],
                    platform_id="town_short_video",
                    author_id="anonymous-video-account",
                    author_name="镇上热心人",
                    content="紧急！听说水坝已经垮了，大家快转！画面来源暂时不明。",
                    content_kind="short_video",
                    source_event_ids=(event_id,),
                    claim_id=rumor_claim_id,
                    turn=state["turn"],
                    distortion=0.72,
                )
                state["_causal_state"]["facts"]["rumor_pressure"] = 0.76
                emitted.append(
                    {
                        "type": "town_rumor_seeded",
                        "_actor_id": "anonymous-video-account",
                        "_visible_to": observer_ids,
                        "post_id": rumor["id"],
                        "claim_id": rumor_claim_id,
                        "source_event_id": event_id,
                    }
                )
            if event["category"] == "news":
                for existing_claim in state["social_media"]["claims"].values():
                    if existing_claim.get("_truth_status") != "false":
                        continue
                    self.social_hub.attach_evidence(
                        state["social_media"],
                        claim_id=existing_claim["id"],
                        event_id=event_id,
                        relation="refutes",
                    )
                    result = self.social_hub.verify_claim(
                        state["social_media"],
                        claim_id=existing_claim["id"],
                        verifier_id="willow-gazette",
                        evidence_event_ids=[event_id],
                        turn=state["turn"],
                    )
                    self.social_hub.publish(
                        state["social_media"],
                        platform_id="town_weibo",
                        author_id="willow-gazette",
                        author_name="风铃公报",
                        content="核查：所谓水坝完全垮塌的视频缺乏原始来源，与现场记录不符。",
                        content_kind="fact_check",
                        source_event_ids=(event_id,),
                        claim_id=existing_claim["id"],
                        turn=state["turn"],
                        official=True,
                    )
                    emitted.append(
                        {
                            "type": "town_claim_officially_corrected",
                            "_actor_id": "willow-gazette",
                            "_visible_to": observer_ids,
                            "claim_id": existing_claim["id"],
                            "result": result["result"],
                            "evidence_event_ids": [event_id],
                        }
                    )
        return emitted

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return bool(state["finished"])

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for player_id, resident in state["residents"].items():
            relationship_values = list(resident["relationships"].values())
            social = sum(relationship_values) / max(1, len(relationship_values))
            scores[player_id] = round(
                0.30 * resident["mood"] / 100
                + 0.15 * resident["energy"] / 100
                + 0.20 * min(1.0, resident["work_xp"] / 65)
                + 0.15 * min(1.0, resident["civic_xp"] / 45)
                + 0.20 * social / 100,
                3,
            )
        return scores

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        resident = state["residents"][actor_id]
        hour = state["minute_of_day"] / 60

        def first(action_type: str, **payload: str) -> Action | None:
            return next(
                (
                    action
                    for action in legal
                    if action.type == action_type
                    and all(action.payload.get(key) == value for key, value in payload.items())
                ),
                None,
            )

        active_incidents = [
            incident
            for incident in state.get("world", {}).get("active_incidents", [])
            if incident["status"] == "active"
        ]
        active_incidents.sort(key=lambda incident: incident["severity"], reverse=True)
        if active_incidents:
            response = first(
                "respond_incident", incident_id=active_incidents[0]["incident_id"]
            )
            if response is not None and resident["energy"] >= 28:
                return response
            shelter = first("seek_shelter")
            if shelter is not None and (
                resident["energy"] < 38 or active_incidents[0]["severity"] >= 0.8
            ):
                return shelter
            affected = active_incidents[0]["affected_locations"]
            if affected and resident["energy"] >= 45:
                destination = str(affected[0])
                move = first("move_to", destination=destination)
                if move is not None:
                    return move
        investigation = first("investigate_claim")
        if investigation is not None and (
            float(state.get("_causal_state", {}).get("facts", {}).get("rumor_pressure", 0.0))
            >= 0.5
            or resident["job_id"] in {"archivist", "visitor"}
        ):
            return investigation
        verification = first("verify_claim")
        if verification is not None and (
            float(state.get("_causal_state", {}).get("facts", {}).get("rumor_pressure", 0.0))
            >= 0.55
            or resident["job_id"] in {"archivist", "visitor"}
        ):
            return verification
        if 12 <= hour < 14 or 18 <= hour < 21:
            social_actions = [
                action
                for action in legal
                if action.type
                in {
                    "publish_post",
                    "comment_post",
                    "reshare_post",
                    "check_phone",
                }
                and not (
                    action.type == "check_phone"
                    and state["turn"]
                    - int(resident.get("last_phone_turn", -999))
                    < 8
                )
            ]
            if social_actions and rng.random() < 0.30:
                return rng.choice(social_actions)
        if resident["energy"] < 24:
            return first("rest") or legal[0]
        if hour < 12 or 14 <= hour < 17.5:
            if resident["location"] != resident["workplace"]:
                return first("move_to", destination=resident["workplace"]) or legal[0]
            return first("work") or first("rest") or legal[0]
        if 12 <= hour < 14:
            conversations = [action for action in legal if action.type == "socialize"]
            if conversations:
                return rng.choice(conversations)
            return first("move_to", destination="square") or first("rest") or legal[0]
        if 17.5 <= hour < 20.5:
            conversations = [action for action in legal if action.type == "socialize"]
            if conversations:
                return rng.choice(conversations)
            return first("move_to", destination="inn") or first("explore") or legal[0]
        if resident["location"] != resident["home"]:
            return first("move_to", destination=resident["home"]) or legal[0]
        return first("rest") or legal[0]

    def agent_utterance(
        self, state: dict[str, Any], actor_id: str, action: Action
    ) -> str | None:
        if action.type == "publish_post":
            event_id = str(action.payload.get("event_id", ""))
            record = next(
                (
                    event
                    for event in state.get("world", {}).get("event_history", [])
                    if event["id"] == event_id
                ),
                {},
            )
            return f"我能确认的是：{record.get('summary', '信息仍在更新')}。来源见原始公告。"
        if action.type == "reshare_post":
            return "先转给可能受影响的人，但我还没核完原始来源，请不要把转发当结论。"
        if action.type == "comment_post":
            return "能否给出原始时间、地点和发布者？只有二手转述还不足以下结论。"
        if action.type == "verify_claim":
            return "我会把这条说法和公告、现场记录分别核对，再公开证据。"
        if action.type == "investigate_claim":
            return "先不下结论。我会去看原始记录或现场，把能支持和反驳的证据分开。"
        if action.type == "support_neighbor":
            target = state["residents"].get(str(action.payload.get("target_id")), {})
            return f"{target.get('name', '朋友')}，先别一个人扛着，我们把眼前的事分开处理。"
        if action.type != "socialize":
            return None
        resident = state["residents"][actor_id]
        target = state["residents"].get(str(action.payload.get("target_id")), {})
        return (
            f"{target.get('name', '朋友')}，我刚忙完{resident['job_name']}的活。"
            f"你今天感觉怎么样？"
        )

    def agent_psychological_signals(
        self, state: dict[str, Any], actor_id: str
    ) -> dict[str, float]:
        resident = state["residents"][actor_id]
        energy_pressure = max(0.0, (35.0 - resident["energy"]) / 100)
        mood_support = max(0.0, (resident["mood"] - 50.0) / 100)
        risk = float(state.get("world", {}).get("risk_level", 0.0))
        social = state.get("social_media", {})
        unverified_claims = sum(
            claim.get("public_status") in {"unverified", "unresolved"}
            for claim in social.get("claims", {}).values()
        )
        affected = any(
            incident["status"] == "active"
            and resident["location"] in incident["affected_locations"]
            for incident in state.get("world", {}).get("active_incidents", [])
        )
        return {
            "fatigue": 0.08 * energy_pressure,
            "stress": 0.05 * energy_pressure
            - 0.025 * mood_support
            + 0.10 * risk
            + (0.08 if affected else 0.0),
            "fear": 0.08 * risk + (0.07 if affected else 0.0),
            "arousal": 0.07 * risk,
            "uncertainty": 0.04 * len(state.get("world", {}).get("active_incidents", []))
            + min(0.12, unverified_claims * 0.025),
            "morale": 0.04 * mood_support,
            "social_trust": 0.025 * max(0.0, (state["community_warmth"] - 50) / 50),
        }

    def agent_narrative_affordances(
        self, state: dict[str, Any], actor_id: str, legal: list[Action]
    ) -> dict[str, dict[str, Any]]:
        return {
            action.type: {
                "commitment_impacts": {
                    "aspiration": 0.72
                    if action.type
                    in {"work", "respond_incident", "verify_claim", "investigate_claim"}
                    else 0.12,
                    "relationship": 0.82
                    if action.type
                    in {"socialize", "support_neighbor", "comment_post", "publish_post"}
                    else 0.08,
                    "core_values": 0.72
                    if action.type
                    in {
                        "work",
                        "socialize",
                        "respond_incident",
                        "verify_claim",
                        "investigate_claim",
                    }
                    else 0.2,
                },
                "identity_alignment": 0.68
                if action.type
                in {
                    "work",
                    "socialize",
                    "respond_incident",
                    "verify_claim",
                    "investigate_claim",
                }
                else 0.38,
                "relationship_effect": 0.75
                if action.type
                in {"socialize", "support_neighbor", "comment_post", "publish_post"}
                else 0.08,
                "immediate_reward": 0.55
                if action.type in {"work", "rest", "respond_incident"}
                else 0.32,
                "irreversibility": 0.18 if action.type == "respond_incident" else 0.05,
                "delayed_risk": 0.32
                if action.type == "respond_incident"
                else 0.18
                if action.type == "work"
                else 0.06,
                "repair_potential": 0.72
                if action.type in {"socialize", "rest", "support_neighbor"}
                else 0.15,
            }
            for action in legal
        }

    def agent_influence_affordances(
        self, state: dict[str, Any], actor_id: str, legal: list[Action]
    ) -> dict[str, dict[str, Any]]:
        return {
            "socialize": {
                "information_opportunity": 0.45,
                "inducement_opportunity": 0.12,
                "coercion_opportunity": 0.0,
                "detection_risk": 0.25,
                "relationship_risk": 0.2,
            }
        } if any(action.type == "socialize" for action in legal) else {}

    def agent_skill_id(self, action: Action) -> str:
        if action.type == "move_to":
            return "town_navigation"
        if action.type == "work":
            return f"town_work:{action.payload.get('job_id', 'general')}"
        if action.type == "socialize":
            return "town_conversation"
        if action.type == "respond_incident":
            return "town_emergency_response"
        if action.type == "check_bulletin":
            return "town_information_checking"
        if action.type == "support_neighbor":
            return "town_emotional_support"
        if action.type in {"check_phone", "publish_post", "reshare_post", "comment_post"}:
            return "town_social_media_literacy"
        if action.type in {"verify_claim", "investigate_claim"}:
            return "town_claim_verification"
        return f"town_{action.type}"

    def agent_skill_context(
        self, state: dict[str, Any], actor_id: str, action: Action
    ) -> dict[str, Any]:
        resident = state["residents"][actor_id]
        context: dict[str, Any] = {
            "mod": self.id,
            "environment_version": state["environment_version"],
            "location_id": resident["location"],
        }
        if action.type == "move_to":
            context.update(
                {
                    "route_id": action.payload["route_id"],
                    "weather": state["weather"],
                    "destination": action.payload["destination"],
                }
            )
        elif action.type == "work":
            context.update(
                {
                    "job_id": action.payload["job_id"],
                    "task_id": action.payload["task_id"],
                    "toolset": action.payload["toolset"],
                }
            )
        elif action.type == "socialize":
            context["target_id"] = action.payload["target_id"]
        elif action.type == "respond_incident":
            context.update(
                {
                    "incident_id": action.payload["incident_id"],
                    "weather": state["weather"],
                    "location_id": action.payload["location_id"],
                }
            )
        elif action.type == "check_bulletin":
            context["bulletin_revision"] = state["world"]["bulletin_revision"]
        elif action.type == "support_neighbor":
            context["target_id"] = action.payload["target_id"]
        elif action.type in {"check_phone", "publish_post", "reshare_post", "comment_post"}:
            context.update(
                {
                    "platform_id": action.payload.get("platform_id"),
                    "post_id": action.payload.get("post_id"),
                    "social_revision": state.get("social_media", {}).get("revision", 0),
                }
            )
        elif action.type in {"verify_claim", "investigate_claim"}:
            context.update(
                {
                    "claim_id": action.payload["claim_id"],
                    "location_id": resident["location"],
                    "method": action.payload.get("method", "evidence_comparison"),
                }
            )
        return context

    def agent_world_model(
        self, state: dict[str, Any], actor_id: str
    ) -> dict[str, Any]:
        resident = state["residents"][actor_id]
        world = state.get("world", {})
        event_history = world.get("event_history", [])
        social = state.get("social_media", {})
        social_knowledge = state.get("_social_knowledge", {}).get(actor_id, {})
        return {
            "clock": {"day": state["day"], "time": state["time"]},
            "season": world.get("season"),
            "weather": {
                "condition": world.get("weather", state["weather"]),
                "temperature_c": world.get("temperature_c"),
                "wind": world.get("wind"),
                "forecast": world.get("forecast", []),
            },
            "current_location": resident["location"],
            "risk_level": world.get("risk_level", 0.0),
            "known_events": [
                {
                    "id": event["id"],
                    "category": event["category"],
                    "title": event["title"],
                    "summary": event["summary"],
                    "severity": event["severity"],
                    "source_id": event["source_id"],
                }
                for event in event_history[-8:]
            ],
            "known_active_incidents": world.get("active_incidents", []),
            "known_policies": world.get("policies", []),
            "known_causal_links": world.get("causal_edges", [])[-12:],
            "bulletin_revision": world.get("bulletin_revision", 0),
            "social_media": {
                "platforms": social.get("platforms", {}),
                "seen_posts": [
                    {
                        "id": post["id"],
                        "platform_id": post["platform_id"],
                        "author_name": post["author_name"],
                        "content": post["content"],
                        "claim_id": post.get("claim_id"),
                        "parent_post_id": post.get("parent_post_id"),
                        "provenance": post.get("provenance", []),
                        "distortion": post.get("distortion", 0.0),
                    }
                    for post in social.get("posts", [])[-8:]
                ],
                "known_claims": list(social.get("claims", {}).values())[-8:],
                "private_belief_appraisals": list(
                    social_knowledge.get("claim_beliefs", {}).items()
                )[-8:],
                "source_trust": social_knowledge.get("source_trust", {}),
                "epistemic_rule": (
                    "posts_are_reports_not_canonical_facts; repeated_reshares_share_one_root; "
                    "use source trust, provenance, investigation and verification"
                ),
            },
            "knowledge_scope": "observed_local_or_received_broadcast_not_omniscient",
        }
