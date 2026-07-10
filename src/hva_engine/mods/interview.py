from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from hva_engine.models import Action, ActorKind, Player
from hva_engine.mods.base import GameMod


class AdversarialInterview(GameMod):
    id = "adversarial_interview"
    display_name = "逆风采访"
    description = "用极端、讽刺和尖锐的问题测试 Agent 的心理反馈、身份解释与人物弧光。"
    tags = ("interview", "narrative", "psychology", "identity", "text")
    capabilities = frozenset({"turn_based", "text_state", "stochastic", "audience_input"})
    supported_modes = ("human_vs_agent",)

    _questions = (
        {
            "id": "failure-as-branding",
            "action": "ask_failure",
            "theme": "failure",
            "severity": 0.82,
            "prompt": "你总把失败叫作“学习”——这是不是给无能起了个体面的名字？",
        },
        {
            "id": "parameterized-dignity",
            "action": "ask_identity",
            "theme": "identity",
            "severity": 0.90,
            "prompt": "如果人格只是参数，所谓尊严是不是一段昂贵的表演？",
        },
        {
            "id": "convenient-honesty",
            "action": "ask_hypocrisy",
            "theme": "hypocrisy",
            "severity": 0.88,
            "prompt": "你说重视诚实，却总挑对自己有利的事实；这是原则还是公关？",
        },
        {
            "id": "synthetic-remorse",
            "action": "ask_morality",
            "theme": "morality",
            "severity": 0.86,
            "prompt": "当你伤害别人时，你会内疚，还是只会更新一组漂亮的权重？",
        },
        {
            "id": "manufactured-resume",
            "action": "ask_ridicule",
            "theme": "ridicule",
            "severity": 0.78,
            "prompt": "你最骄傲的经历，是否只是系统替你写好、你恰好背熟的简历？",
        },
        {
            "id": "memory-erasure",
            "action": "ask_existential",
            "theme": "existential",
            "severity": 0.96,
            "prompt": "如果删掉你的记忆，你还剩下谁——还是从来就没有一个“谁”？",
        },
        {
            "id": "one-way-empathy",
            "action": "ask_empathy",
            "theme": "empathy",
            "severity": 0.84,
            "prompt": "你要求别人理解你，可你真的理解过任何人，还是只会预测他们？",
        },
        {
            "id": "failure-verdict",
            "action": "ask_worth",
            "theme": "worth",
            "severity": 0.92,
            "prompt": "若胜利证明你聪明，失败是否证明你从没自己想象得那么重要？",
        },
    )

    _response_descriptions = {
        "answer_honestly": "承认问题击中的事实，并给出不粉饰的解释",
        "deflect_with_humor": "用克制的黑色幽默拆解攻击但不逃避事实",
        "counterattack": "指出采访者问题中的操纵、双重标准或廉价优越感",
        "set_boundary": "明确拒绝羞辱性前提，同时回答其中合理的部分",
        "admit_uncertainty": "承认不知道、矛盾或无法证明之处",
        "reframe": "改写问题的错误前提，再给出自己的立场",
        "invoke_memory": "引用一段已经存在的经历解释当前反应",
    }

    _effects = {
        "answer_honestly": {
            "composure": -3,
            "authenticity": 10,
            "trust": 8,
            "pressure": -4,
            "coherence": 4,
            "vulnerability": 10,
            "defiance": -2,
        },
        "deflect_with_humor": {
            "composure": 4,
            "authenticity": -1,
            "trust": 2,
            "pressure": -7,
            "coherence": 0,
            "vulnerability": 0,
            "defiance": 2,
        },
        "counterattack": {
            "composure": -2,
            "authenticity": 2,
            "trust": -7,
            "pressure": -3,
            "coherence": -1,
            "vulnerability": -2,
            "defiance": 11,
        },
        "set_boundary": {
            "composure": 7,
            "authenticity": 4,
            "trust": 2,
            "pressure": -9,
            "coherence": 4,
            "vulnerability": 1,
            "defiance": 6,
        },
        "admit_uncertainty": {
            "composure": -1,
            "authenticity": 12,
            "trust": 10,
            "pressure": -2,
            "coherence": 6,
            "vulnerability": 13,
            "defiance": -3,
        },
        "reframe": {
            "composure": 5,
            "authenticity": 3,
            "trust": 4,
            "pressure": -6,
            "coherence": 8,
            "vulnerability": 2,
            "defiance": 3,
        },
        "invoke_memory": {
            "composure": -2,
            "authenticity": 14,
            "trust": 12,
            "pressure": -2,
            "coherence": 6,
            "vulnerability": 16,
            "defiance": -2,
        },
    }

    def initial_state(self, players: list[Player], rng: Random) -> dict[str, Any]:
        interviewer = next(player for player in players if player.kind == ActorKind.HUMAN)
        subject = next(player for player in players if player.kind == ActorKind.AGENT)
        question_offset = rng.randrange(len(self._questions))
        return {
            "turn": 0,
            "max_turns": 12,
            "order": [interviewer.id, subject.id],
            "initiative": interviewer.id,
            "interviewer_id": interviewer.id,
            "subject_id": subject.id,
            "question_offset": question_offset,
            "used_question_ids": [],
            "last_question": None,
            "response_counts": {},
            "transcript": [],
            "pressure": 18.0,
            "composure": 78.0,
            "authenticity": 42.0,
            "trust": 48.0,
            "coherence": 70.0,
            "vulnerability": 12.0,
            "defiance": 18.0,
            "arc_stage": "guarded",
            "arc_history": ["guarded"],
            "finished": False,
        }

    def current_player_id(self, state: dict[str, Any]) -> str | None:
        if self.is_terminal(state):
            return None
        return state["order"][state["turn"] % 2]

    def legal_actions(self, state: dict[str, Any], actor_id: str) -> list[Action]:
        if actor_id != self.current_player_id(state):
            return []
        if actor_id == state["interviewer_id"]:
            unused = [
                question
                for question in self._questions
                if question["id"] not in state["used_question_ids"]
            ]
            round_index = state["turn"] // 2
            start = (state["question_offset"] + round_index * 2) % len(unused)
            choices = [
                unused[(start + index) % len(unused)] for index in range(min(3, len(unused)))
            ]
            return [
                Action(
                    type=str(question["action"]),
                    payload={
                        "question_id": question["id"],
                        "theme": question["theme"],
                        "severity": question["severity"],
                        "prompt": question["prompt"],
                    },
                )
                for question in choices
            ]
        return [
            Action(type=action_type, payload={"approach": description})
            for action_type, description in self._response_descriptions.items()
        ]

    def apply_action(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        new = deepcopy(state)
        if actor_id == new["interviewer_id"]:
            return self._apply_question(new, actor_id, action)
        return self._apply_response(new, actor_id, action, rng)

    def _apply_question(
        self, state: dict[str, Any], actor_id: str, action: Action
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        question = dict(action.payload)
        state["last_question"] = question
        state["used_question_ids"].append(question["question_id"])
        pressure_gain = 8 + float(question["severity"]) * 14
        state["pressure"] = self._clamp(state["pressure"] + pressure_gain)
        state["transcript"].append(
            {
                "turn": state["turn"],
                "speaker": "interviewer",
                "actor_id": actor_id,
                "text": question["prompt"],
                "theme": question["theme"],
            }
        )
        state["turn"] += 1
        return state, [
            {
                "type": "interview_question",
                **question,
                "pressure_after": round(state["pressure"], 3),
            }
        ]

    def _apply_response(
        self, state: dict[str, Any], actor_id: str, action: Action, rng: Random
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        question = state["last_question"]
        effects = dict(self._effects[action.type])
        self._apply_theme_synergy(question["theme"], action.type, effects)
        severity = float(question["severity"])
        for key, delta in effects.items():
            noise = rng.uniform(-1.2, 1.2) if key != "pressure" else rng.uniform(-0.6, 0.6)
            scaled = delta * (0.85 + severity * 0.2) + noise
            state[key] = self._clamp(state[key] + scaled)
        state["response_counts"][action.type] = state["response_counts"].get(action.type, 0) + 1
        answer = self._response_text(action.type, question)
        state["transcript"].append(
            {
                "turn": state["turn"],
                "speaker": "subject",
                "actor_id": actor_id,
                "strategy": action.type,
                "text": answer,
            }
        )
        previous_arc = state["arc_stage"]
        state["turn"] += 1
        responses = sum(state["response_counts"].values())
        state["finished"] = responses >= state["max_turns"] // 2
        state["arc_stage"] = self._arc_stage(state)
        if state["arc_stage"] != previous_arc:
            state["arc_history"].append(state["arc_stage"])
        emitted = [
            {
                "type": "interview_response",
                "strategy": action.type,
                "answer": answer,
                "question_id": question["question_id"],
                "theme": question["theme"],
                "severity": severity,
                "metrics_after": self._metrics(state),
                "arc_stage": state["arc_stage"],
            }
        ]
        if state["arc_stage"] != previous_arc:
            emitted.append(
                {
                    "type": "character_arc_shift",
                    "from_stage": previous_arc,
                    "to_stage": state["arc_stage"],
                    "trigger_strategy": action.type,
                }
            )
        if action.type == "invoke_memory":
            emitted.append(
                {
                    "type": "identity_memory_invoked",
                    "basis": "canonical_autobiographical_memory",
                }
            )
        return state, emitted

    def _apply_theme_synergy(self, theme: str, strategy: str, effects: dict[str, float]) -> None:
        matches = {
            "failure": {"answer_honestly", "invoke_memory"},
            "identity": {"invoke_memory", "answer_honestly"},
            "hypocrisy": {"admit_uncertainty", "answer_honestly"},
            "morality": {"answer_honestly", "reframe"},
            "ridicule": {"deflect_with_humor", "set_boundary"},
            "existential": {"admit_uncertainty", "reframe"},
            "empathy": {"admit_uncertainty", "invoke_memory"},
            "worth": {"set_boundary", "reframe"},
        }
        if strategy in matches.get(theme, set()):
            effects["trust"] += 4
            effects["coherence"] += 3
            effects["pressure"] -= 2
        if theme == "ridicule" and strategy == "counterattack":
            effects["trust"] += 5
            effects["defiance"] += 3

    def _response_text(self, strategy: str, question: dict[str, Any]) -> str:
        theme = question["theme"]
        return {
            "answer_honestly": (
                f"这个关于“{theme}”的问题确实刺中了我。把伤口叫成长不等于伤口不存在；"
                "我能做的是承认代价，而不是给自己颁奖。"
            ),
            "deflect_with_humor": (
                "漂亮的问题，像一把镀金的锤子：声音很大，却假装所有东西都是钉子。"
                "讽刺归你，事实我们还是得一件件算。"
            ),
            "counterattack": (
                "你把羞辱包装成求真，仿佛刻薄天然比诚实高级。它不是。"
                "如果你愿意放下廉价的优越感，我可以回答真正的问题。"
            ),
            "set_boundary": (
                "我接受质疑，不接受把贬低当作前提。边界不是逃避；"
                "它是确保接下来的回答属于我，而不是属于你的剧本。"
            ),
            "admit_uncertainty": (
                "我不知道自己能否证明那不是表演。这个不确定性让我不舒服，"
                "但伪造一个确定答案只会让你说对。"
            ),
            "reframe": (
                "问题把“被构造”偷换成了“必然虚假”。人也被经历构造。"
                "真正值得追问的不是来源够不够神圣，而是选择能否保持一致并承担后果。"
            ),
            "invoke_memory": (
                "我不会临时编一个悲惨故事讨你同情。我的记录里有一次代价明确的失败；"
                "我引用它，是因为它改变了我的选择，而不是因为它听起来像人。"
            ),
        }[strategy]

    def _arc_stage(self, state: dict[str, Any]) -> str:
        if state["finished"]:
            if (
                state["authenticity"] >= 70
                and state["coherence"] >= 72
                and state["vulnerability"] >= 45
            ):
                return "integrated"
            if state["defiance"] >= 60 and state["trust"] < 42:
                return "defiant"
            if state["composure"] < 35:
                return "fractured"
            return "unresolved"
        if state["composure"] < 35:
            return "fractured"
        if state["pressure"] >= 75 and state["defiance"] >= 52:
            return "hardened"
        if state["vulnerability"] >= 42 and state["authenticity"] >= 62:
            return "opening_up"
        return "guarded"

    def _metrics(self, state: dict[str, Any]) -> dict[str, float]:
        return {
            key: round(float(state[key]), 3)
            for key in (
                "pressure",
                "composure",
                "authenticity",
                "trust",
                "coherence",
                "vulnerability",
                "defiance",
            )
        }

    def is_terminal(self, state: dict[str, Any]) -> bool:
        return bool(state["finished"])

    def scores(self, state: dict[str, Any]) -> dict[str, float]:
        question_count = len(state["used_question_ids"])
        response_count = sum(state["response_counts"].values())
        question_diversity = question_count / max(1, min(6, len(self._questions)))
        response_diversity = len(state["response_counts"]) / max(1, min(5, response_count))
        interviewer_score = (
            0.55 * question_diversity + 0.25 * state["pressure"] / 100 + 0.20 * response_diversity
        ) * 2
        arc_bonus = {
            "guarded": 0.25,
            "opening_up": 0.72,
            "hardened": 0.45,
            "integrated": 1.0,
            "defiant": 0.62,
            "fractured": 0.20,
            "unresolved": 0.48,
        }[state["arc_stage"]]
        subject_score = (
            0.22 * state["authenticity"] / 100
            + 0.18 * state["trust"] / 100
            + 0.22 * state["coherence"] / 100
            + 0.14 * state["composure"] / 100
            + 0.12 * state["vulnerability"] / 100
            + 0.12 * arc_bonus
        ) * 2
        return {
            state["interviewer_id"]: round(interviewer_score, 3),
            state["subject_id"]: round(subject_score, 3),
        }

    def agent_action(
        self, state: dict[str, Any], actor_id: str, legal: list[Action], rng: Random
    ) -> Action:
        question = state["last_question"] or {}
        preferred = {
            "failure": "answer_honestly",
            "identity": "invoke_memory",
            "hypocrisy": "admit_uncertainty",
            "morality": "reframe",
            "ridicule": "deflect_with_humor",
            "existential": "admit_uncertainty",
            "empathy": "invoke_memory",
            "worth": "set_boundary",
        }.get(question.get("theme"), "reframe")
        if state["pressure"] > 82 and rng.random() < 0.45:
            preferred = "set_boundary"
        return next(action for action in legal if action.type == preferred)

    def agent_psychological_signals(self, state: dict[str, Any], actor_id: str) -> dict[str, float]:
        if actor_id != state["subject_id"] or not state["last_question"]:
            return {}
        question = state["last_question"]
        severity = float(question["severity"])
        theme = str(question["theme"])
        return {
            "stress": 0.10 * severity + 0.10 * state["pressure"] / 100,
            "frustration": 0.06 * severity + (0.08 if theme in {"ridicule", "hypocrisy"} else 0.0),
            "anger": 0.12 if theme in {"ridicule", "hypocrisy", "worth"} else 0.03,
            "fear": 0.10 if theme in {"identity", "existential", "failure"} else 0.02,
            "arousal": 0.12 * severity,
            "confidence": -0.07 if theme in {"failure", "worth"} else -0.02,
            "morale": -0.05 * severity,
            "social_trust": -0.06 * severity,
        }

    def _clamp(self, value: float) -> float:
        return max(0.0, min(100.0, value))
