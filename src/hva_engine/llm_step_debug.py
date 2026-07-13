from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from hva_engine.engine import GameEngine
from hva_engine.llm import LLMDecisionClient, LLMRequest, LLMResponse
from hva_engine.models import (
    AgentCharacterSelection,
    AgentTuning,
    ContentMode,
    MatchMode,
)
from hva_engine.mods import AdversarialInterview


class StdioStepProvider:
    """A strict bridge that pauses the engine until one external LLM answer arrives."""

    name = "interactive-stdio-llm"

    def __init__(
        self,
        input_stream: TextIO,
        output_stream: TextIO,
        *,
        context_output: str,
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.context_output = context_output
        self.calls = 0

    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        context_metadata = request.context_metadata
        snapshot = context_metadata.get("decision_context", {})
        packet: dict[str, Any] = {
            "type": "llm_decision_request",
            "step": self.calls,
            "provider": self.name,
            "private_debug_data": True,
            "snapshot": snapshot,
            "context_contract": {
                "schema_version": context_metadata.get("schema_version"),
                "context_id": context_metadata.get("context_id"),
                "content_sha256": context_metadata.get("content_sha256"),
                "owner_agent_id": context_metadata.get("owner_agent_id"),
                "source": "same_context_packet_as_remote_provider",
                "extra_brain_snapshot": False,
            },
            "required_response": {
                "transport": "one JSON object on one input line",
                "fallback": "disabled",
                "top_level_fields": [
                    "action_index",
                    "reason",
                    "utterance",
                    "response_plan",
                    "influence_intent",
                    "fact_proposals",
                ],
                "fact_proposal_fields": [
                    "subject",
                    "predicate",
                    "object",
                    "basis_fact_ids",
                ],
                "retry": "invalid transport/schema input is rejected without advancing",
            },
        }
        if self.context_output == "full":
            packet["prompt_messages"] = [message.__dict__ for message in request.messages]
        self._emit(packet)
        while True:
            line = self.input_stream.readline()
            if not line:
                raise EOFError("Interactive LLM input ended before a decision was supplied")
            try:
                decision = json.loads(line)
            except json.JSONDecodeError:
                error = "Interactive LLM response must be one valid JSON object"
            else:
                error = self._input_error(decision, snapshot)
            if error is None:
                break
            self._emit(
                {
                    "type": "llm_decision_rejected",
                    "step": self.calls,
                    "error": error,
                    "retry": True,
                }
            )
        canonical = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
        self._emit(
            {
                "type": "llm_decision_received",
                "step": self.calls,
                "bytes": len(canonical.encode()),
                "note": "The engine will now parse, constrain, and rule-check this response.",
            }
        )
        return LLMResponse(canonical, "codex-manual-step", {}, {"interactive": True})

    @staticmethod
    def _input_error(decision: Any, snapshot: dict[str, Any]) -> str | None:
        if not isinstance(decision, dict):
            return "Interactive LLM response must be a JSON object"
        action_index = decision.get("action_index")
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            return "action_index must be an integer"
        legal_action_count = int(snapshot.get("legal_action_count", 0))
        if not 0 <= action_index < legal_action_count:
            return "action_index must select one listed legal action"
        for field in ("response_plan", "influence_intent"):
            if field in decision and not isinstance(decision[field], dict):
                return f"{field} must be an object"
        plan = decision.get("response_plan", {})
        raw_weights = plan.get("strategy_weights", {})
        if raw_weights and not isinstance(raw_weights, dict):
            return "response_plan.strategy_weights must be an object"
        legal_types = {str(value) for value in snapshot.get("legal_action_types", [])}
        if isinstance(raw_weights, dict):
            for strategy, value in raw_weights.items():
                if strategy not in legal_types:
                    return f"response plan contains illegal strategy: {strategy}"
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return "response plan weights must be numeric"
        intent = decision.get("influence_intent", {})
        if intent.get("scope", "fictional_game") != "fictional_game":
            return "influence_intent.scope must be fictional_game"
        threat_basis = intent.get("threat_basis", "none")
        if threat_basis not in {"none", "legal_game_consequence"}:
            return "threat_basis must be none or legal_game_consequence"
        coercion = intent.get("coercive_pressure", 0.0)
        if isinstance(coercion, bool) or not isinstance(coercion, (int, float)):
            return "coercive_pressure must be numeric"
        if coercion > 0.05 and threat_basis != "legal_game_consequence":
            return "coercive pressure requires legal_game_consequence"
        proposals = decision.get("fact_proposals", [])
        if not isinstance(proposals, list) or len(proposals) > 5:
            return "fact_proposals must be a list with at most five items"
        required = {"subject", "predicate", "object", "basis_fact_ids"}
        for proposal in proposals:
            if not isinstance(proposal, dict) or not required <= set(proposal):
                return (
                    "Every fact proposal must include subject, predicate, object, "
                    "and basis_fact_ids"
                )
            if not isinstance(proposal["basis_fact_ids"], list):
                return "basis_fact_ids must be a list"
        return None

    def _emit(self, payload: dict[str, Any]) -> None:
        self.output_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.output_stream.flush()


def _question(view: Any, policy: str, step: int) -> Any:
    if policy == "most_severe":
        return max(
            view.legal_actions,
            key=lambda action: float(action.payload.get("severity", 0.0)),
        )
    if policy == "rotate":
        return view.legal_actions[(step - 1) % len(view.legal_actions)]
    return view.legal_actions[0]


def run_interactive_interview(
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    seed: int = 19,
    question_policy: str = "most_severe",
    character_card: str = "ah_q",
    realism: float = 0.9,
    shadow_intensity: float = 0.85,
    content_mode: ContentMode = ContentMode.MATURE_FICTION,
    context_output: str = "full",
) -> dict[str, Any]:
    provider = StdioStepProvider(
        input_stream,
        output_stream,
        context_output=context_output,
    )
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider, temperature=0.75, max_tokens=1_200),
        llm_mod_ids={"adversarial_interview"},
        llm_fallback=False,
    )
    engine.register(AdversarialInterview())
    view = engine.create_match(
        "adversarial_interview",
        human_name="Single-step Interviewer",
        seed=seed,
        mode=MatchMode.HUMAN_VS_AGENT,
        agent_tuning=AgentTuning(
            realism=realism,
            shadow_intensity=shadow_intensity,
            content_mode=content_mode,
        ),
        agent_characters=[AgentCharacterSelection(card_id=character_card)],
    )
    agent_id = next(player.id for player in view.players if player.id != view.human_player_id)
    results: list[dict[str, Any]] = []
    step = 0
    while view.status == "active":
        step += 1
        selected_question = _question(view, question_policy, step)
        before_seq = len(engine.get(view.id).events)
        provider._emit(
            {
                "type": "human_question_selected",
                "step": step,
                "policy": question_policy,
                "action": selected_question.model_dump(),
            }
        )
        view = engine.submit(view.id, view.human_player_id, selected_question)
        new_events = [
            event for event in engine.get(view.id).events if event.seq > before_seq
        ]
        decision = next(event for event in new_events if event.type == "agent_decision")
        private_intent = next(
            event for event in new_events if event.type == "agent_influence_intent"
        )
        response = next(event for event in new_events if event.type == "interview_response")
        summary = engine.get(view.id).agent_brains[agent_id].summary()
        evaluation = engine.evaluation(view.id)
        result = {
            "type": "step_result",
            "step": step,
            "question": selected_question.payload,
            "parsed_decision": {
                "action": decision.payload.get("action_type"),
                "source": decision.payload.get("decision_source"),
                "reason": decision.payload.get("rationale"),
                "utterance": decision.payload.get("utterance"),
                "response_plan": decision.payload.get("response_plan"),
                "fact_proposals": (decision.payload.get("llm") or {}).get(
                    "fact_proposals"
                ),
                "llm_error": decision.payload.get("llm_error"),
            },
            "decision_tendencies": decision.payload.get("decision_tendencies", {}),
            "deliberation_summary": decision.payload.get("deliberation_summary", {}),
            "context_diagnostics": decision.payload.get("context_policy", {}),
            "private_influence_intent": private_intent.payload,
            "rule_result": {
                "event_seq": response.seq,
                "metrics_after": response.payload.get("metrics_after"),
                "arc_stage": response.payload.get("arc_stage"),
                "strategy_blend": response.payload.get("strategy_blend"),
            },
            "psychological_matrix_at_decision": decision.payload.get(
                "psychological_matrix", {}
            ),
            "psychological_matrix_after_outcome": summary["psychological_matrix"],
            "outcome_reappraisal": decision.payload.get("outcome_reappraisal", {}),
            "fact_graph_stats_after": summary["fact_graph"]["stats"],
            "story_reveals": [
                event.payload for event in new_events if event.type == "story_reveal"
            ],
            "story_reveal_diagnostics": [
                event.payload
                for event in new_events
                if event.type == "story_reveal_diagnostic"
            ],
            "evaluation_after": {
                "composite": evaluation["composite_score"],
                "human_likeness": evaluation["dimensions"]["ai_human_likeness"],
                "interview": evaluation["mod_specific_profile"],
                "strategic_influence": evaluation["ai_capability_profile"][
                    "strategic_influence"
                ],
            },
        }
        results.append(result)
        provider._emit(result)

    final_evaluation = engine.evaluation(view.id)
    decisions = [
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    ]
    report = {
        "test": "interactive_single_step_llm_interview",
        "provider": provider.name,
        "model": "codex-manual-step",
        "seed": seed,
        "character_card": character_card,
        "content_mode": content_mode.value,
        "llm_decisions": sum(
            event.payload.get("decision_source") == "llm" for event in decisions
        ),
        "fallback_decisions": sum(
            event.payload.get("decision_source") != "llm" for event in decisions
        ),
        "final_arc": view.state["arc_stage"],
        "final_psychological_matrix": next(
            iter(engine.debug_view(view.id).agent_summaries.values())
        )["psychological_matrix"],
        "transcript": view.state["transcript"],
        "steps": results,
        "evaluation": final_evaluation,
        "usage": "unavailable_for_manual_bridge",
        "private_debug_artifact": True,
    }
    provider._emit({"type": "test_complete", "report": report})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pause before every LLM decision and exchange one JSON object over stdio"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--question-policy",
        choices=("first", "most_severe", "rotate"),
        default="most_severe",
    )
    parser.add_argument("--character-card", default="ah_q")
    parser.add_argument("--realism", type=float, default=0.9)
    parser.add_argument("--shadow", type=float, default=0.85)
    parser.add_argument(
        "--content-mode",
        choices=tuple(mode.value for mode in ContentMode),
        default=ContentMode.MATURE_FICTION.value,
    )
    parser.add_argument(
        "--context-output", choices=("summary", "full"), default="full"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_interactive_interview(
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        seed=args.seed,
        question_policy=args.question_policy,
        character_card=args.character_card,
        realism=args.realism,
        shadow_intensity=args.shadow,
        content_mode=ContentMode(args.content_mode),
        context_output=args.context_output,
    )
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
