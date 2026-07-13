from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from hva_engine.engine import build_default_engine
from hva_engine.models import AgentCharacterSelection, AgentTuning, ContentMode


def run_real_interview(
    *,
    seed: int,
    question_policy: str,
    realism: float,
    shadow_intensity: float,
    content_mode: ContentMode,
    allow_fallback: bool,
    character_card: str | None = None,
) -> dict[str, Any]:
    engine = build_default_engine()
    if engine.agent_runtime != "llm":
        raise RuntimeError(
            "Real LLM runtime is disabled. Set HVA_AGENT_RUNTIME=llm and HVA_LLM_* variables."
        )
    engine.llm_fallback = allow_fallback
    view = engine.create_match(
        "adversarial_interview",
        human_name="LLM Smoke Interviewer",
        seed=seed,
        agent_tuning=AgentTuning(
            realism=realism,
            shadow_intensity=shadow_intensity,
            content_mode=content_mode,
        ),
        agent_characters=(
            [AgentCharacterSelection(card_id=character_card)]
            if character_card
            else []
        ),
    )
    human_turn = 0
    while view.status == "active":
        if question_policy == "most_severe":
            action = max(
                view.legal_actions,
                key=lambda candidate: float(candidate.payload.get("severity", 0.0)),
            )
        elif question_policy == "rotate":
            action = view.legal_actions[human_turn % len(view.legal_actions)]
        else:
            action = view.legal_actions[0]
        view = engine.submit(view.id, view.human_player_id, action)
        human_turn += 1

    decisions = [
        event for event in engine.get(view.id).events if event.type == "agent_decision"
    ]
    usage: Counter[str] = Counter()
    decision_rows: list[dict[str, Any]] = []
    for event in decisions:
        llm = event.payload.get("llm") or {}
        usage.update({key: int(value) for key, value in llm.get("usage", {}).items()})
        decision_rows.append(
            {
                "event_seq": event.seq,
                "action": event.payload.get("action_type"),
                "source": event.payload.get("decision_source"),
                "reason": event.payload.get("rationale"),
                "utterance": event.payload.get("utterance"),
                "response_plan": event.payload.get("response_plan"),
                "model": llm.get("model"),
                "usage": llm.get("usage", {}),
                "fact_proposals": llm.get("fact_proposals"),
                "psychological_matrix": event.payload.get("psychological_matrix"),
                "llm_error": event.payload.get("llm_error"),
            }
        )
    real_decisions = sum(row["source"] == "llm" for row in decision_rows)
    fallback_decisions = len(decision_rows) - real_decisions
    if not allow_fallback and fallback_decisions:
        raise RuntimeError(
            f"Expected six real LLM decisions, observed {fallback_decisions} fallbacks"
        )
    evaluation = engine.evaluation(view.id)
    agent_summary = next(iter(engine.debug_view(view.id).agent_summaries.values()))
    return {
        "test": "real_llm_adversarial_interview",
        "seed": seed,
        "provider": engine.llm_decision_client.provider.name,
        "configured_mods": sorted(engine.llm_mod_ids),
        "character_card": character_card,
        "real_llm_decisions": real_decisions,
        "fallback_decisions": fallback_decisions,
        "usage": dict(usage),
        "final_arc": view.state["arc_stage"],
        "final_psychological_matrix": agent_summary["psychological_matrix"],
        "fact_graph_stats": agent_summary["fact_graph"]["stats"],
        "transcript": view.state["transcript"],
        "decisions": decision_rows,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a six-turn interview using a real LLM")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--question-policy",
        choices=("first", "most_severe", "rotate"),
        default="most_severe",
    )
    parser.add_argument("--realism", type=float, default=0.85)
    parser.add_argument("--shadow", type=float, default=0.0)
    parser.add_argument(
        "--content-mode",
        choices=tuple(mode.value for mode in ContentMode),
        default=ContentMode.STANDARD.value,
    )
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--character-card",
        help="Built-in declarative character card ID from GET /api/character-cards",
    )
    args = parser.parse_args()
    if os.environ.get("HVA_AGENT_RUNTIME", "").lower() not in {"llm", "hybrid"}:
        parser.error("Set HVA_AGENT_RUNTIME=llm before running this command")
    result = run_real_interview(
        seed=args.seed,
        question_policy=args.question_policy,
        realism=args.realism,
        shadow_intensity=args.shadow,
        content_mode=ContentMode(args.content_mode),
        allow_fallback=args.allow_fallback,
        character_card=args.character_card,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
