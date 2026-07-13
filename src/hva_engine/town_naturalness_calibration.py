from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hva_engine.blind_eval import BlindSample, SQLiteBlindEvaluationStore
from hva_engine.engine import GameEngine
from hva_engine.llm import LLMDecisionClient, OpenAICompatibleProvider
from hva_engine.mods import AgentTown


def _observable_transcript(engine: GameEngine, match_id: str) -> tuple[str, ...]:
    match = engine.get(match_id)
    names = {player.id: player.name for player in match.players}
    rows: list[str] = []
    for event in match.events:
        if event.type.startswith("town_world_"):
            rows.append(
                f"世界：{event.payload.get('headline')}——{event.payload.get('summary')}"
            )
        elif event.type == "town_conversation":
            rows.append(
                f"{names.get(event.actor_id, 'Agent')}：{event.payload.get('dialogue', '')}"
            )
        elif event.type in {
            "town_social_posted",
            "town_social_reshared",
            "town_social_commented",
        }:
            rows.append(
                f"{names.get(event.actor_id, 'Agent')}（社交媒体）："
                f"{event.payload.get('content', '')}"
            )
        elif event.type == "town_claim_verified":
            rows.append(
                f"{names.get(event.actor_id, 'Agent')} 求证："
                f"{event.payload.get('result')}，证据={event.payload.get('evidence_event_ids')}"
            )
        elif event.type in {
            "town_incident_response",
            "town_sheltered",
            "town_neighbor_supported",
        }:
            rows.append(f"{names.get(event.actor_id, 'Agent')} 行动：{event.type}")
    return tuple(rows[-80:])


def _run_condition(
    *, seed: int, rounds: int, decision_client: LLMDecisionClient | None
) -> tuple[tuple[str, ...], dict[str, Any]]:
    engine = GameEngine(
        llm_decision_client=decision_client,
        llm_mod_ids={"agent_town"} if decision_client else set(),
        llm_fallback=False,
    )
    engine.register(AgentTown())
    view = engine.create_match(
        "agent_town",
        seed=seed,
        agent_memory_owner_ids=["blind-astra", "blind-nova", "blind-mira"],
    )
    completed_rounds = 0
    while view.status == "active" and completed_rounds < rounds:
        wait = next(action for action in view.legal_actions if action.type == "wait")
        view = engine.submit(view.id, view.human_player_id, wait)
        completed_rounds += 1
    evaluation = engine.evaluation(view.id)
    provider = decision_client.provider if decision_client else None
    return _observable_transcript(engine, view.id), {
        "seed": seed,
        "rounds": completed_rounds,
        "provider": provider.name if provider else "baseline",
        "model": getattr(provider, "model", None) if provider else None,
        "condition": "real_llm" if provider else "baseline",
        "engine_composite": evaluation["composite_score"],
        "town_proxy": evaluation["mod_specific_profile"],
    }


def run_calibration(
    *, seed: int, rounds: int, study_id: str, database_path: Path
) -> dict[str, Any]:
    missing = [
        key
        for key in ("HVA_LLM_BASE_URL", "HVA_LLM_MODEL")
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(
            "Real-LLM calibration requires configured provider values: "
            + ", ".join(missing)
        )
    provider = OpenAICompatibleProvider.from_env()
    baseline_text, baseline_metadata = _run_condition(
        seed=seed, rounds=rounds, decision_client=None
    )
    llm_text, llm_metadata = _run_condition(
        seed=seed,
        rounds=rounds,
        decision_client=LLMDecisionClient(provider),
    )
    store = SQLiteBlindEvaluationStore(database_path)
    trial = store.create_trial(
        study_id,
        BlindSample("baseline", baseline_text, baseline_metadata),
        BlindSample("real_llm", llm_text, llm_metadata),
        seed,
    )
    return {
        "study_id": study_id,
        "trial": trial,
        "rating_endpoint": "/api/evaluations/blind-ratings",
        "summary_endpoint": f"/api/evaluations/blind-summary/{study_id}",
        "real_provider_required": True,
        "fallback_allowed": False,
        "database_path": str(database_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a blinded baseline-vs-real-LLM Agent Town trial"
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--study-id", default="agent-town-naturalness-v1")
    parser.add_argument(
        "--database", type=Path, default=Path("data/evaluation/hva-blind.sqlite3")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_calibration(
                seed=args.seed,
                rounds=args.rounds,
                study_id=args.study_id,
                database_path=args.database,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
