from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from statistics import mean, pstdev
from typing import Any

from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.models import ActorKind, MatchMode


def run_benchmark(
    engine: GameEngine,
    mod_id: str,
    mode: MatchMode | str,
    seeds: Iterable[int] = range(50),
) -> dict[str, Any]:
    selected_mode = MatchMode(mode)
    evaluations: list[dict[str, Any]] = []
    outcomes: list[str] = []
    initiative_results: list[float] = []
    identity_results: list[float] = []
    seat_results: list[float] = []

    seed_list = list(seeds)
    adversarial = selected_mode in {
        MatchMode.HUMAN_VS_AGENT,
        MatchMode.AGENT_VS_AGENT,
    }
    mirrored_competition = adversarial and engine.mods[mod_id].competitive_balance_applicable
    cases = [
        (seed, reverse)
        for seed in seed_list
        for reverse in ((False, True) if mirrored_competition else (False,))
    ]
    for seed, reverse_seats in cases:
        view = engine.create_match(
            mod_id,
            "SyntheticHuman",
            seed,
            selected_mode,
            reverse_seats=reverse_seats,
        )
        used: Counter[str] = Counter()
        while view.status == "active":
            legal = view.legal_actions
            least_used = min(used[action.type] for action in legal)
            choices = [action for action in legal if used[action.type] == least_used]
            action = choices[(seed + sum(used.values())) % len(choices)]
            used[action.type] += 1
            if view.human_player_id is None:
                raise RuntimeError("Active Agent-only match should be advanced by the engine")
            view = engine.submit(view.id, view.human_player_id, action)

        evaluation = engine.evaluation(view.id)
        evaluations.append(evaluation)
        if "coop" in selected_mode.value:
            if "success" not in view.state:
                outcomes.append("completed")
            else:
                outcomes.append("success" if view.state.get("success") is True else "failure")
            continue
        if not engine.mods[mod_id].competitive_balance_applicable:
            outcomes.append("completed")
            continue

        top = max(view.scores.values())
        winners = [player.id for player in view.players if view.scores[player.id] == top]
        if len(winners) != 1:
            outcomes.append("draw")
            initiative_results.append(0.5)
            if selected_mode == MatchMode.AGENT_VS_AGENT:
                identity_results.append(0.5)
                seat_results.append(0.5)
            continue
        winner = winners[0]
        initiative_results.append(1.0 if winner == view.state["initiative"] else 0.0)
        humans = [player.id for player in view.players if player.kind == ActorKind.HUMAN]
        if humans:
            outcomes.append("human_win" if winner == humans[0] else "agent_win")
        else:
            astra = next(player.id for player in view.players if player.name == "Astra")
            outcomes.append("astra_win" if winner == astra else "nova_win")
            identity_results.append(1.0 if winner == astra else 0.0)
            seat_results.append(1.0 if winner == view.players[0].id else 0.0)

    composite = [item["composite_score"] for item in evaluations]
    dimension_keys = evaluations[0]["dimensions"]
    outcome_counts = Counter(outcomes)

    def centered_balance(value: float | None) -> float | None:
        return round(1 - min(1.0, abs(value - 0.5) * 2), 3) if value is not None else None

    initiative_equivalent = mean(initiative_results) if initiative_results else None
    identity_equivalent = mean(identity_results) if identity_results else None
    seat_equivalent = mean(seat_results) if seat_results else None
    balance: dict[str, Any]
    if mirrored_competition:
        draw_rate = outcome_counts["draw"] / len(evaluations)
        balance = {
            "initiative_balance": centered_balance(initiative_equivalent),
            "identity_balance": centered_balance(identity_equivalent),
            "seat_balance": centered_balance(seat_equivalent),
            "draw_rate": round(draw_rate, 3),
            "repeated_draw_penalty": round(1 - min(1.0, max(0.0, draw_rate - 0.35) / 0.65), 3),
        }
    elif "coop" in selected_mode.value and "completed" not in outcome_counts:
        success_rate = outcome_counts["success"] / len(evaluations)
        balance = {
            "success_rate": round(success_rate, 3),
            "target_success_balance": round(1 - min(1.0, abs(success_rate - 0.65) / 0.65), 3),
        }
    elif "coop" in selected_mode.value:
        balance = {
            "applicable": False,
            "reason": "sandbox_has_no_binary_success_outcome",
        }
    else:
        balance = {
            "applicable": False,
            "reason": "asymmetric_non_zero_sum_roles",
        }
    score_layer_keys = evaluations[0]["score_layers"]
    return {
        "evaluation_version": evaluations[0]["version"],
        "mod": mod_id,
        "mode": selected_mode.value,
        "matches": len(evaluations),
        "mirror_pairs": len(seed_list) if mirrored_competition else 0,
        "config_sha256": evaluations[0]["config_sha256"],
        "composite_mean": round(mean(composite), 3),
        "composite_sd": round(pstdev(composite), 3),
        "dimensions": {
            key: (round(mean(values), 3) if values else None)
            for key in dimension_keys
            if (
                values := [
                    item["dimensions"][key]
                    for item in evaluations
                    if item["dimensions"][key] is not None
                ]
            )
            or evaluations
        },
        "score_layers": {
            key: (
                round(mean(values), 3)
                if (
                    values := [
                        item["score_layers"][key]["score"]
                        for item in evaluations
                        if item["score_layers"][key]["score"] is not None
                    ]
                )
                else None
            )
            for key in score_layer_keys
        },
        "outcomes": dict(outcome_counts),
        "balance": balance,
        "initiative_win_equivalent": (
            round(initiative_equivalent, 3) if initiative_equivalent is not None else None
        ),
        "identity_anchor_win_equivalent": (
            round(identity_equivalent, 3) if identity_equivalent is not None else None
        ),
        "seat0_win_equivalent": (
            round(seat_equivalent, 3) if seat_equivalent is not None else None
        ),
        "rules_valid_rate": round(
            mean(float(item["valid_for_comparison"]) for item in evaluations), 3
        ),
    }


def run_suite(seeds: int = 25) -> list[dict[str, Any]]:
    engine = build_default_engine()
    return [
        run_benchmark(engine, mod.id, mode, range(seeds))
        for mod in engine.mods.values()
        for mode in mod.supported_modes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible MOD balance benchmarks")
    parser.add_argument(
        "--seeds", type=int, default=25, help="Seeds per mode; adversarial modes run both seats"
    )
    args = parser.parse_args()
    print(json.dumps(run_suite(args.seeds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
