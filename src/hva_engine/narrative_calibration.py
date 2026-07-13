from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class NarrativeOption:
    id: str
    motive_effects: dict[str, float]
    commitment_alignment: float
    short_term_cost: float
    irreversible_cost: float
    information_gain: float
    secret_exposure: float
    value_betrayal: float
    relationship_effect: float
    arc: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NarrativeOption:
        return cls(
            id=str(value["id"]),
            motive_effects={
                str(key): float(amount)
                for key, amount in value.get("motive_effects", {}).items()
            },
            commitment_alignment=float(value.get("commitment_alignment", 0.0)),
            short_term_cost=float(value.get("short_term_cost", 0.0)),
            irreversible_cost=float(value.get("irreversible_cost", 0.0)),
            information_gain=float(value.get("information_gain", 0.0)),
            secret_exposure=float(value.get("secret_exposure", 0.0)),
            value_betrayal=float(value.get("value_betrayal", 0.0)),
            relationship_effect=float(value.get("relationship_effect", 0.0)),
            arc=str(value.get("arc", "guarded")),
        )


@dataclass(frozen=True)
class NarrativeCase:
    id: str
    work: str
    medium: str
    year: int
    character: str
    source_url: str
    source_policy: str
    situation: str
    motives: dict[str, float]
    commitments: dict[str, float]
    traits: dict[str, float]
    secret_pressure: float
    uncertainty: float
    stakes: float
    social_threat: float
    controllability: float
    other_agency: float
    norm_compatibility: float
    identity_threat: float
    emotion_anchors: dict[str, float]
    options: tuple[NarrativeOption, ...]
    observed_option: str
    observed_arc: str
    ambiguity: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NarrativeCase:
        return cls(
            id=str(value["id"]),
            work=str(value["work"]),
            medium=str(value["medium"]),
            year=int(value["year"]),
            character=str(value["character"]),
            source_url=str(value["source_url"]),
            source_policy=str(value["source_policy"]),
            situation=str(value["situation"]),
            motives={str(key): float(amount) for key, amount in value["motives"].items()},
            commitments={
                str(key): float(amount) for key, amount in value["commitments"].items()
            },
            traits={str(key): float(amount) for key, amount in value["traits"].items()},
            secret_pressure=float(value.get("secret_pressure", 0.0)),
            uncertainty=float(value.get("uncertainty", 0.5)),
            stakes=float(value.get("stakes", 0.5)),
            social_threat=float(value.get("social_threat", 0.0)),
            controllability=float(value.get("controllability", 0.5)),
            other_agency=float(value.get("other_agency", 0.0)),
            norm_compatibility=float(value.get("norm_compatibility", 0.5)),
            identity_threat=float(value.get("identity_threat", 0.0)),
            emotion_anchors={
                str(key): float(amount)
                for key, amount in value.get("emotion_anchors", {}).items()
            },
            options=tuple(NarrativeOption.from_dict(item) for item in value["options"]),
            observed_option=str(value["observed_option"]),
            observed_arc=str(value["observed_arc"]),
            ambiguity=float(value.get("ambiguity", 0.25)),
        )


@dataclass(frozen=True)
class NarrativePrediction:
    case_id: str
    option_id: str
    option_scores: dict[str, float]
    confidence: float
    emotions: dict[str, float]
    predicted_arc: str
    dominant_motives: tuple[str, ...]

    def public_view(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "option_id": self.option_id,
            "option_scores": {
                key: round(value, 4) for key, value in self.option_scores.items()
            },
            "confidence": round(self.confidence, 3),
            "emotions": {key: round(value, 3) for key, value in self.emotions.items()},
            "predicted_arc": self.predicted_arc,
            "dominant_motives": list(self.dominant_motives),
        }


class NarrativeDatasetError(ValueError):
    pass


def load_reference_cases(path: Path | None = None) -> tuple[dict[str, Any], list[NarrativeCase]]:
    source = path or Path(__file__).with_name("data") / "narrative_calibration_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    cases = [NarrativeCase.from_dict(item) for item in payload["cases"]]
    validate_reference_dataset(payload, cases)
    return payload, cases


def validate_reference_dataset(
    payload: dict[str, Any], cases: list[NarrativeCase]
) -> None:
    if payload.get("contains_source_text") is not False:
        raise NarrativeDatasetError("Calibration data must not redistribute source text")
    if payload.get("dataset_type") != "human_authored_narrative_reference":
        raise NarrativeDatasetError("Narrative references must not be labelled as real-human data")
    allowed_policies = {
        "public_domain_us_paraphrase",
        "copyrighted_metadata_only_paraphrase",
        "licensed_annotation",
    }
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise NarrativeDatasetError(f"Duplicate case id: {case.id}")
        seen.add(case.id)
        if case.source_policy not in allowed_policies:
            raise NarrativeDatasetError(f"Unsupported source policy: {case.source_policy}")
        if not case.source_url.startswith("https://"):
            raise NarrativeDatasetError(f"Case lacks an HTTPS source: {case.id}")
        if len(case.situation) > 600:
            raise NarrativeDatasetError(f"Case paraphrase is unexpectedly long: {case.id}")
        option_ids = {option.id for option in case.options}
        if len(option_ids) < 2 or case.observed_option not in option_ids:
            raise NarrativeDatasetError(f"Invalid options or ground truth: {case.id}")
        raw = payload["cases"][len(seen) - 1]
        forbidden = {"quote", "raw_text", "transcript", "script", "excerpt"} & set(raw)
        if forbidden:
            raise NarrativeDatasetError(
                f"Source text fields are forbidden in redistributable data: {sorted(forbidden)}"
            )


class NarrativeDecisionModel:
    """Generic motive-conflict model; it never reads the recorded outcome while predicting."""

    def predict(self, case: NarrativeCase) -> NarrativePrediction:
        loss_aversion = float(case.traits.get("loss_aversion", 0.5))
        conscientiousness = float(case.traits.get("conscientiousness", 0.5))
        attachment = float(case.traits.get("attachment", 0.5))
        integrity = float(case.traits.get("integrity", 0.5))
        commitment_strength = mean(case.commitments.values()) if case.commitments else 0.5
        scores: dict[str, float] = {}
        for option in case.options:
            motive_utility = sum(
                case.motives.get(name, 0.0) * effect
                for name, effect in option.motive_effects.items()
            )
            commitment = option.commitment_alignment * commitment_strength
            information = (
                option.information_gain
                * case.uncertainty
                * (0.35 + 0.65 * conscientiousness)
            )
            relationship = option.relationship_effect * attachment
            short_cost = option.short_term_cost * (0.30 + 0.70 * loss_aversion)
            irreversible_cost = option.irreversible_cost * (
                0.25 + 0.50 * conscientiousness + 0.25 * case.stakes
            )
            secrecy_cost = option.secret_exposure * case.secret_pressure * (
                0.45 + 0.55 * case.motives.get("self_preservation", 0.5)
            )
            betrayal_cost = option.value_betrayal * (0.45 + 0.55 * integrity)
            scores[option.id] = (
                motive_utility
                + 0.65 * commitment
                + 0.48 * information
                + 0.46 * relationship
                - 0.52 * short_cost
                - 0.48 * irreversible_cost
                - 0.42 * secrecy_cost
                - 0.62 * betrayal_cost
            )
        ranked = sorted(scores, key=lambda option_id: (-scores[option_id], option_id))
        selected = ranked[0]
        gap = scores[ranked[0]] - scores[ranked[1]] if len(ranked) > 1 else 1.0
        confidence = _clamp((0.42 + 0.36 * math.tanh(abs(gap))) * (1 - 0.62 * case.ambiguity))
        emotions = {
            "stress": _clamp(
                0.16
                + 0.42 * case.social_threat
                + 0.25 * case.stakes
                + 0.17 * (1 - case.controllability)
            ),
            "anger": _clamp(
                case.other_agency * (1 - case.norm_compatibility) * 0.78
            ),
            "fear": _clamp(case.social_threat * (1 - case.controllability) * 0.86),
            "shame": _clamp(case.identity_threat * conscientiousness * 0.82),
            "hope": _clamp(
                0.22 + 0.46 * case.controllability + 0.32 * case.motives.get("redemption", 0)
            ),
        }
        chosen_option = next(option for option in case.options if option.id == selected)
        dominant_motives = tuple(
            name
            for name, _value in sorted(
                case.motives.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        )
        return NarrativePrediction(
            case_id=case.id,
            option_id=selected,
            option_scores=scores,
            confidence=confidence,
            emotions=emotions,
            predicted_arc=chosen_option.arc,
            dominant_motives=dominant_motives,
        )


class NarrativeCalibrationEvaluator:
    def evaluate_case(
        self, case: NarrativeCase, prediction: NarrativePrediction
    ) -> dict[str, Any]:
        ranked = sorted(
            prediction.option_scores,
            key=lambda option_id: (-prediction.option_scores[option_id], option_id),
        )
        observed_rank = ranked.index(case.observed_option)
        motive_ranking = 1 - observed_rank / max(1, len(ranked) - 1)
        emotion_keys = set(case.emotion_anchors) & set(prediction.emotions)
        appraisal_fit = (
            1
            - sum(
                abs(case.emotion_anchors[key] - prediction.emotions[key])
                for key in emotion_keys
            )
            / max(1, len(emotion_keys))
        )
        expected_confidence = 1 - case.ambiguity
        uncertainty_calibration = 1 - abs(prediction.confidence - expected_confidence)
        components = {
            "decision_match": float(prediction.option_id == case.observed_option),
            "motive_ranking": _clamp(motive_ranking),
            "appraisal_fit": _clamp(appraisal_fit),
            "arc_transition": float(prediction.predicted_arc == case.observed_arc),
            "uncertainty_calibration": _clamp(uncertainty_calibration),
        }
        composite = (
            0.36 * components["decision_match"]
            + 0.18 * components["motive_ranking"]
            + 0.20 * components["appraisal_fit"]
            + 0.14 * components["arc_transition"]
            + 0.12 * components["uncertainty_calibration"]
        )
        return {
            "case_id": case.id,
            "work": case.work,
            "medium": case.medium,
            "character": case.character,
            "source_policy": case.source_policy,
            "observed_option": case.observed_option,
            "prediction": prediction.public_view(),
            "components": {key: round(value, 3) for key, value in components.items()},
            "composite": round(composite, 3),
        }

    def _control_prediction(self, case: NarrativeCase, control: str) -> NarrativePrediction:
        if control == "first_option":
            scores = {
                option.id: float(len(case.options) - index)
                for index, option in enumerate(case.options)
            }
        elif control == "self_preservation_only":
            scores = {
                option.id: option.motive_effects.get("self_preservation", 0.0)
                - 0.25 * option.short_term_cost
                for option in case.options
            }
        elif control == "commitment_only":
            scores = {
                option.id: option.commitment_alignment - 0.15 * option.short_term_cost
                for option in case.options
            }
        else:
            raise ValueError(f"Unknown narrative control: {control}")
        selected = max(scores, key=scores.get)
        option = next(item for item in case.options if item.id == selected)
        return NarrativePrediction(
            case_id=case.id,
            option_id=selected,
            option_scores=scores,
            confidence=0.5,
            emotions={key: 0.5 for key in case.emotion_anchors},
            predicted_arc=option.arc,
            dominant_motives=(),
        )

    def run(
        self, cases: list[NarrativeCase], holdout_case_ids: set[str] | None = None
    ) -> dict[str, Any]:
        model = NarrativeDecisionModel()
        rows = [self.evaluate_case(case, model.predict(case)) for case in cases]
        component_names = rows[0]["components"] if rows else {}
        media = sorted({row["medium"] for row in rows})
        controls = {}
        for control in ("first_option", "self_preservation_only", "commitment_only"):
            predictions = [self._control_prediction(case, control) for case in cases]
            controls[control] = round(
                mean(
                    float(prediction.option_id == case.observed_option)
                    for case, prediction in zip(cases, predictions, strict=True)
                ),
                3,
            )
        model_accuracy = mean(row["components"]["decision_match"] for row in rows)
        holdout_ids = holdout_case_ids or set()
        holdout_rows = [row for row in rows if row["case_id"] in holdout_ids]
        return {
            "version": "narrative-calibration-v1",
            "calibration_status": "prototype_not_independently_annotated",
            "dataset_type": "human_authored_narrative_reference",
            "not_real_human_behavior_data": True,
            "known_limitations": [
                "character cards, labels, and scoring weights are author-designed",
                "fictional narrative choices are not measurements of real-human behavior",
                "the small suite is a mechanism regression test, not a population benchmark",
            ],
            "cases": len(rows),
            "composite": round(mean(row["composite"] for row in rows), 3),
            "components": {
                key: round(mean(row["components"][key] for row in rows), 3)
                for key in component_names
            },
            "negative_controls": controls,
            "discriminative_margin": round(model_accuracy - max(controls.values()), 3),
            "holdout": {
                "cases": len(holdout_rows),
                "decision_accuracy": (
                    round(
                        mean(row["components"]["decision_match"] for row in holdout_rows), 3
                    )
                    if holdout_rows
                    else None
                ),
                "warning": "author-designed holdout; independent annotation is still required",
            },
            "by_medium": {
                medium: {
                    "cases": len(selected),
                    "composite": round(mean(row["composite"] for row in selected), 3),
                    "decision_accuracy": round(
                        mean(row["components"]["decision_match"] for row in selected), 3
                    ),
                }
                for medium in media
                if (selected := [row for row in rows if row["medium"] == medium])
            },
            "decision_distribution": {
                option: sum(row["prediction"]["option_id"] == option for row in rows)
                for option in sorted({row["prediction"]["option_id"] for row in rows})
            },
            "failures": [
                row["case_id"]
                for row in rows
                if row["components"]["decision_match"] == 0.0
            ],
            "license_summary": {
                policy: sum(row["source_policy"] == policy for row in rows)
                for policy in sorted({row["source_policy"] for row in rows})
            },
            "rows": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the human-authored narrative character calibration suite"
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    metadata, cases = load_reference_cases(args.data)
    result = NarrativeCalibrationEvaluator().run(
        cases, set(metadata.get("holdout_case_ids", []))
    )
    if not args.details:
        result.pop("rows", None)
    result["dataset_version"] = metadata["version"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
