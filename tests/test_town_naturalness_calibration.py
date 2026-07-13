import pytest

from hva_engine.llm import LLMDecisionClient, LLMResponse
from hva_engine.town_naturalness_calibration import _run_condition, run_calibration


class CalibrationProvider:
    name = "calibration-provider"
    model = "calibration-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_sync(self, request):
        self.calls += 1
        return LLMResponse(
            '{"action_index":0,"reason":"observe",'
            '"utterance":"我先核对眼前的信息。","fact_proposals":[]}',
            self.model,
            {"total_tokens": 80},
            {},
        )


def test_calibration_condition_uses_provider_without_fallback() -> None:
    provider = CalibrationProvider()
    transcript, metadata = _run_condition(
        seed=7,
        rounds=2,
        decision_client=LLMDecisionClient(provider),
    )
    assert provider.calls == 6
    assert transcript
    assert metadata["condition"] == "real_llm"
    assert metadata["provider"] == provider.name


def test_real_calibration_refuses_to_run_without_provider_configuration(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("HVA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("HVA_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="requires configured provider"):
        run_calibration(
            seed=7,
            rounds=1,
            study_id="missing-provider",
            database_path=tmp_path / "blind.sqlite3",
        )
