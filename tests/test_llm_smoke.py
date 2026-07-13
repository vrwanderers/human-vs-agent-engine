from __future__ import annotations

import json

import pytest

import hva_engine.llm_smoke as smoke_module
from hva_engine.engine import GameEngine, build_default_engine
from hva_engine.llm import LLMDecisionClient, LLMResponse
from hva_engine.llm_smoke import run_real_interview
from hva_engine.models import ContentMode
from hva_engine.mods import AdversarialInterview


class SmokeProvider:
    name = "smoke-provider"

    def __init__(self) -> None:
        self.calls = 0

    def complete_sync(self, _request):
        self.calls += 1
        return LLMResponse(
            json.dumps(
                {
                    "action_index": self.calls % 3,
                    "reason": "strict smoke response",
                    "utterance": f"第{self.calls}次回答会结合当前问题与既有事实。",
                    "fact_proposals": [],
                },
                ensure_ascii=False,
            ),
            "smoke-model-1",
            {"total_tokens": 50},
            {},
        )


def test_real_smoke_runner_exercises_strict_provider_path(monkeypatch) -> None:
    provider = SmokeProvider()
    engine = GameEngine(
        llm_decision_client=LLMDecisionClient(provider),
        llm_mod_ids={"adversarial_interview"},
        llm_fallback=False,
    )
    engine.register(AdversarialInterview())
    monkeypatch.setattr(smoke_module, "build_default_engine", lambda: engine)

    result = run_real_interview(
        seed=7,
        question_policy="rotate",
        realism=0.85,
        shadow_intensity=0.4,
        content_mode=ContentMode.MATURE_FICTION,
        allow_fallback=False,
        character_card="dou_e",
    )
    assert result["real_llm_decisions"] == 6
    assert result["fallback_decisions"] == 0
    assert result["usage"]["total_tokens"] == 300
    assert len(result["transcript"]) == 12
    assert result["evaluation"]["valid_for_provider_comparison"] is True


def test_real_smoke_runner_rejects_disabled_runtime(monkeypatch) -> None:
    monkeypatch.setattr(smoke_module, "build_default_engine", build_default_engine)
    with pytest.raises(RuntimeError, match="disabled"):
        run_real_interview(
            seed=1,
            question_policy="first",
            realism=0.7,
            shadow_intensity=0.0,
            content_mode=ContentMode.STANDARD,
            allow_fallback=False,
        )
