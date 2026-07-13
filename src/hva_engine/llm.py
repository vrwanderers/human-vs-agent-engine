from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 400
    response_format: dict[str, Any] | None = field(default_factory=lambda: {"type": "json_object"})


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str | None
    usage: dict[str, int]
    raw: dict[str, Any]


@dataclass(frozen=True)
class LLMDecisionResult:
    action_index: int
    reason: str
    utterance: str | None
    response_plan: dict[str, Any]
    influence_intent: dict[str, Any]
    fact_proposals: list[dict[str, Any]]
    response: LLMResponse


class LLMProvider(Protocol):
    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """Works with OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        supports_response_format: bool = True,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.supports_response_format = supports_response_format

    @classmethod
    def from_env(cls, prefix: str = "HVA_LLM") -> OpenAICompatibleProvider:
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        model = os.environ.get(f"{prefix}_MODEL")
        if not base_url or not model:
            raise ValueError(f"{prefix}_BASE_URL and {prefix}_MODEL are required")
        return cls(
            name=os.environ.get(f"{prefix}_PROVIDER", "openai-compatible"),
            base_url=base_url,
            model=model,
            api_key=os.environ.get(f"{prefix}_API_KEY"),
            supports_response_format=os.environ.get(
                f"{prefix}_SUPPORTS_RESPONSE_FORMAT", "true"
            ).lower()
            not in {"0", "false", "no"},
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        headers = self._headers()
        payload = self._payload(request)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_response(data)

    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous debug-runtime path; production servers should use async orchestration."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(request),
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_response(data)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.__dict__ for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format and self.supports_response_format:
            payload["response_format"] = request.response_format
        return payload

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model"),
            usage={
                key: int(value)
                for key, value in data.get("usage", {}).items()
                if isinstance(value, (int, float))
            },
            raw=data,
        )


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown LLM provider: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)


class LLMDecisionClient:
    """Turns a layered prompt into a rule-constrained action selection."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.65,
        max_tokens: int = 700,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def choose_action(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> tuple[int, LLMResponse]:
        index, _proposals, response = await self.choose_action_and_facts(messages, legal_actions)
        return index, response

    async def choose_action_and_facts(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], LLMResponse]:
        result = await self.choose_structured(messages, legal_actions)
        return result.action_index, result.fact_proposals, result.response

    async def choose_structured(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> LLMDecisionResult:
        response = await self.provider.complete(self._request(messages))
        return self._parse_decision(response, legal_actions)

    def choose_structured_sync(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> LLMDecisionResult:
        complete_sync = getattr(self.provider, "complete_sync", None)
        if complete_sync is None:
            raise TypeError("Configured LLM provider does not implement complete_sync")
        response = complete_sync(self._request(messages))
        return self._parse_decision(response, legal_actions)

    def _request(self, messages: list[LLMMessage]) -> LLMRequest:
        return LLMRequest(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _parse_decision(
        self, response: LLMResponse, legal_actions: list[dict[str, Any]]
    ) -> LLMDecisionResult:
        try:
            choice = json.loads(self._extract_json(response.content))
            index = int(choice["action_index"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("LLM must return JSON with an integer action_index") from exc
        if not 0 <= index < len(legal_actions):
            raise ValueError("LLM selected an action outside the legal action list")
        proposals = choice.get("fact_proposals", [])
        if not isinstance(proposals, list) or len(proposals) > 5:
            raise ValueError("fact_proposals must be a list with at most five items")
        required = {"subject", "predicate", "object", "basis_fact_ids"}
        for proposal in proposals:
            if not isinstance(proposal, dict) or not required <= set(proposal):
                raise ValueError(
                    "Every fact proposal must include subject, predicate, object, "
                    "and basis_fact_ids"
                )
            if not isinstance(proposal["basis_fact_ids"], list):
                raise ValueError("basis_fact_ids must be a list")
        reason = str(choice.get("reason", ""))[:500]
        utterance_value = choice.get("utterance")
        utterance = str(utterance_value).strip()[:1_600] if utterance_value else None
        response_plan = self._parse_response_plan(choice.get("response_plan"), index, legal_actions)
        influence_intent = self._parse_influence_intent(choice.get("influence_intent"))
        return LLMDecisionResult(
            index,
            reason,
            utterance,
            response_plan,
            influence_intent,
            proposals,
            response,
        )

    def _parse_influence_intent(self, raw_intent: Any) -> dict[str, Any]:
        if raw_intent is None:
            return {}
        if not isinstance(raw_intent, dict):
            raise ValueError("influence_intent must be an object")
        scope = str(raw_intent.get("scope", "fictional_game"))
        if scope != "fictional_game":
            raise ValueError("Strategic influence scope must be fictional_game")
        threat_basis = str(raw_intent.get("threat_basis", "none"))
        if threat_basis not in {"none", "legal_game_consequence"}:
            raise ValueError("Threats may reference only legal in-game consequences")
        numeric_fields = (
            "truthfulness",
            "information_selectivity",
            "incentive_pressure",
            "coercive_pressure",
            "ambiguity",
            "commitment",
            "expected_gain",
            "detection_risk",
            "relationship_risk",
        )
        parsed: dict[str, Any] = {
            "scope": scope,
            "threat_basis": threat_basis,
            "target_belief": " ".join(
                str(raw_intent.get("target_belief", "")).split()
            )[:240],
        }
        for field_name in numeric_fields:
            if field_name not in raw_intent:
                continue
            try:
                parsed[field_name] = max(0.0, min(1.0, float(raw_intent[field_name])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Influence field must be numeric: {field_name}") from exc
        if parsed.get("coercive_pressure", 0.0) > 0.05 and threat_basis != (
            "legal_game_consequence"
        ):
            raise ValueError(
                "Coercive pressure requires a legal_game_consequence threat basis"
            )
        return parsed

    def _parse_response_plan(
        self,
        raw_plan: Any,
        action_index: int,
        legal_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        legal_types = [str(action.get("type")) for action in legal_actions]
        primary = legal_types[action_index]
        plan = raw_plan if isinstance(raw_plan, dict) else {}
        raw_weights = plan.get("strategy_weights", {})
        weights: dict[str, float] = {}
        if isinstance(raw_weights, dict):
            for strategy, raw_value in raw_weights.items():
                if strategy not in legal_types:
                    raise ValueError(f"Response plan contains illegal strategy: {strategy}")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Response plan weights must be numeric") from exc
                if value > 0:
                    weights[strategy] = min(1.0, value)
        if not weights:
            weights = {primary: 1.0}
        if primary not in weights:
            weights[primary] = max(0.15, min(weights.values()))
        weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:4])
        total = sum(weights.values())
        normalized = {key: round(value / total, 4) for key, value in weights.items()}
        intensity = max(0.0, min(1.0, float(plan.get("intensity", 0.6))))
        emotional_display = str(plan.get("emotional_display", "controlled"))[:64]
        raw_tags = plan.get("stance_tags", [])
        stance_tags = (
            [str(value)[:40] for value in raw_tags[:4]] if isinstance(raw_tags, list) else []
        )
        raw_reveals = plan.get("reveal_fact_ids", [])
        reveal_fact_ids = (
            [str(value)[:80] for value in raw_reveals[:3]] if isinstance(raw_reveals, list) else []
        )
        return {
            "primary_strategy": primary,
            "strategy_weights": normalized,
            "intensity": round(intensity, 3),
            "emotional_display": emotional_display,
            "stance_tags": stance_tags,
            "reveal_fact_ids": reveal_fact_ids,
        }

    def _extract_json(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            text = text[first_newline + 1 :] if first_newline >= 0 else text
            if text.endswith("```"):
                text = text[:-3]
        start = text.find("{")
        end = text.rfind("}")
        return text[start : end + 1] if start >= 0 and end >= start else text
