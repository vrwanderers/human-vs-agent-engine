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
        return LLMDecisionResult(index, reason, utterance, proposals, response)

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
