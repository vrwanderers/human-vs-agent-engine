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
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.__dict__ for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format and self.supports_response_format:
            payload["response_format"] = request.response_format
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model"),
            usage={key: int(value) for key, value in data.get("usage", {}).items()},
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

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def choose_action(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> tuple[int, LLMResponse]:
        index, _proposals, response = await self.choose_action_and_facts(messages, legal_actions)
        return index, response

    async def choose_action_and_facts(
        self, messages: list[LLMMessage], legal_actions: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], LLMResponse]:
        response = await self.provider.complete(LLMRequest(messages=messages))
        try:
            choice = json.loads(response.content)
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
        return index, proposals, response
