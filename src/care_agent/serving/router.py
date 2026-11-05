"""LLM router across on-prem GPUs and cloud endpoints.

Policy, in order:

1. **Data classification decides eligibility.** A request is ``phi`` when it carries
   patient data — any tool result, a patient-scoped system prompt, or PII in the text.
   PHI may only go to backends marked ``phi_allowed`` (on-prem DGX, private-VPC GKE).
   This is never relaxed for availability: if every PHI-capable backend is down, the
   request fails rather than leaking to a public API.
2. **Tier hint narrows the pool.** Agents declare ``model: small|large``; a missing tier
   falls back to any tier.
3. **Ordering**: ``priority`` (lower first), then cost, then observed latency (EWMA).
4. **Failover with circuit breakers.** Errors/timeouts open a backend's breaker for
   ``cooldown_s``; after that one probe request is allowed (half-open).

Every response carries ``meta["routing"]`` (backend, classification, attempts) which
ends up in the agent trace and Langfuse.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from care_agent.llm.base import LLM, LLMResponse, Message
from care_agent.llm.openai_compat import OpenAICompatLLM

Classification = Literal["phi", "public"]

_PII = re.compile(
    r"\b\d{6}\s?-\s?[1-4]\d{6}\b"  # resident registration number
    r"|\b01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}\b"  # mobile number
    r"|patient_id\s*[=:]"
)


class NoBackendAvailable(RuntimeError):
    pass


def classify(messages: list[Message]) -> Classification:
    for m in messages:
        if m.get("role") == "tool":
            return "phi"
        content = m.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [p.get("text", "") for p in content or [] if isinstance(p, dict)]
        )
        if any(
            p.get("type") == "image_url" for p in (content if isinstance(content, list) else [])
        ):
            return "phi"  # documents (lab reports) are always treated as PHI
        if any(_PII.search(t or "") for t in texts):
            return "phi"
    return "public"


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    model: str
    location: Literal["onprem", "private_cloud", "public_cloud"]
    phi_allowed: bool
    tier: str = "large"
    priority: int = 100
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    timeout_s: float = 60.0
    api_key_env: str | None = None


class PolicyConfig(BaseModel):
    failure_threshold: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=30.0, gt=0)


class RouterConfig(BaseModel):
    backends: list[BackendConfig] = Field(min_length=1)
    policy: PolicyConfig = PolicyConfig()


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float, clock=time.monotonic) -> None:  # noqa: ANN001
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.opened_at = self.clock()


@dataclass
class Backend:
    config: BackendConfig
    llm: LLM
    breaker: CircuitBreaker
    ewma_latency_ms: float | None = None
    requests: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def observe(self, latency_ms: float, usage: dict[str, int]) -> None:
        self.ewma_latency_ms = (
            latency_ms
            if self.ewma_latency_ms is None
            else 0.8 * self.ewma_latency_ms + 0.2 * latency_ms
        )
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)

    @property
    def cost_usd(self) -> float:
        c = self.config
        return round(
            self.tokens_in / 1000 * c.cost_per_1k_input
            + self.tokens_out / 1000 * c.cost_per_1k_output,
            6,
        )

    def status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "location": self.config.location,
            "phi_allowed": self.config.phi_allowed,
            "tier": self.config.tier,
            "breaker": self.breaker.state,
            "ewma_latency_ms": None
            if self.ewma_latency_ms is None
            else round(self.ewma_latency_ms, 1),
            "requests": self.requests,
            "failures": self.failures,
            "cost_usd": self.cost_usd,
        }


@dataclass
class _Shared:
    backends: list[Backend]
    decisions: list[dict[str, Any]] = field(default_factory=list)


class RoutingLLM:
    def __init__(
        self, backends: list[Backend], *, tier: str | None = None, _shared: _Shared | None = None
    ) -> None:
        self._shared = _shared or _Shared(backends)
        self.tier = tier

    @property
    def backends(self) -> list[Backend]:
        return self._shared.backends

    @classmethod
    def from_config(cls, config: RouterConfig, *, llm_factory=None) -> RoutingLLM:  # noqa: ANN001
        def default_factory(b: BackendConfig) -> LLM:
            key = os.getenv(b.api_key_env, "") if b.api_key_env else "EMPTY"
            return OpenAICompatLLM(b.base_url, b.model, key or "EMPTY", timeout=b.timeout_s)

        make = llm_factory or default_factory
        p = config.policy
        return cls(
            [
                Backend(b, make(b), CircuitBreaker(p.failure_threshold, p.cooldown_s))
                for b in config.backends
            ]
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> RoutingLLM:
        data = yaml.safe_load(Path(path).read_text("utf-8"))
        return cls.from_config(RouterConfig.model_validate(data), **kwargs)

    def with_hint(self, tier: str | None) -> RoutingLLM:
        """A view with a tier preference that shares backends, breakers and stats."""
        return RoutingLLM([], tier=tier, _shared=self._shared)

    def candidates(self, classification: Classification) -> list[Backend]:
        pool = [b for b in self.backends if classification == "public" or b.config.phi_allowed]
        if self.tier and any(b.config.tier == self.tier for b in pool):
            pool = [b for b in pool if b.config.tier == self.tier]
        return sorted(
            pool,
            key=lambda b: (
                b.config.priority,
                b.config.cost_per_1k_input + b.config.cost_per_1k_output,
                b.ewma_latency_ms if b.ewma_latency_ms is not None else 0.0,
            ),
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        classification: Classification | None = None,
    ) -> LLMResponse:
        detected = classify(messages)
        # an explicit label can only make routing stricter, never looser
        cls_: Classification = "phi" if "phi" in (detected, classification) else "public"
        pool = self.candidates(cls_)
        attempts: list[dict[str, Any]] = []
        if not pool:
            raise NoBackendAvailable(f"no backend allowed for {cls_} data")

        for backend in pool:
            if not backend.breaker.allow():
                attempts.append({"backend": backend.config.name, "skipped": "circuit_open"})
                continue
            backend.requests += 1
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    backend.llm.complete(messages, tools), backend.config.timeout_s
                )
            except Exception as e:  # any backend failure → try the next one
                backend.failures += 1
                backend.breaker.failure()
                attempts.append(
                    {"backend": backend.config.name, "error": f"{type(e).__name__}: {e}"[:200]}
                )
                continue
            latency = (time.perf_counter() - t0) * 1000
            backend.breaker.success()
            backend.observe(latency, resp.usage)
            decision = {
                "backend": backend.config.name,
                "location": backend.config.location,
                "classification": cls_,
                "tier": self.tier,
                "attempts": attempts,
                "latency_ms": round(latency, 1),
            }
            self._shared.decisions.append(decision)
            resp.meta["routing"] = decision
            resp.model = f"{backend.config.name}/{resp.model or backend.config.model}"
            return resp

        raise NoBackendAvailable(f"all {cls_}-eligible backends failed: {attempts}")

    def status(self) -> list[dict[str, Any]]:
        return [b.status() for b in self.backends]
