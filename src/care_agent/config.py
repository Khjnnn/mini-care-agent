from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    fhir_base_url: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    agents_dir: str
    memory_path: str
    plugins: tuple[str, ...]
    #: Optional YAML file describing LLM backends for the router (week 5).
    router_config: str | None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            fhir_base_url=os.getenv("FHIR_BASE_URL", "http://localhost:8080/fhir"),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            agents_dir=os.getenv("AGENTS_DIR", "agents"),
            memory_path=os.getenv("MEMORY_PATH", ".data/memory.sqlite3"),
            plugins=tuple(p for p in os.getenv("PLUGINS", "").split(",") if p),
            router_config=os.getenv("ROUTER_CONFIG") or None,
        )
