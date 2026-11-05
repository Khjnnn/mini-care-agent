"""The platform: one shared tool registry + declaratively defined agents.

How another team ships something:

* **A tool / skill** – a Python module with ``register(registry, platform)`` listed in
  ``plugins``. Tools are plain typed async functions (see ``skills/lab_reference.py``).
* **An agent** – a YAML file in ``agents/``: prompt, allowed tools, sub-agents, limits.
  No Python needed.
* **A sub-agent** – any agent listed in another agent's ``subagents``. It is exposed to
  the parent as an ``ask_<name>`` tool and inherits the parent's patient scope and
  approver through ``RunContext`` (the model cannot pick a patient for it).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from care_agent.agent.loop import Agent, current_run
from care_agent.agent.prompts import SYSTEM_PROMPT
from care_agent.agent.trace import TraceEvent
from care_agent.documents.extract import LabReportExtractor, OCRBackend
from care_agent.documents.pipeline import LabReportPipeline
from care_agent.fhir.client import FHIRClient
from care_agent.llm.base import LLM
from care_agent.memory.store import MemoryStore
from care_agent.memory.tools import build_memory_tools
from care_agent.skills import lab_reference
from care_agent.tools.fhir_tools import build_fhir_tools
from care_agent.tools.registry import ToolRegistry

MAX_SUBAGENT_DEPTH = 2


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    description: str = Field(min_length=5)
    system_prompt: str = SYSTEM_PROMPT
    tools: list[str] = Field(default_factory=list)
    subagents: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=6, ge=1, le=20)
    #: Use session memory (conversation history) when the caller passes a session id.
    memory: bool = True
    #: Hint for the LLM router (week 5): e.g. "small", "large".
    model: str | None = None


class Platform:
    def __init__(
        self,
        llm: LLM,
        fhir: FHIRClient,
        *,
        memory: MemoryStore | None = None,
        plugins: Iterable[str] = (),
        document_llm: LLM | None = None,
        ocr: OCRBackend | None = None,
    ) -> None:
        self.llm = llm
        self.fhir = fhir
        self.memory = memory or MemoryStore()
        #: Document structuring may use a different (vision) model than the chat agents.
        self.lab_reports = LabReportPipeline(LabReportExtractor(document_llm or llm, ocr=ocr), fhir)
        self.tools = ToolRegistry()
        build_fhir_tools(fhir, self.tools)
        build_memory_tools(self.memory, self.tools)
        lab_reference.register(self.tools, self)
        for module in plugins:
            self.load_plugin(module)
        self.specs: dict[str, AgentSpec] = {}
        #: Optional trace exporter (``observability.langfuse.LangfuseExporter``).
        self.exporter: Any = None

    # -- registration ----------------------------------------------------------------
    def load_plugin(self, module: str) -> None:
        mod = importlib.import_module(module)
        register = getattr(mod, "register", None)
        if not callable(register):
            raise TypeError(f"plugin {module!r} has no register(registry, platform) function")
        register(self.tools, self)

    def register_agent(self, spec: AgentSpec | dict[str, Any]) -> AgentSpec:
        spec = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
        if spec.name in self.specs:
            raise ValueError(f"agent {spec.name!r} is already registered")
        tools = self.tools.subset(spec.tools)  # fail fast on unknown tools
        # An agent that can read a patient's record must be told whose record it is.
        # Agents without patient-scoped tools (e.g. lab_explainer) deliberately never
        # see the patient id, so their requests can be routed as non-PHI.
        if any(t.patient_scoped for t in tools) and "{patient_id}" not in spec.system_prompt:
            raise ValueError(
                f"agent {spec.name!r} uses patient-scoped tools; system_prompt needs {{patient_id}}"
            )
        self.specs[spec.name] = spec
        return spec

    def load_agents(self, directory: str | Path) -> list[AgentSpec]:
        loaded = [
            self.register_agent(yaml.safe_load(path.read_text("utf-8")))
            for path in sorted(Path(directory).glob("*.y*ml"))
        ]
        for spec in loaded:  # validate the sub-agent graph once everything is loaded
            self._check_graph(spec.name, ())
        return loaded

    def _check_graph(self, name: str, stack: tuple[str, ...]) -> None:
        if name in stack:
            raise ValueError(f"sub-agent cycle: {' -> '.join((*stack, name))}")
        if name not in self.specs:
            raise KeyError(f"unknown sub-agent {name!r} (referenced by {stack[-1]!r})")
        for sub in self.specs[name].subagents:
            self._check_graph(sub, (*stack, name))

    # -- construction ----------------------------------------------------------------
    def agent(self, name: str, *, llm: LLM | None = None) -> Agent:
        self._check_graph(name, ())
        spec = self.specs[name]
        tools = self.tools.subset(spec.tools)
        for sub in spec.subagents:
            tools.add(self._subagent_tool(sub, llm))
        base = llm or self.llm
        with_hint = getattr(base, "with_hint", None)  # the router honours the tier hint
        return Agent(
            with_hint(spec.model) if callable(with_hint) and spec.model else base,
            tools,
            name=spec.name,
            system_prompt=spec.system_prompt,
            max_steps=spec.max_steps,
            memory=self.memory if spec.memory else None,
        )

    def _subagent_tool(self, name: str, llm: LLM | None):
        spec = self.specs[name]
        holder = ToolRegistry()

        async def run_subagent(
            question: Annotated[
                str, Field(min_length=2, description="하위 에이전트에게 맡길 질문")
            ],
        ) -> dict[str, Any]:
            ctx = current_run()
            if ctx.depth >= MAX_SUBAGENT_DEPTH:
                return {"error": f"sub-agent depth limit ({MAX_SUBAGENT_DEPTH}) reached"}
            sub = self.agent(name, llm=llm)
            sub.memory = None  # sub-agents never read or write the user's session
            result = await sub.run(question, patient_id=ctx.patient_id, approver=ctx.approver)
            ctx.trace.append(
                TraceEvent(
                    "subagent",
                    {
                        "agent": name,
                        "stop_reason": result.stop_reason,
                        "trace": [e.to_dict() for e in result.trace],
                    },
                    step=ctx.step,
                )
            )
            return {"answer": result.answer, "stop_reason": result.stop_reason}

        holder.register(
            run_subagent, name=f"ask_{name}", description=f"[하위 에이전트] {spec.description}"
        )
        tool = holder.get(f"ask_{name}")
        assert tool is not None
        return tool
