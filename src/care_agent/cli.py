"""Command-line entry point.

care-agent tools [--schema]                         # list registered tools
care-agent agents                                   # list agents from AGENTS_DIR
care-agent ask --patient demo-001 "최근 당화혈색소 수치 어때?"
care-agent ask --offline --patient demo-001 "..."   # bundled data, no Docker
care-agent serve [--offline] [--port 8100]          # HTTP API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from care_agent.config import Settings
from care_agent.fhir.client import FHIRClient
from care_agent.fhir.memory_server import InMemoryFHIR
from care_agent.llm.base import LLM
from care_agent.llm.openai_compat import OpenAICompatLLM
from care_agent.memory.store import MemoryStore
from care_agent.platform.core import Platform


def build_llm(settings: Settings) -> LLM:
    if settings.router_config:
        from care_agent.serving.router import RoutingLLM

        return RoutingLLM.from_yaml(settings.router_config)
    return OpenAICompatLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)


def build_platform(
    settings: Settings, *, offline: bool, memory_path: str | None = None
) -> Platform:
    fhir = FHIRClient(settings.fhir_base_url, transport=InMemoryFHIR() if offline else None)
    path = memory_path or settings.memory_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    platform = Platform(
        build_llm(settings), fhir, memory=MemoryStore(path), plugins=settings.plugins
    )
    platform.load_agents(settings.agents_dir)
    return platform


async def _ask(args: argparse.Namespace) -> int:
    platform = build_platform(Settings.from_env(), offline=args.offline)
    try:
        result = await platform.agent(args.agent).run(
            args.question, patient_id=args.patient, session_id=args.session
        )
    finally:
        await platform.fhir.aclose()

    print(result.answer)
    if args.trace:
        print("\n--- trace ---", file=sys.stderr)
        for event in result.trace:
            print(json.dumps(event.to_dict(), ensure_ascii=False, default=str), file=sys.stderr)
    print(
        f"\n[stop={result.stop_reason} steps={result.steps} tools={result.tool_calls}]",
        file=sys.stderr,
    )
    return 0


def _tools(args: argparse.Namespace) -> int:
    platform = build_platform(Settings.from_env(), offline=True, memory_path=":memory:")
    for tool in platform.tools:
        print(f"{tool.name}  [{tool.permission}]\n  {tool.description}")
        if args.schema:
            print("  " + json.dumps(tool.json_schema(), ensure_ascii=False))
    return 0


def _agents(_args: argparse.Namespace) -> int:
    platform = build_platform(Settings.from_env(), offline=True, memory_path=":memory:")
    for spec in platform.specs.values():
        print(f"{spec.name}: {spec.description}\n  tools={spec.tools} subagents={spec.subagents}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from care_agent.platform.api import create_app

    app = create_app(build_platform(Settings.from_env(), offline=args.offline))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="care-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask an agent a question about one patient")
    ask.add_argument("question")
    ask.add_argument("--patient", required=True, help="FHIR Patient id")
    ask.add_argument("--agent", default="care_chat")
    ask.add_argument("--session", help="session id for multi-turn memory")
    ask.add_argument("--offline", action="store_true", help="use bundled synthetic records")
    ask.add_argument("--trace", action="store_true", help="print trace events to stderr")

    tools = sub.add_parser("tools", help="list registered tools")
    tools.add_argument("--schema", action="store_true")

    sub.add_parser("agents", help="list registered agents")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--offline", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)

    args = parser.parse_args(argv)
    handlers = {
        "ask": lambda a: asyncio.run(_ask(a)),
        "tools": _tools,
        "agents": _agents,
        "serve": _serve,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
