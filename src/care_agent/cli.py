"""Command-line entry point.

care-agent tools                                   # list registered tools
care-agent ask --patient demo-001 "최근 당화혈색소 수치 어때?"
care-agent ask --offline --patient demo-001 "..."  # bundled data, no Docker
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from care_agent.agent.loop import Agent
from care_agent.config import Settings
from care_agent.fhir.client import FHIRClient
from care_agent.fhir.memory_server import InMemoryFHIR
from care_agent.llm.openai_compat import OpenAICompatLLM
from care_agent.tools.fhir_tools import build_fhir_tools


def _client(settings: Settings, offline: bool) -> FHIRClient:
    return FHIRClient(settings.fhir_base_url, transport=InMemoryFHIR() if offline else None)


async def _ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    llm = OpenAICompatLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    async with _client(settings, args.offline) as fhir:
        agent = Agent(llm, build_fhir_tools(fhir), max_steps=args.max_steps)
        try:
            result = await agent.run(args.question, patient_id=args.patient)
        finally:
            await llm.aclose()

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
    registry = build_fhir_tools(_client(Settings.from_env(), offline=True))
    for tool in registry:
        print(f"{tool.name}  [{tool.permission}]\n  {tool.description}")
        if args.schema:
            print("  " + json.dumps(tool.json_schema(), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="care-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask the agent a question about one patient")
    ask.add_argument("question")
    ask.add_argument("--patient", required=True, help="FHIR Patient id")
    ask.add_argument("--offline", action="store_true", help="use bundled synthetic records")
    ask.add_argument("--trace", action="store_true", help="print trace events to stderr")
    ask.add_argument("--max-steps", type=int, default=6)

    tools = sub.add_parser("tools", help="list registered tools")
    tools.add_argument("--schema", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "ask":
        return asyncio.run(_ask(args))
    return _tools(args)


if __name__ == "__main__":
    raise SystemExit(main())
