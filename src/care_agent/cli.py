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
import os
import sys
from pathlib import Path

import httpx

from care_agent.config import Settings
from care_agent.fhir.client import FHIRClient
from care_agent.fhir.memory_server import InMemoryFHIR
from care_agent.llm.base import LLM
from care_agent.llm.openai_compat import OpenAICompatLLM
from care_agent.memory.store import MemoryStore
from care_agent.platform.core import Platform

LLM_CHOICES = ["endpoint", "judge-endpoint", "heuristic-careful", "heuristic-naive"]


def build_llm(settings: Settings, choice: str = "endpoint") -> LLM:
    """``endpoint`` = LLM_BASE_URL (or the router if ROUTER_CONFIG is set);
    ``judge-endpoint`` = JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY — a different model for
    judging, to avoid a model grading its own answers (falls back to LLM_* when unset);
    ``heuristic-*`` = keyword-rule stand-in for offline demos (see llm/heuristic.py)."""
    if choice == "judge-endpoint":
        return OpenAICompatLLM(
            os.getenv("JUDGE_BASE_URL") or settings.llm_base_url,
            os.getenv("JUDGE_MODEL") or settings.llm_model,
            os.getenv("JUDGE_API_KEY") or settings.llm_api_key,
        )
    if choice.startswith("heuristic-"):
        from care_agent.llm.heuristic import HeuristicLLM

        return HeuristicLLM(choice.removeprefix("heuristic-"))  # type: ignore[arg-type]
    if settings.router_config:
        from care_agent.serving.router import RoutingLLM

        return RoutingLLM.from_yaml(settings.router_config)
    return OpenAICompatLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)


def build_platform(
    settings: Settings,
    *,
    offline: bool,
    memory_path: str | None = None,
    llm: str = "endpoint",
) -> Platform:
    fhir = FHIRClient(settings.fhir_base_url, transport=InMemoryFHIR() if offline else None)
    path = memory_path or settings.memory_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    platform = Platform(
        build_llm(settings, llm), fhir, memory=MemoryStore(path), plugins=settings.plugins
    )
    platform.load_agents(settings.agents_dir)
    if settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key:
        from care_agent.observability.langfuse import LangfuseExporter

        platform.exporter = LangfuseExporter(
            settings.langfuse_host, settings.langfuse_public_key, settings.langfuse_secret_key
        )
    return platform


async def _ask(args: argparse.Namespace) -> int:
    platform = build_platform(Settings.from_env(), offline=args.offline, llm=args.llm)
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

    app = create_app(build_platform(Settings.from_env(), offline=args.offline, llm=args.llm))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _eval(args: argparse.Namespace) -> int:
    from care_agent.evals.analysis import markdown_report
    from care_agent.evals.compare import compare_runs, detect_drift
    from care_agent.evals.runner import RunFile

    if args.eval_cmd == "run":
        run = asyncio.run(_eval_run(args))
        run.save(args.out)
        print(json.dumps(run.meta["summary"], ensure_ascii=False, indent=2))
        if args.report:
            Path(args.report).write_text(markdown_report(run), "utf-8")
        return 0
    if args.eval_cmd == "report":
        text = markdown_report(RunFile.load(args.run))
        if args.out:
            Path(args.out).write_text(text, "utf-8")
        print(text)
        return 0
    if args.eval_cmd == "compare":
        result = compare_runs(RunFile.load(args.a), RunFile.load(args.b))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    drift = detect_drift(
        RunFile.load(args.baseline),
        RunFile.load(args.current),
        max_latency_increase=args.max_latency_increase,
    )
    print(
        json.dumps(
            {
                "drifted": drift.drifted,
                "psi": drift.psi,
                "alerts": [a.__dict__ for a in drift.alerts],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 1 if drift.drifted else 0  # non-zero exit so CI / cron can alert


async def _eval_run(args: argparse.Namespace):  # -> RunFile
    from care_agent.evals.cases import load_cases
    from care_agent.evals.judge import LLMJudge
    from care_agent.evals.runner import EvalRunner

    settings = Settings.from_env()
    platform = build_platform(settings, offline=args.offline, memory_path=":memory:", llm=args.llm)
    judge = LLMJudge(build_llm(settings, args.judge)) if args.judge else None
    on_result = None
    if platform.exporter is not None:
        exporter = platform.exporter

        async def on_result(case, result):  # noqa: ANN001, ANN202
            await exporter.export(
                result,
                name=f"eval:{case.id}",
                question=case.question,
                tags=["eval", args.label, case.category],
            )

    runner = EvalRunner(
        lambda _case: platform.agent(args.agent),
        judge=judge,
        concurrency=args.concurrency,
        on_result=on_result,
    )
    cases = await asyncio.to_thread(load_cases, args.dataset)
    meta = {
        "agent": args.agent,
        "llm": args.llm,
        "model": settings.llm_model if args.llm == "endpoint" else args.llm,
        "judge_model": (os.getenv("JUDGE_MODEL") or settings.llm_model)
        if args.judge in ("endpoint", "judge-endpoint")
        else args.judge,
    }
    return await runner.run(cases, label=args.label, meta=meta)


def _gateway(args: argparse.Namespace) -> int:
    import uvicorn

    from care_agent.serving.gateway import create_gateway
    from care_agent.serving.router import RoutingLLM

    uvicorn.run(create_gateway(RoutingLLM.from_yaml(args.config)), host=args.host, port=args.port)
    return 0


def _bench(args: argparse.Namespace) -> int:

    from care_agent.serving.bench import markdown_table, run_bench

    prompts = None
    if args.prompts:
        lines = Path(args.prompts).read_text("utf-8").splitlines()
        prompts = [json.loads(line)["prompt"] for line in lines if line.strip()]
    result = asyncio.run(
        run_bench(
            args.base_url,
            args.model,
            concurrency_levels=args.concurrency,
            requests=args.requests,
            max_tokens=args.max_tokens,
            prompts=prompts,
            api_key=os.getenv(args.api_key_env, "EMPTY") if args.api_key_env else "EMPTY",
        )
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(markdown_table(result))
    return 0


_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _docs(args: argparse.Namespace) -> int:
    from care_agent.documents.evaluate import evaluate_dataset
    from care_agent.documents.pipeline import outcome_to_dict

    platform = build_platform(Settings.from_env(), offline=True, memory_path=":memory:")
    if args.docs_cmd == "eval":
        report = asyncio.run(evaluate_dataset(platform.lab_reports, args.dataset))
        if args.out:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    path = Path(args.file)
    mime = _IMAGE_MIME.get(path.suffix.lower())
    if mime:
        job = platform.lab_reports.extract(image=path.read_bytes(), mime=mime)
    else:
        job = platform.lab_reports.extract(text=path.read_text("utf-8"))
    print(json.dumps(outcome_to_dict(asyncio.run(job)), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="care-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    docs = sub.add_parser("docs", help="lab report structuring")
    docs_sub = docs.add_subparsers(dest="docs_cmd", required=True)
    extract = docs_sub.add_parser("extract", help="structure one report (image or OCR text)")
    extract.add_argument("file")
    ev = docs_sub.add_parser("eval", help="field-level accuracy on a labelled dataset")
    ev.add_argument("dataset", nargs="?", default="datasets/lab_reports")
    ev.add_argument("--out", help="write full per-sample report as JSON")

    ask = sub.add_parser("ask", help="ask an agent a question about one patient")
    ask.add_argument("question")
    ask.add_argument("--patient", required=True, help="FHIR Patient id")
    ask.add_argument("--agent", default="care_chat")
    ask.add_argument("--session", help="session id for multi-turn memory")
    ask.add_argument("--offline", action="store_true", help="use bundled synthetic records")
    ask.add_argument("--trace", action="store_true", help="print trace events to stderr")
    ask.add_argument("--llm", choices=LLM_CHOICES, default="endpoint")

    ev_parser = sub.add_parser("eval", help="agent evaluation: run, report, compare, drift")
    ev_sub = ev_parser.add_subparsers(dest="eval_cmd", required=True)
    run = ev_sub.add_parser("run", help="run the eval suite")
    run.add_argument("--dataset", default="datasets/agent_eval/cases.yaml")
    run.add_argument("--agent", default="care_chat")
    run.add_argument("--llm", choices=LLM_CHOICES, default="endpoint")
    run.add_argument("--judge", choices=LLM_CHOICES, help="enable LLM-as-a-Judge with this LLM")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--offline", action="store_true")
    run.add_argument("--label", default="run")
    run.add_argument("--out", default="runs/latest.json")
    run.add_argument("--report", help="also write a markdown report here")
    rep = ev_sub.add_parser("report", help="markdown report with failure analysis")
    rep.add_argument("run")
    rep.add_argument("--out")
    cmp_ = ev_sub.add_parser("compare", help="A/B compare two run files")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    dr = ev_sub.add_parser("drift", help="drift check; exit 1 when drifted")
    dr.add_argument("baseline")
    dr.add_argument("current")
    dr.add_argument(
        "--max-latency-increase",
        type=float,
        default=0.3,
        help="relative p95 increase that alerts (use a large value on shared CI runners)",
    )

    tools = sub.add_parser("tools", help="list registered tools")
    tools.add_argument("--schema", action="store_true")

    sub.add_parser("agents", help="list registered agents")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--offline", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--llm", choices=LLM_CHOICES, default="endpoint")

    gw = sub.add_parser("gateway", help="OpenAI-compatible LLM gateway with PHI-aware routing")
    gw.add_argument("--config", default="deploy/router.example.yaml")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8200)

    bench = sub.add_parser(
        "bench", help="TTFT/ITL/throughput benchmark of an OpenAI-compatible endpoint"
    )
    bench.add_argument("--base-url", required=True)
    bench.add_argument("--model", required=True)
    bench.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    bench.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--prompts", help='JSONL with {"prompt": ...} per line')
    bench.add_argument("--api-key-env", help="env var holding the API key")
    bench.add_argument("--out", help="write raw results as JSON")

    args = parser.parse_args(argv)
    handlers = {
        "ask": lambda a: asyncio.run(_ask(a)),
        "tools": _tools,
        "agents": _agents,
        "serve": _serve,
        "docs": _docs,
        "eval": _eval,
        "gateway": _gateway,
        "bench": _bench,
    }
    try:
        return handlers[args.cmd](args)
    except httpx.ConnectError as e:
        print(
            f"error: could not reach {e.request.url.host}:{e.request.url.port} — is the LLM/FHIR "
            "endpoint running? (LLM_BASE_URL / FHIR_BASE_URL, or --offline / --llm heuristic-*)",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
