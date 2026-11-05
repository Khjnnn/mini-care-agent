# mini-care-agent

Safety-first healthcare agent harness over synthetic FHIR R4 data. Roadmap and design rationale: README.md.

## Commands
- `uv sync` — install
- `uv run pytest -q` — tests (no Docker or LLM needed; uses `InMemoryFHIR` + `ScriptedLLM`)
- `uv run ruff check . && uv run ruff format .` — lint/format (CI runs both with `--check`)

## Conventions
- Tools are typed `async def`s registered on a `ToolRegistry`; never hand-write JSON schemas.
- Tools touching one patient's data must take `patient_id` (that is how `ToolGuard` scopes them).
- Tools return summaries from `fhir/summaries.py`, never raw FHIR resources, and never names/addresses/telecom.
- Anything that can change state is `Permission.WRITE` and is not exported over MCP.
- Errors inside the agent loop go back to the model as tool results and into the trace; don't raise.
- Only synthetic data in the repo. Code, docstrings and comments in English; user-facing strings and prompts in Korean.

## Layout
- `agent/` loop + trace · `tools/` registry + FHIR tools · `safety/` input/tool/output gates · `memory/`
- `platform/` agent specs (YAML in `agents/`), plugins, FastAPI · `documents/` lab-report structuring
- `evals/` suite (`datasets/agent_eval`), judge, runner, analysis, compare/drift · `observability/` Langfuse
- `serving/` PHI-aware router, gateway, bench · `deploy/` vLLM / GKE / router config

## Rules added later
- Agents with patient-scoped tools must have `{patient_id}` in their prompt; agents without them must not get it.
- PHI routing (`serving/router.py`) may only ever get stricter; never add a fallback from PHI to a non-`phi_allowed` backend.
- Behaviour changes to gates or prompts: run `uv run care-agent eval run --offline --llm heuristic-careful` and keep the CI drift check green (refresh `evals/baselines/` only on intended changes).
