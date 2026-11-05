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
