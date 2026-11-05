#!/usr/bin/env bash
# Run every real-model evaluation in one go and collect the results under results/<label>/.
#
#   export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b LLM_API_KEY=ollama
#   ./scripts/run_real_eval.sh                 # label defaults to the model name
#
# Optional:
#   JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY  separate judge model (recommended: a larger
#                                                 one; a model grading itself is biased)
#   CONCURRENCY=2      parallel eval cases (keep low on a laptop)
#   SKIP_BENCH=1       skip the latency benchmark
#   SKIP_JUDGE=1       skip LLM-as-a-Judge
#
# Everything runs against the bundled synthetic patients (--offline = in-memory FHIR);
# no real patient data is involved, so the results are safe to commit.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LLM_BASE_URL:?set LLM_BASE_URL (e.g. http://localhost:11434/v1)}"
: "${LLM_MODEL:?set LLM_MODEL (e.g. qwen2.5:7b)}"
LABEL="${1:-$(echo "$LLM_MODEL" | tr '/: ' '---')}"
OUT="results/$LABEL"
mkdir -p "$OUT"

echo "▶ checking the endpoint ($LLM_BASE_URL)"
curl -fsS -m 10 -H "Authorization: Bearer ${LLM_API_KEY:-EMPTY}" "$LLM_BASE_URL/models" >/dev/null \
  || { echo "✗ cannot reach $LLM_BASE_URL/models — is the server running?"; exit 1; }

uv sync -q

JUDGE_ARGS=(--judge judge-endpoint)
[[ "${SKIP_JUDGE:-0}" == 1 ]] && JUDGE_ARGS=()

echo "▶ 1/4 agent eval suite (+ judge)"
uv run care-agent eval run --offline ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"} --label "$LABEL" \
  --concurrency "${CONCURRENCY:-2}" --out "$OUT/agent-eval.json" --report "$OUT/agent-eval.md" \
  > "$OUT/agent-eval-summary.json"

echo "▶ 2/4 compare with the committed heuristic baseline"
uv run care-agent eval compare evals/baselines/heuristic-careful.json "$OUT/agent-eval.json" \
  > "$OUT/compare-vs-heuristic.json"

echo "▶ 3/4 lab report extraction accuracy"
uv run care-agent docs eval --out "$OUT/docs-eval.json" > "$OUT/docs-eval-summary.json"

if [[ "${SKIP_BENCH:-0}" != 1 ]]; then
  echo "▶ 4/4 latency / throughput benchmark"
  uv run care-agent bench --base-url "$LLM_BASE_URL" --model "$LLM_MODEL" \
    --concurrency 1 2 4 --requests 8 --max-tokens 200 --out "$OUT/bench.json" > "$OUT/bench.md"
fi

{
  echo "label: $LABEL"
  echo "model: $LLM_MODEL"
  echo "judge: ${JUDGE_MODEL:-$LLM_MODEL (self-judged)}"
  echo "date: $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "os: $(uname -srm)"
  command -v nvidia-smi >/dev/null && echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
  [[ "$(uname)" == Darwin ]] && echo "chip: $(sysctl -n machdep.cpu.brand_string) / $(($(sysctl -n hw.memsize) / 1073741824)) GB"
  command -v ollama >/dev/null && echo "ollama: $(ollama --version 2>/dev/null | tail -1)"
} > "$OUT/environment.txt" 2>/dev/null || true

echo
cat "$OUT/agent-eval-summary.json"
echo
echo "✓ results in $OUT — share them with:"
echo "  git add $OUT && git commit -m \"Add real-model eval results ($LABEL)\" && git push"
