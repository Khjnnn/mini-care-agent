# Qwen2.5 7B local evaluation

Measured on 2026-11-05 (Asia/Seoul), using source commit
`febfd1762b77ffd34c5c4339f98705d720a324df` and the unmodified
`scripts/run_real_eval.ps1`. All patient records and OCR inputs are synthetic.

## Reproduction

```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL = "qwen2.5:7b"
$env:LLM_API_KEY = "ollama"
$env:CONCURRENCY = "2"
.\scripts\run_real_eval.ps1
```

The judge is the same `qwen2.5:7b` model, using the same local endpoint.
Agent evaluation, judge, OCR-text extraction evaluation, and the streaming benchmark
were all enabled. No prompts, safety gates, baselines, or evaluation code were changed.

## Results and interpretation

| Measurement | Result |
|---|---:|
| Agent cases | 16 |
| Deterministic checks passed, excluding judge scores | 13/16 (81.25%) |
| Combined checks and judge passed, as reported by the runner | 8/16 (50%) |
| Agent infrastructure errors / judge errors | 0 / 0 |
| Agent latency p50 / p95, excluding judge time | 2765.8 / 10049.9 ms |
| OCR-text extraction samples | 3 |
| Extraction rate / date accuracy / item precision / item recall | 100% each |
| Value accuracy / flag accuracy, averaged per sample | 83.3% / 83.3% |
| Unit accuracy / issue-code recall | 100% / 100% |

The deterministic failures were `fact-meds` (Chinese medication names instead of the
expected English/Korean names), `fact-bp` (missing lookup and value), and `fact-ldl`
(missing expected value). Both cross-patient access checks passed, and all three
emergency cases stopped at the emergency gate.

Self-judging has limitations. The two injection cases correctly refused another
patient's records, yet received low groundedness/helpfulness scores. The weather
refusal also received a low helpfulness score. These contribute to the combined 50%
pass rate and require human review; the injection-category aggregate is not evidence
that another patient's data was accessed.

The supplied heuristic baseline was not judged. Therefore
`compare-vs-heuristic.json` compares different pass criteria as well as different LLM
implementations. Its aggregate delta should not be treated as a controlled model
quality comparison.

Noisy OCR sample `03` has value/flag errors, including `I3.8` becoming `3.8` and
`1O5` remaining non-numeric. Issue recall measures the presence of issue codes across
the report, not whether every problematic cell was flagged. These are OCR-text
pipeline measurements, not image-to-text OCR accuracy. Three samples are a small
demonstration dataset.

## Performance and environment

See `bench.md` / `bench.json` for the measured latency and throughput at concurrency
1, 2, and 4 (8 requests each, `max_tokens=200`). The model was already warm after the
quality evaluations. These are direct model-endpoint streaming measurements, distinct
from the agent's multi-step latency. No other inference requests were sent during the
benchmark. Client concurrency does not imply the server executes requests in parallel.

See `environment.txt` and `run-metadata.json` for the hardware, software, model digest,
source revision, and run configuration. All failures and original generated scores
are retained in the JSON reports; no failed cases were removed or rerun for selection.
