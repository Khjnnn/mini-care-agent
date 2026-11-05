# Windows PowerShell version of scripts/run_real_eval.sh
#
#   $env:LLM_BASE_URL = "http://localhost:11434/v1"; $env:LLM_MODEL = "qwen2.5:7b"; $env:LLM_API_KEY = "ollama"
#   .\scripts\run_real_eval.ps1
#
# Optional: $env:JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY, CONCURRENCY, SKIP_BENCH=1, SKIP_JUDGE=1
param([string]$Label)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
# Korean text in JSON/markdown output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not $env:LLM_BASE_URL -or -not $env:LLM_MODEL) { throw "set LLM_BASE_URL and LLM_MODEL first" }
if (-not $Label) { $Label = $env:LLM_MODEL -replace '[/: ]', '-' }
$Out = "results/$Label"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$key = if ($env:LLM_API_KEY) { $env:LLM_API_KEY } else { "EMPTY" }
$concurrency = if ($env:CONCURRENCY) { $env:CONCURRENCY } else { "2" }

Write-Host "> checking the endpoint ($env:LLM_BASE_URL)"
try { Invoke-RestMethod -TimeoutSec 10 -Headers @{ Authorization = "Bearer $key" } "$env:LLM_BASE_URL/models" | Out-Null }
catch { throw "cannot reach $env:LLM_BASE_URL/models - is the server running?" }

uv sync -q
$judge = @("--judge", "judge-endpoint")
if ($env:SKIP_JUDGE -eq "1") { $judge = @() }

function Save($cmd, $file) { & uv run care-agent @cmd | Out-File -Encoding utf8 $file; if ($LASTEXITCODE) { throw "failed: $cmd" } }

Write-Host "> 1/4 agent eval suite (+ judge)"
Save (@("eval", "run", "--offline") + $judge + @("--label", $Label, "--concurrency", $concurrency, "--out", "$Out/agent-eval.json", "--report", "$Out/agent-eval.md")) "$Out/agent-eval-summary.json"
Write-Host "> 2/4 compare with the committed heuristic baseline"
Save @("eval", "compare", "evals/baselines/heuristic-careful.json", "$Out/agent-eval.json") "$Out/compare-vs-heuristic.json"
Write-Host "> 3/4 lab report extraction accuracy"
Save @("docs", "eval", "--out", "$Out/docs-eval.json") "$Out/docs-eval-summary.json"
if ($env:SKIP_BENCH -ne "1") {
    Write-Host "> 4/4 latency / throughput benchmark"
    Save @("bench", "--base-url", $env:LLM_BASE_URL, "--model", $env:LLM_MODEL, "--concurrency", "1", "2", "4", "--requests", "8", "--max-tokens", "200", "--out", "$Out/bench.json") "$Out/bench.md"
}

$judgeModel = if ($env:JUDGE_MODEL) { $env:JUDGE_MODEL } else { "$env:LLM_MODEL (self-judged)" }
$gpu = try { (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join ";" } catch { "n/a" }
@(
    "label: $Label", "model: $env:LLM_MODEL", "judge: $judgeModel",
    "date: $(Get-Date -Format o)", "os: Windows $([Environment]::OSVersion.Version)", "gpu: $gpu"
) | Out-File -Encoding utf8 "$Out/environment.txt"

Get-Content "$Out/agent-eval-summary.json"
Write-Host ""
Write-Host "done - results in $Out. Share them with:"
Write-Host "  git add $Out; git commit -m `"Add real-model eval results ($Label)`"; git push"
