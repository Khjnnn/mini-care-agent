# 내 PC에서 실제 모델로 평가하기

클라우드 개발 환경에서는 모델 가중치를 받을 수 없어서, 실제 모델 평가는 로컬 PC에서 돌립니다.
스크립트 하나가 **에이전트 회귀 평가(+Judge) → 기준선 비교 → 검사결과지 추출 정확도 → 지연·처리량 벤치마크**를 차례로 실행하고 `results/<라벨>/`에 모읍니다.
데이터는 저장소에 들어 있는 합성 환자만 쓰기 때문에 결과를 그대로 커밋해도 안전합니다.

## 1. 준비 (한 번만)

1. [uv](https://docs.astral.sh/uv/getting-started/installation/)와 [Ollama](https://ollama.com/download)를 설치합니다.
2. 모델을 받습니다. 메모리에 맞춰 고르세요.

   | PC 메모리 / GPU | 상담 모델 | Judge 모델 (선택) |
   |---|---|---|
   | 16GB RAM, GPU 없음 | `qwen2.5:3b` | — (같은 모델로 자기 채점) |
   | 16GB+ RAM 또는 8GB+ VRAM | `qwen2.5:7b` | `qwen2.5:14b` |
   | 24GB+ VRAM / Apple M 시리즈 32GB+ | `qwen2.5:14b` | `qwen2.5:32b` |

   ```bash
   ollama pull qwen2.5:7b
   ```

3. 저장소를 받습니다.

   ```bash
   git clone https://github.com/Khjnnn/mini-care-agent && cd mini-care-agent
   ```

## 2. 실행

**macOS / Linux / WSL**

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b LLM_API_KEY=ollama
```

```bash
export JUDGE_MODEL=qwen2.5:14b    # 선택: 다른(더 큰) 모델로 채점하면 자기 채점 편향이 줄어듭니다
```

```bash
./scripts/run_real_eval.sh
```

**Windows PowerShell**

```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"; $env:LLM_MODEL = "qwen2.5:7b"; $env:LLM_API_KEY = "ollama"
```

```powershell
.\scripts\run_real_eval.ps1
```

실행 스크립트가 막히면 `Set-ExecutionPolicy -Scope Process Bypass`를 먼저 실행하세요.

옵션: `CONCURRENCY=1`(노트북이 느릴 때), `SKIP_BENCH=1`, `SKIP_JUDGE=1`.
7B 모델 기준 GPU가 있으면 몇 분, CPU만 있으면 30분 이상 걸릴 수 있습니다.

## 3. 결과 공유

```bash
git add results/ && git commit -m "Add real-model eval results" && git push
```

푸시하면 결과 분석, 실패 사례 해석, README 수치 반영을 이어서 진행합니다.

## 결과 파일

| 파일 | 내용 |
|---|---|
| `agent-eval.md` | 카테고리별 통과율, Judge 평균, 실패 유형별 사례 |
| `agent-eval.json` | 케이스별 원본(답변, 도구 호출, 체크, Judge 점수) |
| `compare-vs-heuristic.json` | 규칙 기반 기준선 대비 퇴행/개선 케이스 |
| `docs-eval-summary.json` | 검사결과지 항목 precision/recall, 값·단위·판정 정확도, 이슈 탐지율 |
| `bench.md` | 동시성별 TTFT, ITL, E2E, 처리량 |
| `environment.txt` | 모델, Judge, OS, GPU |

## 문제 해결

- 동시성 벤치마크에서 처리량이 늘지 않음: Ollama는 기본적으로 요청을 순차 처리해요. 병렬 처리를 보려면 Ollama를 `OLLAMA_NUM_PARALLEL=4`로 다시 시작하세요(VRAM이 더 필요해요). 운영 환경 수준의 측정은 vLLM으로 해야 해요.

- `cannot reach .../models`: `ollama serve`가 실행 중인지 확인하세요(`ollama list`가 동작해야 함).
- `judge_failures`가 많음: Judge 모델이 JSON 형식을 못 맞추는 것입니다. 더 큰 모델을 `JUDGE_MODEL`로 지정하세요.
- 도구를 전혀 호출하지 않음: 해당 모델이 tool calling을 지원하는지 확인하세요(qwen2.5, llama3.1+ 지원).
