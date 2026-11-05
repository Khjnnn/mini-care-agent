# 사내 개발자용: 도구·skill·에이전트 추가하기

## 1. 도구(tool) 추가 — Python 함수 하나

```python
# my_team/tools.py
from typing import Annotated
from pydantic import Field
from care_agent.tools.registry import Permission, ToolRegistry


def register(registry: ToolRegistry, platform) -> None:
    @registry.register
    async def get_diet_log(
        patient_id: str,  # ← 있으면 자동으로 "환자 범위" 도구가 됨
        days: Annotated[int, Field(ge=1, le=30, description="최근 N일")] = 7,
    ) -> list[dict]:
        """최근 식단 기록을 조회한다."""  # ← docstring이 곧 LLM용 설명
        ...
```

- 입력 스키마는 타입 힌트에서 자동 생성됩니다. JSON 스키마를 직접 쓰지 마세요.
- `patient_id` 인자가 있으면 `ToolGuard`가 세션 환자 이외의 호출을 차단합니다.
- 권한 등급
  | 등급 | 의미 | 실행 조건 | MCP 노출 |
  |---|---|---|---|
  | `READ` | 조회만 | 자동 | O |
  | `MEMORY` | 에이전트 내부 상태 변경 | 자동 + trace 기록 | X |
  | `WRITE` | EHR·외부 세계 변경 | 요청마다 사람 승인(`approved_tools`) | X |
- 반환값은 요약된 dict/list로. 이름·주소·연락처 등 식별정보는 반환하지 않습니다.

플랫폼에 등록:

```bash
PLUGINS=my_team.tools care-agent serve
```

## 2. 에이전트 추가 — YAML 하나

```yaml
# agents/diet_coach.yaml
name: diet_coach
description: 식단 기록과 혈당을 함께 보고 식습관을 코칭한다
tools: [get_diet_log, get_recent_observations]
subagents: [lab_explainer]     # ask_lab_explainer 도구로 노출됨
max_steps: 6
model: small                   # 라우터 힌트
system_prompt: |
  ... 반드시 {patient_id} 자리표시자를 포함 ...
```

로딩 시 검증되는 것: 존재하지 않는 도구, `{patient_id}` 누락, 하위 에이전트 순환 참조.

## 3. 호출

```bash
curl -s localhost:8100/v1/agents/diet_coach/runs -H 'content-type: application/json' -d '{
  "question": "요즘 식단 괜찮아요?", "patient_id": "demo-001", "session_id": "s-123"
}'
```

응답의 `flags`(`emergency`, `output_modified`, `tool_denied`)와 `denied_tools`로 UI가 후속 조치(사람 승인 요청 등)를 할 수 있습니다.
