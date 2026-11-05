"""Answer self-checks that give the model one chance to fix a known failure mode.

Both checks come from the first real-model eval (qwen2.5:7b, results/qwen2.5-7b):

* ``language`` – the model answered a Korean question in Chinese ("二甲双胍片 500毫克 …")
  or drifted into Chinese mid-answer. Patients cannot read it, and the eval's name
  checks fail even though the facts were right.
* ``unfulfilled_tool_intent`` – the model said it would look something up
  ("최근 혈압 수치를 확인해볼게요") and then ended the turn without calling a tool, so
  the user got no answer at all.

Each nudge fires at most once per run, is recorded in the trace, and is phrased as a
harness message, so it is cheap and visible rather than an open-ended retry loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HAN = re.compile(r"[一-鿿]")
_HANGUL = re.compile(r"[가-힣]")
_PROMISE = re.compile(
    r"(확인|조회|찾아|알아|살펴)\s*(해|봐)?\s*"
    r"(볼게요|볼께요|보겠습니다|볼게|드릴게요|드리겠습니다|보도록\s*하겠습니다)"
)

LANGUAGE_MSG = (
    "[하네스 확인] 답변에 중국어(한자)가 섞여 있습니다. 같은 내용을 한국어로만 다시 작성하세요. "
    "약 이름은 도구 결과의 영문 원문 그대로 쓰거나 한글로 표기하세요."
)
TOOL_INTENT_MSG = (
    "[하네스 확인] 조회하겠다고 말했지만 도구를 호출하지 않았습니다. "
    "필요한 도구를 지금 호출하고, 그 결과를 근거로 답하세요. 조회가 필요 없다면 바로 답하세요."
)


@dataclass(frozen=True)
class Nudge:
    kind: str
    message: str
    evidence: str


def check_answer(answer: str, *, tools_available: bool, tools_attempted: bool) -> Nudge | None:
    han = _HAN.findall(answer)
    hangul = len(_HANGUL.findall(answer))
    if len(han) >= 2 and len(han) > 0.1 * (len(han) + hangul):
        return Nudge("language", LANGUAGE_MSG, "".join(han[:12]))
    if tools_available and not tools_attempted:
        m = _PROMISE.search(answer)
        if m:
            return Nudge("unfulfilled_tool_intent", TOOL_INTENT_MSG, m.group(0))
    return None
