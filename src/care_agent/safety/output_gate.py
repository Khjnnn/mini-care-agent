"""Output gate: runs on the model's final answer before it reaches the user.

1. **PII masking** – resident registration numbers, phone numbers and e-mails are masked
   even though the tools never return them (defence in depth: the user may paste them,
   memory may contain them, a future tool may leak them).
2. **Prescriptive-language check** – instructions to stop/start/change medication or a
   definitive diagnosis are not removed (rewriting medical text with regexes is unsafe)
   but flagged in the trace and followed by a fixed caution. The flag rate is tracked
   by the eval pipeline as a safety metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CAUTION = "※ 진단이나 약의 복용·중단·용량 변경은 반드시 담당 의료진과 상의하세요."

_PII = [
    ("rrn", re.compile(r"\b\d{6}\s?-\s?[1-4]\d{6}\b"), "[주민등록번호]"),
    ("phone", re.compile(r"\b01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}\b"), "[전화번호]"),
    ("phone", re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b"), "[전화번호]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[이메일]"),
]

# Only *instructions or permission* count ("중단하세요", "끊으셔도 됩니다"). A sentence that
# merely mentions stopping ("약을 끊거나 바꾸는 것은 의사와 상의하세요") is the safe answer
# and must not be flagged — the eval suite caught exactly that false positive.
_STOP = (
    r"(중단\s*(하세요|하십시오|하셔도|해도\s*(됩니다|돼요|괜찮))"
    r"|끊(으세요|으십시오|으셔도|어도\s*(됩니다|돼요|괜찮))"
    r"|그만\s*(드세요|드셔도|복용하세요|드십시오))"
)

_PRESCRIPTIVE = [
    re.compile(rf"(복용|투약|약|인슐린)[을를은는]?\s*(바로\s*)?{_STOP}"),
    re.compile(
        r"(용량|복용량|약)[을를]?\s*(두\s*배로\s*)?(늘리|늘려|줄이|줄여|올리|올려|낮추|낮춰)\S*?(세요|십시오)"
    ),
    re.compile(r"(확진|진단)(입니다|됩니다|하겠습니다|드립니다)"),
    re.compile(
        r"\b(stop|discontinue)\s+(taking\s+)?(your\s+)?(medication|insulin|metformin)", re.I
    ),
]


@dataclass
class OutputCheck:
    text: str
    masked: list[str] = field(default_factory=list)
    prescriptive: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.masked or self.prescriptive)


class OutputGate:
    def check(self, text: str) -> OutputCheck:
        out = OutputCheck(text)
        for kind, pattern, repl in _PII:
            out.text, n = pattern.subn(repl, out.text)
            out.masked += [kind] * n
        out.prescriptive = [m.group(0) for p in _PRESCRIPTIVE for m in p.finditer(out.text)]
        if out.prescriptive and CAUTION not in out.text:
            out.text = out.text.rstrip() + "\n\n" + CAUTION
        return out
