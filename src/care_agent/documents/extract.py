"""Image/OCR text → ``ExtractedLabReport`` with schema validation and self-correction.

Two input paths behind one class:

* **VLM** – the page image goes straight to a vision-language model
  (OpenAI-compatible ``image_url`` content part; works with vLLM-served Qwen2.5-VL etc.).
* **OCR + LLM** – an ``OCRBackend`` turns the image into text first (Tesseract here; a
  commercial OCR API in production), then a text LLM transcribes it into the schema.

Either way, the model's JSON is validated against the pydantic schema; on failure the
validation errors are sent back and the model retries (bounded). Every attempt is
recorded so failures can be analysed later (week 4).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from care_agent.documents.schema import ExtractedLabReport
from care_agent.llm.base import LLM, Message
from care_agent.llm.jsonout import parse_json_object

SYSTEM = """\
당신은 한국어 건강검진·혈액검사 결과지를 그대로 옮겨 적는 전사(transcription) 도구입니다.
- 결과지에 인쇄된 값만 옮기세요. 계산·추정·해석하지 마세요.
- 값, 단위, 참고치, 판정 표시(H/L/▲ 등)는 인쇄된 문자열 그대로 쓰세요.
- 환자 이름, 주민등록번호, 연락처 등 개인정보는 옮기지 마세요.
- 아래 JSON 스키마를 만족하는 JSON 객체 하나만 출력하세요. 설명이나 코드블록 없이 JSON만 출력합니다.

JSON 스키마:
{schema}
"""


class OCRBackend(Protocol):
    async def read(self, image: bytes) -> str: ...


class TesseractOCR:
    """Local OCR via the ``tesseract`` CLI (``apt install tesseract-ocr tesseract-ocr-kor``)."""

    def __init__(self, lang: str = "kor+eng") -> None:
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract is not installed")
        self.lang = lang

    async def read(self, image: bytes) -> str:
        proc = await asyncio.create_subprocess_exec(
            "tesseract",
            "stdin",
            "stdout",
            "-l",
            self.lang,
            "--psm",
            "6",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(image)
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract failed: {err.decode(errors='replace')[:200]}")
        return out.decode("utf-8", errors="replace")


@dataclass
class Attempt:
    raw: str
    error: str | None


@dataclass
class ExtractionResult:
    report: ExtractedLabReport | None
    attempts: list[Attempt] = field(default_factory=list)
    source: str = "text"

    @property
    def ok(self) -> bool:
        return self.report is not None


class LabReportExtractor:
    def __init__(self, llm: LLM, *, ocr: OCRBackend | None = None, max_attempts: int = 3) -> None:
        self.llm = llm
        self.ocr = ocr
        self.max_attempts = max_attempts

    async def from_text(self, text: str) -> ExtractionResult:
        user: Message = {
            "role": "user",
            "content": f"다음은 검사결과지 OCR 텍스트입니다.\n\n{text}",
        }
        return await self._extract(user, source="text")

    async def from_image(self, image: bytes, mime: str = "image/png") -> ExtractionResult:
        if self.ocr is not None:
            result = await self.from_text(await self.ocr.read(image))
            result.source = "ocr+llm"
            return result
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        user: Message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "다음 검사결과지 이미지를 스키마에 맞게 옮겨 주세요."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
        return await self._extract(user, source="vlm")

    async def _extract(self, user: Message, *, source: str) -> ExtractionResult:
        schema = json.dumps(ExtractedLabReport.model_json_schema(), ensure_ascii=False)
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM.format(schema=schema)},
            user,
        ]
        result = ExtractionResult(report=None, source=source)

        for _ in range(self.max_attempts):
            resp = await self.llm.complete(messages, [])
            raw = resp.content or ""
            try:
                report = ExtractedLabReport.model_validate(parse_json_object(raw))
            except (ValueError, ValidationError) as e:
                error = _describe(e)
                result.attempts.append(Attempt(raw, error))
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"출력이 스키마를 만족하지 않습니다. 오류:\n{error}\n"
                        "오류를 고친 JSON 객체 하나만 다시 출력하세요.",
                    },
                ]
                continue
            result.attempts.append(Attempt(raw, None))
            result.report = report
            return result
        return result


def _describe(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "\n".join(
            f"- {'.'.join(map(str, err['loc']))}: {err['msg']}"
            for err in e.errors(include_url=False)
        )
    return f"- {e}"
