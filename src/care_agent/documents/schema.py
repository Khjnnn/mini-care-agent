"""Schemas for lab-report structuring.

Two layers on purpose:

* ``Extracted*`` – what the model must transcribe from the page, *as printed*
  (strings, no interpretation). Keeping the model's job to transcription makes it
  checkable and keeps medical logic out of the prompt.
* ``LabResult`` / ``StructuredLabReport`` – the result of deterministic normalisation:
  LOINC mapping, numeric parsing, unit conversion, reference ranges, flags and the
  issues that require a human to review before anything is written to the EHR.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractedLabItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="검사 항목명 (결과지에 인쇄된 그대로)")
    value: str = Field(min_length=1, description="결과값 (인쇄된 그대로, 예: '7.8', '<5', '1,234')")
    unit: str | None = Field(default=None, description="단위 (인쇄된 그대로)")
    reference_range: str | None = Field(default=None, description="참고치 (예: '4.0~5.6', '<200')")
    flag: str | None = Field(default=None, description="판정 표시 (예: 'H', 'L', '▲', '높음')")


class ExtractedLabReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_date: date | None = Field(default=None, description="검사일 (YYYY-MM-DD)")
    institution: str | None = Field(default=None, description="검사기관명")
    items: list[ExtractedLabItem] = Field(min_length=1)


Interpretation = Literal["H", "L", "N"]

IssueCode = Literal[
    "unmapped_test",
    "non_numeric_value",
    "unit_converted",
    "unit_mismatch",
    "implausible_value",
    "flag_mismatch",
    "missing_reference_range",
    "missing_report_date",
    "duplicate_test",
    "value_not_in_source",
]

#: Issues that block automatic saving; others are informational.
BLOCKING_ISSUES: frozenset[str] = frozenset(
    {
        "unmapped_test",
        "non_numeric_value",
        "unit_mismatch",
        "implausible_value",
        "flag_mismatch",
        "missing_report_date",
        "duplicate_test",
        "value_not_in_source",
    }
)


class Issue(BaseModel):
    code: IssueCode
    message: str
    item: str | None = None


class LabResult(BaseModel):
    raw_name: str
    name: str
    loinc: str | None
    value: float | None
    value_text: str
    unit: str | None
    ref_low: float | None = None
    ref_high: float | None = None
    interpretation: Interpretation | None = None


class StructuredLabReport(BaseModel):
    report_date: date | None
    institution: str | None
    results: list[LabResult]
    issues: list[Issue] = Field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return any(i.code in BLOCKING_ISSUES for i in self.issues)
