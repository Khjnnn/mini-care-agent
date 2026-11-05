"""Deterministic normalisation of a transcribed lab report."""

from __future__ import annotations

import re

from care_agent.documents.schema import (
    ExtractedLabItem,
    ExtractedLabReport,
    Interpretation,
    Issue,
    IssueCode,
    LabResult,
    StructuredLabReport,
)
from care_agent.knowledge.lab_catalog import LabTest, by_loinc, lookup

_NUM = r"[-+]?\d+(?:\.\d+)?"

# Canonical spelling for units as they show up on Korean lab reports / OCR output.
_UNIT_ALIASES = {
    "mg/dl": "mg/dL",
    "mg/dℓ": "mg/dL",
    "mg%": "mg/dL",
    "g/dl": "g/dL",
    "g/dℓ": "g/dL",
    "mmol/l": "mmol/L",
    "umol/l": "umol/L",
    "μmol/l": "umol/L",
    "µmol/l": "umol/L",
    "mmol/mol": "mmol/mol",
    "%": "%",
    "u/l": "U/L",
    "iu/l": "U/L",
    "ml/min/1.73m2": "mL/min/1.73m2",
    "ml/min/1.73㎡": "mL/min/1.73m2",
    "ml/min/1.73m^2": "mL/min/1.73m2",
    "ml/min": "mL/min/1.73m2",
}

# (LOINC, from-unit) -> factor or function to the catalogue unit.
_CONVERSIONS = {
    ("1558-6", "mmol/L"): lambda v: v * 18.016,
    ("2093-3", "mmol/L"): lambda v: v * 38.67,
    ("18262-6", "mmol/L"): lambda v: v * 38.67,
    ("2085-9", "mmol/L"): lambda v: v * 38.67,
    ("2571-8", "mmol/L"): lambda v: v * 88.57,
    ("2160-0", "umol/L"): lambda v: v / 88.4,
    ("4548-4", "mmol/mol"): lambda v: 0.0915 * v + 2.15,  # IFCC → NGSP
}

_HIGH_FLAGS = {"h", "high", "▲", "↑", "높음", "상"}
_LOW_FLAGS = {"l", "low", "▼", "↓", "낮음", "하"}
_NORMAL_FLAGS = {"n", "normal", "정상", ""}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None or not unit.strip():
        return None
    key = re.sub(r"\s+", "", unit).lower()
    return _UNIT_ALIASES.get(key, unit.strip())


def parse_value(text: str) -> float | None:
    """'7.8' → 7.8, '1,234' → 1234, '7.8 H' → 7.8; '<5', 'negative' → None."""
    t = text.strip().replace(",", "")
    if re.match(r"^[<>≤≥]", t):
        return None  # censored values are not safe to store as a number
    m = re.match(rf"^({_NUM})(?:\s*[A-Za-z▲▼↑↓*]*)?$", t)
    return float(m.group(1)) if m else None


def parse_range(text: str | None) -> tuple[float | None, float | None] | None:
    """'4.0~5.6' / '4.0 - 5.6' / '<200' / '≥ 60' / '60 이상' → (low, high)."""
    if not text or not text.strip():
        return None
    t = text.strip().replace(",", "").replace("이하", "≤").replace("미만", "<")
    m = re.match(rf"^({_NUM})\s*(?:~|-|–|∼)\s*({_NUM})$", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(rf"^(<|≤|<=)\s*({_NUM})$", t) or re.match(rf"^({_NUM})\s*(<|≤)$", t)
    if m:
        return None, float(next(g for g in m.groups() if re.match(_NUM, g)))
    m = re.match(rf"^(>|≥|>=)\s*({_NUM})$", t) or re.match(rf"^({_NUM})\s*(이상|초과)$", t)
    if m:
        return float(next(g for g in m.groups() if re.match(_NUM, g))), None
    return None


def parse_flag(flag: str | None) -> Interpretation | None:
    if flag is None:
        return None
    f = flag.strip().lower()
    if f in _HIGH_FLAGS:
        return "H"
    if f in _LOW_FLAGS:
        return "L"
    if f in _NORMAL_FLAGS:
        return "N"
    return None


def interpret(value: float, low: float | None, high: float | None) -> Interpretation | None:
    if low is None and high is None:
        return None
    if high is not None and value > high:
        return "H"
    if low is not None and value < low:
        return "L"
    return "N"


def _normalize_item(item: ExtractedLabItem, issues: list[Issue]) -> LabResult:
    test: LabTest | None = lookup(item.name)
    unit = normalize_unit(item.unit)
    value = parse_value(item.value)
    rng = parse_range(item.reference_range)

    def issue(code: IssueCode, msg: str) -> None:
        issues.append(Issue(code=code, message=msg, item=item.name))

    if test is None:
        issue("unmapped_test", f"'{item.name}' 을(를) 표준 검사 항목에 매핑하지 못했습니다")
    if value is None:
        issue("non_numeric_value", f"결과값 '{item.value}' 이(가) 숫자가 아닙니다")

    converted = False
    if test and value is not None and unit and unit != test.unit:
        convert = _CONVERSIONS.get((test.loinc, unit))
        if convert:
            issue("unit_converted", f"{item.value} {unit} → {test.unit}")
            value = round(convert(value), 2)
            unit = test.unit
            rng = None  # printed range is in the old unit; fall back to the catalogue
            converted = True
        else:
            issue(
                "unit_mismatch",
                f"단위 '{unit}' 은(는) {test.name_ko}의 표준 단위 '{test.unit}' 와 다릅니다",
            )
    if test and unit is None:
        unit = test.unit

    if test and value is not None:
        lo, hi = test.plausible
        if not lo <= value <= hi:
            issue(
                "implausible_value",
                f"{value} {unit} 는 생리적으로 불가능한 값입니다 (OCR 오류 의심)",
            )

    if rng is None and test is not None:
        if item.reference_range and not converted:
            issue(
                "missing_reference_range",
                f"참고치 '{item.reference_range}' 를 해석하지 못해 기본 참고범위를 사용합니다",
            )
        rng = (test.ref_low, test.ref_high)
    low, high = rng if rng else (None, None)

    printed = parse_flag(item.flag)
    computed = interpret(value, low, high) if value is not None else None
    if printed and computed and printed != computed:
        issue("flag_mismatch", f"결과지 판정({item.flag})과 계산된 판정({computed})이 다릅니다")

    return LabResult(
        raw_name=item.name,
        name=test.name_ko if test else item.name,
        loinc=test.loinc if test else None,
        value=value,
        value_text=item.value,
        unit=unit,
        ref_low=low,
        ref_high=high,
        interpretation=printed or computed,
    )


def revalidate(report: StructuredLabReport) -> list[Issue]:
    """Server-side re-check of a (possibly human-edited) structured report before saving.

    The client may edit values or drop issues while reviewing, so blocking conditions
    are recomputed from the results themselves rather than trusted from the payload.
    """
    issues: list[Issue] = []
    seen: set[str] = set()
    for r in report.results:
        if r.loinc is None:
            issues.append(
                Issue(code="unmapped_test", message="LOINC 코드가 없습니다", item=r.raw_name)
            )
        if r.value is None:
            issues.append(
                Issue(code="non_numeric_value", message="숫자 값이 없습니다", item=r.raw_name)
            )
        test = by_loinc(r.loinc) if r.loinc else None
        if test and r.value is not None:
            if r.unit not in (None, test.unit):
                issues.append(
                    Issue(
                        code="unit_mismatch",
                        message=f"단위 {r.unit} ≠ {test.unit}",
                        item=r.raw_name,
                    )
                )
            lo, hi = test.plausible
            if not lo <= r.value <= hi:
                issues.append(
                    Issue(
                        code="implausible_value",
                        message=f"{r.value} 는 불가능한 값",
                        item=r.raw_name,
                    )
                )
        if r.loinc and r.loinc in seen:
            issues.append(Issue(code="duplicate_test", message="중복 검사", item=r.raw_name))
        if r.loinc:
            seen.add(r.loinc)
    if report.report_date is None:
        issues.append(Issue(code="missing_report_date", message="검사일이 없습니다"))
    return issues


def value_in_source(value: str, source: str) -> bool:
    """Does the transcribed value appear in the OCR text as a whole token?

    '3.8' must not match inside 'I3.8': the first real-model run "corrected" the OCR
    noise 'I3.8' to '3.8' — a plausible but wrong haemoglobin that no range check catches.
    """
    v = value.strip()
    if not v:
        return False
    return re.search(rf"(?<![\w.,]){re.escape(v)}(?![\w.,])", source) is not None


def normalize_report(
    report: ExtractedLabReport, *, source_text: str | None = None
) -> StructuredLabReport:
    issues: list[Issue] = []
    results = [_normalize_item(item, issues) for item in report.items]
    if source_text is not None:
        for item in report.items:
            if not value_in_source(item.value, source_text):
                issues.append(
                    Issue(
                        code="value_not_in_source",
                        message=f"결과값 '{item.value}' 이(가) 원문 OCR 텍스트에 그대로 없습니다 (변형·환각 의심)",
                        item=item.name,
                    )
                )
    if report.report_date is None:
        issues.append(Issue(code="missing_report_date", message="검사일을 찾지 못했습니다"))
    seen: set[str] = set()
    for r in results:
        if r.loinc and r.loinc in seen:
            issues.append(
                Issue(
                    code="duplicate_test", message="같은 검사가 두 번 나왔습니다", item=r.raw_name
                )
            )
        if r.loinc:
            seen.add(r.loinc)
    return StructuredLabReport(
        report_date=report.report_date,
        institution=report.institution,
        results=results,
        issues=issues,
    )
