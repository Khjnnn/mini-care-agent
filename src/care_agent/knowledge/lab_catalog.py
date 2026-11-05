"""A small catalogue of common lab tests.

Shared by the ``explain_lab_test`` skill (week 2) and the lab-report structuring
pipeline (week 3), which uses it to map free-text test names from OCR to LOINC, to
normalise units and to reject physically implausible values.

Reference ranges are typical adult ranges for demo purposes; real systems must use the
range printed by the issuing lab, which always takes precedence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LabTest:
    loinc: str
    name_ko: str
    name_en: str
    unit: str
    aliases: tuple[str, ...]
    #: Typical adult reference range (None = open-ended).
    ref_low: float | None
    ref_high: float | None
    #: Anything outside this range is almost certainly an OCR/extraction error.
    plausible: tuple[float, float]
    explanation: str


CATALOG: tuple[LabTest, ...] = (
    LabTest(
        "4548-4",
        "당화혈색소",
        "Hemoglobin A1c",
        "%",
        ("hba1c", "a1c", "당화혈색소", "glycatedhemoglobin", "hemoglobina1c"),
        4.0,
        5.6,
        (2.0, 20.0),
        "최근 2~3개월 평균 혈당을 반영합니다. 5.7~6.4%는 당뇨 전단계, 6.5% 이상은 당뇨 진단 기준 중 하나이며, 당뇨 환자의 일반적 관리 목표는 7% 미만입니다.",
    ),
    LabTest(
        "1558-6",
        "공복혈당",
        "Fasting glucose",
        "mg/dL",
        ("공복혈당", "fbs", "fastingglucose", "fastingbloodsugar", "glucosefasting", "공복시혈당"),
        70,
        99,
        (20, 1000),
        "8시간 이상 금식 후 잰 혈당입니다. 100~125 mg/dL는 공복혈당장애, 126 mg/dL 이상은 당뇨 진단 기준 중 하나입니다.",
    ),
    LabTest(
        "2093-3",
        "총콜레스테롤",
        "Total cholesterol",
        "mg/dL",
        ("총콜레스테롤", "totalcholesterol", "tc", "cholesterol", "콜레스테롤"),
        None,
        200,
        (50, 1000),
        "혈액 속 콜레스테롤 전체 양입니다. 200 mg/dL 미만이 바람직합니다.",
    ),
    LabTest(
        "18262-6",
        "LDL 콜레스테롤",
        "LDL cholesterol",
        "mg/dL",
        ("ldl", "ldlcholesterol", "ldl콜레스테롤", "저밀도콜레스테롤", "ldlc"),
        None,
        130,
        (5, 600),
        "혈관에 쌓여 동맥경화를 일으킬 수 있는 '나쁜' 콜레스테롤입니다. 목표치는 당뇨·심혈관 질환 여부에 따라 달라집니다.",
    ),
    LabTest(
        "2085-9",
        "HDL 콜레스테롤",
        "HDL cholesterol",
        "mg/dL",
        ("hdl", "hdlcholesterol", "hdl콜레스테롤", "고밀도콜레스테롤", "hdlc"),
        40,
        None,
        (5, 200),
        "혈관의 콜레스테롤을 제거하는 '좋은' 콜레스테롤입니다. 40 mg/dL 이상이 바람직합니다.",
    ),
    LabTest(
        "2571-8",
        "중성지방",
        "Triglycerides",
        "mg/dL",
        ("중성지방", "triglyceride", "triglycerides", "tg"),
        None,
        150,
        (10, 5000),
        "음식으로 섭취한 지방과 남는 열량이 저장된 형태입니다. 150 mg/dL 미만이 정상입니다.",
    ),
    LabTest(
        "2160-0",
        "크레아티닌",
        "Creatinine",
        "mg/dL",
        ("크레아티닌", "creatinine", "cr", "scr"),
        0.7,
        1.3,
        (0.1, 20),
        "근육 대사 노폐물로 콩팥 기능을 반영합니다. 높으면 콩팥 기능 저하를 의심할 수 있습니다.",
    ),
    LabTest(
        "33914-3",
        "사구체여과율",
        "eGFR",
        "mL/min/1.73m2",
        ("egfr", "사구체여과율", "추정사구체여과율", "gfr"),
        60,
        None,
        (1, 200),
        "콩팥이 1분 동안 걸러내는 혈액량의 추정치입니다. 60 미만이 3개월 이상 지속되면 만성콩팥병을 의심합니다.",
    ),
    LabTest(
        "1920-8",
        "AST",
        "AST",
        "U/L",
        ("ast", "sgot", "got", "아스파르테이트아미노전달효소"),
        0,
        40,
        (1, 10000),
        "간·근육 등에 있는 효소로, 간세포가 손상되면 혈중 수치가 올라갑니다.",
    ),
    LabTest(
        "1742-6",
        "ALT",
        "ALT",
        "U/L",
        ("alt", "sgpt", "gpt", "알라닌아미노전달효소"),
        0,
        40,
        (1, 10000),
        "주로 간에 있는 효소로, AST보다 간 손상에 더 특이적입니다.",
    ),
    LabTest(
        "718-7",
        "혈색소",
        "Hemoglobin",
        "g/dL",
        ("혈색소", "헤모글로빈", "hemoglobin", "hb", "hgb"),
        12.0,
        17.0,
        (2, 25),
        "적혈구에서 산소를 운반하는 단백질입니다. 낮으면 빈혈을 의심합니다. 정상범위는 성별에 따라 다릅니다.",
    ),
)

_BY_LOINC = {t.loinc: t for t in CATALOG}


def normalize_name(text: str) -> str:
    return re.sub(r"[\s\-_()/\[\].,:]+", "", text).lower()


_BY_ALIAS = {normalize_name(a): t for t in CATALOG for a in (*t.aliases, t.name_ko, t.name_en)}


def by_loinc(code: str) -> LabTest | None:
    return _BY_LOINC.get(code)


def lookup(name: str) -> LabTest | None:
    """Map a free-text test name (Korean or English, OCR-noisy) to a catalogue entry."""
    key = normalize_name(name)
    if not key:
        return None
    if key in _BY_ALIAS:
        return _BY_ALIAS[key]
    # Longest alias (>= 4 chars, so "ast" does not match "fasting") contained in the text
    # wins, e.g. "LDL-콜레스테롤(직접측정)" → LDL.
    hits = [(len(alias), t) for alias, t in _BY_ALIAS.items() if len(alias) >= 4 and alias in key]
    return max(hits, key=lambda h: h[0])[1] if hits else None
