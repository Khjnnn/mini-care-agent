import base64
import json
from pathlib import Path

import httpx
import pytest

from care_agent.documents.evaluate import evaluate_dataset
from care_agent.documents.extract import LabReportExtractor
from care_agent.documents.normalize import normalize_unit, parse_range, parse_value
from care_agent.documents.pipeline import LabReportPipeline, ReviewRequired
from care_agent.documents.schema import ExtractedLabReport
from care_agent.llm.scripted import ScriptedLLM
from care_agent.platform.api import create_app
from tests.helpers import final

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "lab_transcriptions"


def transcription(name: str) -> str:
    return (FIXTURES / f"{name}.json").read_text("utf-8")


@pytest.mark.parametrize(
    ("text", "value"),
    [("7.8", 7.8), ("1,234", 1234.0), ("7.8 H", 7.8), ("<5", None), ("음성", None), ("I3.8", None)],
)
def test_parse_value(text, value):
    assert parse_value(text) == value


@pytest.mark.parametrize(
    ("text", "rng"),
    [
        ("4.0~5.6", (4.0, 5.6)),
        ("70 - 99", (70.0, 99.0)),
        ("<200", (None, 200.0)),
        ("130 미만", (None, 130.0)),
        ("≥ 60", (60.0, None)),
        ("60 이상", (60.0, None)),
        ("음성", None),
    ],
)
def test_parse_range(text, rng):
    assert parse_range(text) == rng


def test_normalize_unit():
    assert normalize_unit("mg/dl") == "mg/dL"
    assert normalize_unit("IU/L") == "U/L"
    assert normalize_unit("mL/min/1.73㎡") == "mL/min/1.73m2"
    assert normalize_unit(" ") is None


@pytest.fixture
def pipeline(fhir):
    def _make(steps):
        return LabReportPipeline(LabReportExtractor(ScriptedLLM(steps)), fhir)

    return _make


async def test_unit_conversion_and_flags(pipeline):
    out = await pipeline([final(transcription("02"))]).extract(text="...")
    by = {r.loinc: r for r in out.report.results}
    assert by["1558-6"].value == 109.9 and by["1558-6"].unit == "mg/dL"
    assert by["4548-4"].value == 6.08 and by["4548-4"].unit == "%"
    assert by["1742-6"].unit == "U/L" and by["1742-6"].interpretation == "H"
    assert by["718-7"].interpretation == "L"
    assert {i.code for i in out.report.issues} == {"unit_converted"}
    assert not out.needs_review


async def test_ocr_noise_is_flagged_not_guessed(pipeline):
    out = await pipeline([final(transcription("03"))]).extract(text="...")
    by = {r.raw_name: r for r in out.report.results}
    assert by["혈색소"].value is None and by["LDL콜레스테롤"].value is None
    codes = [i.code for i in out.report.issues]
    assert codes.count("non_numeric_value") == 3 and "unmapped_test" in codes
    assert out.needs_review


async def test_self_correction_after_invalid_output(pipeline):
    llm_steps = [
        final("결과는 다음과 같습니다: 당화혈색소 7.9"),  # not JSON
        final('```json\n{"report_date": "2026-10-20", "items": []}\n```'),  # empty items
        final(transcription("01")),
    ]
    p = pipeline(llm_steps)
    out = await p.extract(text="...")
    assert out.report is not None and len(out.attempts) == 3
    assert [a.error is None for a in out.attempts] == [False, False, True]
    retry_prompt = p.extractor.llm.calls[2][-1]["content"]
    assert "items" in retry_prompt  # the validation error was fed back


async def test_extraction_gives_up(pipeline):
    out = await pipeline([final("nope")] * 3).extract(text="...")
    assert out.report is None and out.needs_review and out.error


async def test_vlm_message_shape(fhir):
    llm = ScriptedLLM([final(transcription("01"))])
    p = LabReportPipeline(LabReportExtractor(llm), fhir)
    out = await p.extract(image=b"\x89PNG fake", mime="image/png")
    parts = llm.calls[0][1]["content"]
    assert out.source == "vlm"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_dataset_evaluation(pipeline):
    p = pipeline([final(transcription(n)) for n in ("01", "02", "03")])
    report = await evaluate_dataset(p, ROOT / "datasets" / "lab_reports")
    s = report["summary"]
    assert s["samples"] == 3 and s["item_precision"] == 1.0 and s["item_recall"] == 1.0
    assert s["date_accuracy"] == 1.0 and s["issue_recall"] == 1.0
    # sample 03 has two OCR-corrupted values that are (correctly) not turned into numbers
    sample3 = report["samples"][2]
    assert sample3["value_acc"] == 0.5
    assert s["flag_accuracy"] < 1.0


async def test_commit_requires_review_then_writes_and_agent_sees_it(
    pipeline, registry, fhir_server
):
    p = pipeline([final(transcription("01"))])
    report = (await p.extract(text="...")).report

    with pytest.raises(ValueError):
        await p.commit(report, "demo-001", reviewed_by=" ")

    out = await p.commit(report, "demo-001", reviewed_by="nurse-kim")
    assert len(out.created) == 8 and all(c.startswith("Observation/") for c in out.created)

    a1c = await registry.get("get_recent_observations").invoke(
        {"patient_id": "demo-001", "loinc_code": "4548-4", "limit": 1}
    )
    assert a1c[0]["value"] == 7.9 and a1c[0]["date"] == "2026-10-20"


async def test_commit_revalidates_edited_report(pipeline):
    p = pipeline([final(transcription("01"))])
    report = (await p.extract(text="...")).report
    report.results[0].value = 79.0  # reviewer typo; issues list is not trusted
    report.issues = []
    with pytest.raises(ReviewRequired) as e:
        await p.commit(report, "demo-001", reviewed_by="x")
    assert e.value.issues[0].code == "implausible_value"


async def test_hard_issues_cannot_be_acknowledged(pipeline):
    p = pipeline([final(transcription("03"))])
    report = (await p.extract(text="...")).report
    with pytest.raises(ReviewRequired):
        await p.commit(report, "demo-001", reviewed_by="x", acknowledge_issues=True)


async def test_commit_unknown_patient_writes_nothing(pipeline, fhir_server):
    p = pipeline([final(transcription("01"))])
    report = (await p.extract(text="...")).report
    before = len(fhir_server.records)
    with pytest.raises(Exception, match="404"):
        await p.commit(report, "ghost", reviewed_by="x")
    assert len(fhir_server.records) == before


async def test_api_extract_and_commit(make_platform):
    llm = ScriptedLLM([final(transcription("01"))])
    platform = make_platform(llm)
    transport = httpx.ASGITransport(app=create_app(platform))
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        bad = await client.post("/v1/documents/lab-report/extract", json={})
        assert bad.status_code == 422
        img = base64.b64encode(b"fake-png").decode()
        ext = (
            await client.post("/v1/documents/lab-report/extract", json={"image_base64": img})
        ).json()
        assert ext["needs_review"] is False and ext["source"] == "vlm"

        body = {"patient_id": "demo-001", "report": ext["report"], "reviewed_by": "dr-lee"}
        committed = await client.post("/v1/documents/lab-report/commit", json=body)
        assert committed.status_code == 200 and len(committed.json()["created"]) == 8

        tampered = json.loads(json.dumps(body))
        tampered["report"]["results"][1]["value"] = 5000
        rejected = await client.post("/v1/documents/lab-report/commit", json=tampered)
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["issues"][0]["code"] == "implausible_value"


def test_schema_has_no_patient_identifier_fields():
    fields = set(ExtractedLabReport.model_fields) | set(
        ExtractedLabReport.model_json_schema()["$defs"]["ExtractedLabItem"]["properties"]
    )
    assert not fields & {"patient_name", "name_of_patient", "rrn", "phone"}
