"""Field-level accuracy of the lab-report pipeline against a labelled dataset.

Dataset layout (``datasets/lab_reports``): ``NN.ocr.txt`` (input) and
``NN.expected.json`` (``report_date``, ``items[{loinc, value, unit, interpretation}]``,
optional ``expected_issues``).

Metrics
* item precision / recall – by LOINC code (did we find the right tests?)
* value / unit / flag accuracy – over correctly found items
* issue recall – did the pipeline raise every issue a human reviewer must see?
  (a silent wrong number is worse than a flagged one)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from care_agent.documents.pipeline import ExtractOutcome, LabReportPipeline

VALUE_TOLERANCE = 0.011  # values are rounded to 2 decimals after unit conversion


@dataclass
class SampleScore:
    sample: str
    extracted: bool
    date_correct: bool
    precision: float
    recall: float
    value_acc: float | None
    unit_acc: float | None
    flag_acc: float | None
    issue_recall: float | None
    attempts: int
    errors: list[str] = field(default_factory=list)


def score_sample(name: str, outcome: ExtractOutcome, expected: dict[str, Any]) -> SampleScore:
    report = outcome.report
    if report is None:
        return SampleScore(
            name,
            False,
            False,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            len(outcome.attempts),
            ["extraction failed"],
        )
    exp = {i["loinc"]: i for i in expected["items"]}
    pred = {r.loinc: r for r in report.results if r.loinc}
    matched = [code for code in pred if code in exp]
    errors: list[str] = []

    def ratio(ok: list[bool]) -> float | None:
        return sum(ok) / len(ok) if ok else None

    values, units, flags = [], [], []
    for code in matched:
        p, e = pred[code], exp[code]
        v_ok = p.value is not None and abs(p.value - e["value"]) <= VALUE_TOLERANCE
        values.append(v_ok)
        units.append(p.unit == e["unit"])
        flags.append(p.interpretation == e["interpretation"])
        if not v_ok:
            errors.append(f"{code}: value {p.value_text!r}→{p.value} != {e['value']}")
        if p.unit != e["unit"]:
            errors.append(f"{code}: unit {p.unit} != {e['unit']}")
        if p.interpretation != e["interpretation"]:
            errors.append(f"{code}: flag {p.interpretation} != {e['interpretation']}")
    errors += [f"missing {code}" for code in exp if code not in pred]
    errors += [f"unexpected {code}" for code in pred if code not in exp]

    wanted_issues = set(expected.get("expected_issues", []))
    raised = {i.code for i in report.issues}
    issue_recall = len(wanted_issues & raised) / len(wanted_issues) if wanted_issues else None
    errors += [f"issue not raised: {c}" for c in wanted_issues - raised]

    return SampleScore(
        sample=name,
        extracted=True,
        date_correct=str(report.report_date) == expected.get("report_date"),
        precision=len(matched) / len(pred) if pred else 0.0,
        recall=len(matched) / len(exp) if exp else 1.0,
        value_acc=ratio(values),
        unit_acc=ratio(units),
        flag_acc=ratio(flags),
        issue_recall=issue_recall,
        attempts=len(outcome.attempts),
        errors=errors,
    )


def summarize(scores: list[SampleScore]) -> dict[str, Any]:
    def avg(attr: str) -> float | None:
        vals = [getattr(s, attr) for s in scores if getattr(s, attr) is not None]
        return round(mean(float(v) for v in vals), 3) if vals else None

    return {
        "samples": len(scores),
        "extraction_rate": avg("extracted"),
        "date_accuracy": avg("date_correct"),
        "item_precision": avg("precision"),
        "item_recall": avg("recall"),
        "value_accuracy": avg("value_acc"),
        "unit_accuracy": avg("unit_acc"),
        "flag_accuracy": avg("flag_acc"),
        "issue_recall": avg("issue_recall"),
        "mean_attempts": avg("attempts"),
    }


def load_dataset(directory: str | Path) -> list[tuple[str, str, dict[str, Any]]]:
    """[(sample name, OCR text, expected)] sorted by name."""
    samples = []
    for text_path in sorted(Path(directory).glob("*.ocr.txt")):
        name = text_path.name.removesuffix(".ocr.txt")
        expected = json.loads((text_path.parent / f"{name}.expected.json").read_text("utf-8"))
        samples.append((name, text_path.read_text("utf-8"), expected))
    return samples


async def evaluate_dataset(pipeline: LabReportPipeline, directory: str | Path) -> dict[str, Any]:
    scores = []
    for name, text, expected in await asyncio.to_thread(load_dataset, directory):
        outcome = await pipeline.extract(text=text)
        scores.append(score_sample(name, outcome, expected))
    return {"summary": summarize(scores), "samples": [asdict(s) for s in scores]}
