"""Lab report pipeline: extract → normalise → (human review) → commit to FHIR.

Nothing is written to the EHR in ``extract``. ``commit`` is the explicit, human-approved
write: it re-validates the (possibly edited) report server-side, refuses blocking
issues unless the reviewer acknowledges them, and writes all Observations in one
transaction so a partial report never lands in the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from care_agent.documents.extract import Attempt, LabReportExtractor
from care_agent.documents.fhir_export import to_transaction_bundle
from care_agent.documents.normalize import normalize_report, revalidate
from care_agent.documents.schema import BLOCKING_ISSUES, Issue, StructuredLabReport
from care_agent.fhir.client import FHIRClient


class ReviewRequired(Exception):
    def __init__(self, issues: list[Issue]) -> None:
        super().__init__(f"{len(issues)} blocking issue(s) need review")
        self.issues = issues


@dataclass
class ExtractOutcome:
    report: StructuredLabReport | None
    attempts: list[Attempt]
    source: str
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.report is None or self.report.needs_review


@dataclass
class CommitOutcome:
    created: list[str]
    acknowledged_issues: list[Issue] = field(default_factory=list)


class LabReportPipeline:
    def __init__(self, extractor: LabReportExtractor, fhir: FHIRClient) -> None:
        self.extractor = extractor
        self.fhir = fhir

    async def extract(
        self, *, text: str | None = None, image: bytes | None = None, mime: str = "image/png"
    ) -> ExtractOutcome:
        if (text is None) == (image is None):
            raise ValueError("pass exactly one of text or image")
        result = (
            await self.extractor.from_text(text)
            if text is not None
            else await self.extractor.from_image(image or b"", mime)
        )
        if result.report is None:
            return ExtractOutcome(None, result.attempts, result.source, "extraction failed")
        return ExtractOutcome(
            normalize_report(result.report, source_text=result.source_text),
            result.attempts,
            result.source,
        )

    async def commit(
        self,
        report: StructuredLabReport,
        patient_id: str,
        *,
        reviewed_by: str,
        acknowledge_issues: bool = False,
    ) -> CommitOutcome:
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required")
        blocking = [i for i in revalidate(report) if i.code in BLOCKING_ISSUES]
        # unmapped tests / non-numeric values can never be acknowledged into the record
        hard = [
            i
            for i in blocking
            if i.code in {"unmapped_test", "non_numeric_value", "missing_report_date"}
        ]
        if hard or (blocking and not acknowledge_issues):
            raise ReviewRequired(hard or blocking)
        await self.fhir.read("Patient", patient_id)  # 404 before writing anything
        response = await self.fhir.transaction(to_transaction_bundle(report, patient_id))
        created = [e.get("response", {}).get("location", "") for e in response.get("entry", [])]
        return CommitOutcome(created, blocking)


def outcome_to_dict(outcome: ExtractOutcome) -> dict[str, Any]:
    return {
        "report": outcome.report.model_dump(mode="json") if outcome.report else None,
        "needs_review": outcome.needs_review,
        "source": outcome.source,
        "error": outcome.error,
        "attempts": [{"error": a.error} for a in outcome.attempts],
    }
