"""``StructuredLabReport`` → FHIR R4 transaction bundle of Observations."""

from __future__ import annotations

import uuid
from typing import Any

from care_agent.documents.schema import LabResult, StructuredLabReport

_UCUM = {
    "mg/dL": "mg/dL",
    "g/dL": "g/dL",
    "%": "%",
    "U/L": "U/L",
    "mL/min/1.73m2": "mL/min/{1.73_m2}",
}
_INTERP_DISPLAY = {"H": "High", "L": "Low", "N": "Normal"}
PROVENANCE_TAG = {
    "system": "https://github.com/Khjnnn/mini-care-agent",
    "code": "extracted-lab-report",
    "display": "Extracted from a lab report image by mini-care-agent and reviewed by a human",
}


def observation(result: LabResult, patient_id: str, report: StructuredLabReport) -> dict[str, Any]:
    code: dict[str, Any] = {"text": result.name}
    if result.loinc:
        code["coding"] = [
            {"system": "http://loinc.org", "code": result.loinc, "display": result.name}
        ]
    obs: dict[str, Any] = {
        "resourceType": "Observation",
        "meta": {"tag": [PROVENANCE_TAG]},
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                    }
                ]
            }
        ],
        "code": code,
        "subject": {"reference": f"Patient/{patient_id}"},
    }
    if report.report_date:
        obs["effectiveDateTime"] = report.report_date.isoformat()
    if result.value is not None:
        q: dict[str, Any] = {"value": result.value}
        if result.unit:
            q["unit"] = result.unit
            if result.unit in _UCUM:
                q |= {"system": "http://unitsofmeasure.org", "code": _UCUM[result.unit]}
        obs["valueQuantity"] = q
    else:
        obs["valueString"] = result.value_text
    if result.ref_low is not None or result.ref_high is not None:
        rng: dict[str, Any] = {}
        if result.ref_low is not None:
            rng["low"] = {"value": result.ref_low, "unit": result.unit}
        if result.ref_high is not None:
            rng["high"] = {"value": result.ref_high, "unit": result.unit}
        obs["referenceRange"] = [rng]
    if result.interpretation:
        obs["interpretation"] = [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": result.interpretation,
                        "display": _INTERP_DISPLAY[result.interpretation],
                    }
                ]
            }
        ]
    if report.institution:
        obs["note"] = [{"text": f"검사기관: {report.institution}"}]
    return obs


def to_transaction_bundle(report: StructuredLabReport, patient_id: str) -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": observation(r, patient_id, report),
                "request": {"method": "POST", "url": "Observation"},
            }
            for r in report.results
        ],
    }
