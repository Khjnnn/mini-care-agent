"""Turn raw FHIR resources into compact, LLM-friendly dicts.

Two goals:
* **Token efficiency** – raw Synthea resources are huge; the model only needs a few fields.
* **Data minimisation** – direct identifiers (name, address, telecom, MRN) never leave
  this layer, so they cannot end up in a prompt, a trace, or a third-party LLM.
"""

from __future__ import annotations

from datetime import date
from typing import Any


def codeable_text(concept: dict[str, Any] | None) -> str | None:
    if not concept:
        return None
    if concept.get("text"):
        return concept["text"]
    for coding in concept.get("coding", []):
        if coding.get("display"):
            return coding["display"]
        if coding.get("code"):
            return coding["code"]
    return None


def first_code(concept: dict[str, Any] | None) -> str | None:
    for coding in (concept or {}).get("coding", []):
        if coding.get("code"):
            return coding["code"]
    return None


def age_on(birth_date: str, today: date) -> int:
    born = date.fromisoformat(birth_date[:10])
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def summarize_patient(patient: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    birth = patient.get("birthDate")
    return {
        "id": patient.get("id"),
        "gender": patient.get("gender"),
        "birth_year": int(birth[:4]) if birth else None,
        "age": age_on(birth, today) if birth else None,
        "deceased": bool(patient.get("deceasedDateTime") or patient.get("deceasedBoolean")),
    }


def summarize_condition(cond: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": codeable_text(cond.get("code")),
        "snomed": first_code(cond.get("code")),
        "clinical_status": first_code(cond.get("clinicalStatus")),
        "onset": (cond.get("onsetDateTime") or "")[:10] or None,
        "abated": (cond.get("abatementDateTime") or "")[:10] or None,
    }


def _quantity(q: dict[str, Any]) -> tuple[Any, str | None]:
    value = q.get("value")
    if isinstance(value, float):
        value = round(value, 2)
    return value, q.get("unit") or q.get("code")


def _value(obs: dict[str, Any]) -> dict[str, Any]:
    if "valueQuantity" in obs:
        value, unit = _quantity(obs["valueQuantity"])
        return {"value": value, "unit": unit}
    if "valueCodeableConcept" in obs:
        return {"value": codeable_text(obs["valueCodeableConcept"])}
    for key in ("valueString", "valueBoolean", "valueInteger"):
        if key in obs:
            return {"value": obs[key]}
    if obs.get("component"):  # e.g. blood pressure panel
        parts = {}
        for comp in obs["component"]:
            if "valueQuantity" in comp:
                value, unit = _quantity(comp["valueQuantity"])
                parts[codeable_text(comp.get("code")) or "?"] = f"{value} {unit or ''}".strip()
        return {"components": parts}
    return {}


def summarize_observation(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": codeable_text(obs.get("code")),
        "loinc": first_code(obs.get("code")),
        "date": (obs.get("effectiveDateTime") or obs.get("issued") or "")[:10] or None,
        **_value(obs),
    }
    interp = obs.get("interpretation")
    if interp:
        out["interpretation"] = codeable_text(interp[0])
    ranges = obs.get("referenceRange")
    if ranges:
        rng = ranges[0]
        low, high = rng.get("low", {}).get("value"), rng.get("high", {}).get("value")
        out["reference_range"] = rng.get("text") or f"{low}-{high}"
    return out


def summarize_medication_request(med: dict[str, Any]) -> dict[str, Any]:
    dosage = med.get("dosageInstruction") or [{}]
    return {
        "medication": codeable_text(med.get("medicationCodeableConcept"))
        or (med.get("medicationReference") or {}).get("display"),
        "status": med.get("status"),
        "authored_on": (med.get("authoredOn") or "")[:10] or None,
        "dosage": dosage[0].get("text"),
        "reason": codeable_text((med.get("reasonCode") or [None])[0]),
    }
