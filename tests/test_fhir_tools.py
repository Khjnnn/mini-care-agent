from datetime import date

import pytest

from care_agent.fhir.client import FHIRError
from care_agent.fhir.summaries import summarize_patient


async def test_patient_summary_drops_identifiers(registry, fhir):
    out = await registry.get("get_patient_summary").invoke({"patient_id": "demo-001"})
    assert out["gender"] == "male" and out["birth_year"] == 1968
    raw = await fhir.read("Patient", "demo-001")
    assert "name" in raw  # the raw record has identifiers...
    assert not {"name", "address", "telecom"} & out.keys()  # ...the tool output does not


def test_age_calculation():
    assert summarize_patient({"birthDate": "1968-03-14"}, today=date(2026, 3, 13))["age"] == 57
    assert summarize_patient({"birthDate": "1968-03-14"}, today=date(2026, 3, 14))["age"] == 58


async def test_conditions_active_filter(registry):
    tool = registry.get("list_conditions")
    active = await tool.invoke({"patient_id": "demo-001"})
    every = await tool.invoke({"patient_id": "demo-001", "active_only": False})
    assert {c["condition"] for c in active} == {
        "Essential hypertension (disorder)",
        "Diabetes mellitus type 2 (disorder)",
    }
    assert len(every) == 3


async def test_observations_latest_first_with_reference_range(registry):
    tool = registry.get("get_recent_observations")
    a1c = await tool.invoke({"patient_id": "demo-001", "test": "4548-4"})
    assert [o["value"] for o in a1c] == [7.8, 7.2]
    assert a1c[0]["date"] == "2026-09-15"
    assert a1c[0]["reference_range"] == "4.0-5.6"
    assert a1c[0]["interpretation"] == "High"


async def test_blood_pressure_components(registry):
    vitals = await registry.get("get_recent_observations").invoke(
        {"patient_id": "demo-001", "category": "vital-signs", "limit": 5}
    )
    bp = next(o for o in vitals if o["loinc"] == "85354-9")
    assert bp["components"] == {
        "Systolic blood pressure": "148 mm[Hg]",
        "Diastolic blood pressure": "92 mm[Hg]",
    }


async def test_medications(registry):
    meds = await registry.get("list_medications").invoke({"patient_id": "demo-001"})
    assert {m["medication"] for m in meds} == {
        "Metformin hydrochloride 500 MG Oral Tablet",
        "Lisinopril 10 MG Oral Tablet",
    }


async def test_missing_patient_raises(registry):
    with pytest.raises(FHIRError) as e:
        await registry.get("get_patient_summary").invoke({"patient_id": "nope"})
    assert e.value.status_code == 404
