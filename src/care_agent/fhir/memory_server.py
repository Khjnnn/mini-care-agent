"""An in-memory FHIR server exposed as an ``httpx`` transport.

Supports just enough of the R4 search API for the agent's tools, so tests and the
offline demo run without Docker. Behaviour mirrors HAPI for the parameters we use:
``patient``/``subject``, ``category``, ``code``, ``clinical-status``, ``status``,
``_sort=-date`` and ``_count``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import httpx

DATE_FIELDS = ("effectiveDateTime", "authoredOn", "onsetDateTime", "issued")


def load_demo_records() -> list[dict[str, Any]]:
    raw = resources.files("care_agent.fhir").joinpath("data/demo_records.json").read_text("utf-8")
    return json.loads(raw)["resources"]


def _codes(concepts: Any) -> set[str]:
    if isinstance(concepts, dict):
        concepts = [concepts]
    return {c.get("code") for concept in concepts or [] for c in concept.get("coding", [])}


def _date(resource: dict[str, Any]) -> str:
    return next((resource[f] for f in DATE_FIELDS if resource.get(f)), "")


def _matches(resource: dict[str, Any], params: httpx.QueryParams) -> bool:
    for key, value in params.multi_items():
        if key.startswith("_"):
            continue
        wanted = set(value.split(","))
        if key in ("patient", "subject"):
            ref = resource.get("subject", {}).get("reference", "")
            ids = {w.removeprefix("Patient/") for w in wanted}
            if ref.removeprefix("Patient/") not in ids:
                return False
        elif key == "category":
            if not wanted & _codes(resource.get("category")):
                return False
        elif key == "code":
            if not wanted & _codes(resource.get("code")):
                return False
        elif key == "clinical-status":
            if not wanted & _codes(resource.get("clinicalStatus")):
                return False
        elif key == "status":
            if resource.get("status") not in wanted:
                return False
        else:
            raise ValueError(f"unsupported search parameter: {key}")
    return True


class InMemoryFHIR(httpx.AsyncBaseTransport):
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records if records is not None else load_demo_records()
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        parts = [p for p in request.url.path.split("/") if p and p != "fhir"]
        if request.method == "POST" and not parts:
            return self._transaction(json.loads(await request.aread()))
        if request.method != "GET":
            return _outcome(405, "only GET and transaction POST are supported")
        if len(parts) == 2:
            rtype, rid = parts
            for r in self.records:
                if r["resourceType"] == rtype and r["id"] == rid:
                    return httpx.Response(200, json=r)
            return _outcome(404, f"{rtype}/{rid} not found")
        if len(parts) == 1:
            try:
                hits = [
                    r
                    for r in self.records
                    if r["resourceType"] == parts[0] and _matches(r, request.url.params)
                ]
            except ValueError as e:
                return _outcome(400, str(e))
            if request.url.params.get("_sort") in ("-date", "-_lastUpdated"):
                hits.sort(key=_date, reverse=True)
            count = int(request.url.params.get("_count", 50))
            bundle = {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(hits),
                "entry": [{"resource": r, "search": {"mode": "match"}} for r in hits[:count]],
            }
            return httpx.Response(200, json=bundle)
        return _outcome(400, "unsupported path")

    def _transaction(self, bundle: dict[str, Any]) -> httpx.Response:
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction":
            return _outcome(400, "expected a transaction Bundle")
        entries = bundle.get("entry", [])
        if any(e.get("request", {}).get("method") != "POST" for e in entries):
            return _outcome(400, "only POST entries are supported")
        created = []
        for entry in entries:  # validate everything first: all-or-nothing
            resource = dict(entry["resource"])
            resource["id"] = f"mem-{len(self.records) + len(created) + 1}"
            created.append(resource)
        self.records.extend(created)
        return httpx.Response(
            200,
            json={
                "resourceType": "Bundle",
                "type": "transaction-response",
                "entry": [
                    {
                        "response": {
                            "status": "201 Created",
                            "location": f"{r['resourceType']}/{r['id']}",
                        }
                    }
                    for r in created
                ],
            },
        )


def _outcome(status: int, msg: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "resourceType": "OperationOutcome",
            "issue": [{"severity": "error", "code": "processing", "diagnostics": msg}],
        },
    )
