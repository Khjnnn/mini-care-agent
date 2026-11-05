"""Minimal async FHIR R4 REST client (read + search)."""

from __future__ import annotations

from typing import Any, Self

import httpx


class FHIRError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"FHIR {status_code}: {message}")
        self.status_code = status_code


class FHIRClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            transport=transport,
            timeout=timeout,
            headers={"Accept": "application/fhir+json"},
        )

    async def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{resource_type}/{resource_id}")
        return self._json(resp)

    async def search(
        self, resource_type: str, params: dict[str, str | int] | None = None
    ) -> list[dict[str, Any]]:
        """Search and return the matched resources (first page only)."""
        resp = await self._http.get(resource_type, params=params or {})
        bundle = self._json(resp)
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("search", {}).get("mode", "match") == "match" and "resource" in entry
        ]

    async def transaction(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """POST a transaction bundle (all-or-nothing on a conformant server)."""
        if bundle.get("type") != "transaction":
            raise ValueError("bundle.type must be 'transaction'")
        resp = await self._http.post(
            "", json=bundle, headers={"Content-Type": "application/fhir+json"}
        )
        return self._json(resp)

    @staticmethod
    def _json(resp: httpx.Response) -> dict[str, Any]:
        if resp.is_success:
            return resp.json()
        detail = resp.text[:300]
        try:
            issues = resp.json().get("issue", [])
            if issues:
                detail = issues[0].get("diagnostics") or issues[0].get("code") or detail
        except ValueError:
            pass
        raise FHIRError(resp.status_code, detail)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
