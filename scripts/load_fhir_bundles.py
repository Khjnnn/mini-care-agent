"""Load Synthea transaction bundles into a FHIR server, then list the loaded patients.

    uv run python scripts/load_fhir_bundles.py .synthea/output/fhir [--base http://localhost:8080/fhir]

Hospital and practitioner bundles are loaded first because patient bundles reference them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def ordered(paths: list[Path]) -> list[Path]:
    def rank(p: Path) -> int:
        if p.name.startswith("hospitalInformation"):
            return 0
        if p.name.startswith("practitionerInformation"):
            return 1
        return 2

    return sorted(paths, key=lambda p: (rank(p), p.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", default="http://localhost:8080/fhir")
    args = parser.parse_args()

    files = ordered(list(args.directory.glob("*.json")))
    if not files:
        print(f"no bundles in {args.directory}", file=sys.stderr)
        return 1

    failed = 0
    with httpx.Client(
        base_url=args.base, timeout=300, headers={"Content-Type": "application/fhir+json"}
    ) as http:
        for i, path in enumerate(files, 1):
            resp = http.post("", content=path.read_bytes())
            ok = resp.is_success
            failed += not ok
            print(
                f"[{i}/{len(files)}] {'ok ' if ok else 'ERR'} {path.name}"
                + ("" if ok else f" → {resp.status_code} {resp.text[:200]}")
            )

        bundle = http.get(
            "Patient", params={"_count": 50, "_elements": "id,gender,birthDate"}
        ).json()
        print("\nPatients on server (use these ids with --patient):")
        for entry in bundle.get("entry", []):
            p = entry["resource"]
            print(f"  {p['id']:<40} {p.get('gender', '?'):<7} {p.get('birthDate', '?')}")
        print(json.dumps({"loaded": len(files) - failed, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
