"""Pull one JSON object out of free-form model output (code fences, chatter, ...)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(raw: str) -> Any:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])
