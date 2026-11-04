"""Write the API's OpenAPI spec to docs/openapi.json (the formal API documentation deliverable).

  python scripts/export_openapi.py

tests/test_audit_api.py fails if the committed file falls behind the code, so rerun this after changing an endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))

from app.main import app  # noqa: E402

OUT = ROOT / "docs" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
