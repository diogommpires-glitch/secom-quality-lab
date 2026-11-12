"""Generate the HTTP contract without loading models or accessing research data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from secom_quality.api import create_app

root = Path(__file__).resolve().parents[1]
with TemporaryDirectory(prefix="secom-openapi-") as temporary:
    schema = create_app(Path(temporary)).openapi()
(root / "frontend" / "openapi.json").write_text(
    json.dumps(schema, indent=2) + "\n", encoding="utf-8"
)
