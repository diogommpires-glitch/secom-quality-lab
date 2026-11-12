"""Small, transactional history for image inspection batches."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from secom_quality.inference import ApplicationError


class VisionStore:
    def __init__(self, root: Path):
        self.path = root / "vision" / "operational.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, "
                "request_id TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL, "
                "created_at TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def retry(self, request_id: str, fingerprint: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT fingerprint,payload FROM batches WHERE request_id=?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise ApplicationError(
                "request_conflict", "This request ID belongs to different content.", 409
            )
        return {**json.loads(row["payload"]), "replayed": True}

    def save(self, request_id: str, fingerprint: str, result: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO batches VALUES (?,?,?,?,?)",
                    (
                        result["batch_id"],
                        request_id,
                        fingerprint,
                        result["created_at"],
                        json.dumps(result, allow_nan=False),
                    ),
                )
            except sqlite3.IntegrityError:
                existing = self.retry(request_id, fingerprint)
                if existing is not None:
                    return existing
                raise
        return result

    def get(self, batch_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None:
            raise ApplicationError("not_found", "Vision batch not found.", 404)
        return json.loads(row["payload"])

    def list(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT batch_id,created_at,payload FROM batches ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [
            {
                "batch_id": row["batch_id"],
                "created_at": row["created_at"],
                "row_count": json.loads(row["payload"])["row_count"],
            }
            for row in rows
        ]
