"""Small transactional local store; scientific artifacts never depend on this database."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import canonical_json, utc_now
from .inference import ApplicationError


class Store:
    def __init__(self, root: Path):
        self.path = root / "runtime" / "application.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if schema_version not in (0, 1):
                raise ApplicationError(
                    "unsupported_store_schema",
                    "This application cannot open that SQLite schema version.",
                    503,
                )
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                    payload_hash TEXT NOT NULL, created_at TEXT NOT NULL, result TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS streams (
                    stream_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, state TEXT NOT NULL
                );
                PRAGMA user_version = 1;
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def decode(value: str) -> Any:
        import json

        return json.loads(value)

    def save_batch(
        self, request_id: str, payload_hash: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_hash, result FROM batches WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row:
                if row["payload_hash"] != payload_hash:
                    raise ApplicationError(
                        "idempotency_conflict",
                        "This client_request_id already names different input or a different model.",
                        409,
                    )
                return {**self.decode(row["result"]), "replayed": True}
            connection.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                (result["batch_id"], request_id, payload_hash, utc_now(), canonical_json(result)),
            )
        return {**result, "replayed": False}

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT result FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if not row:
            raise ApplicationError("not_found", "Batch not found.", 404)
        return self.decode(row["result"])

    def list_batches(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT result, created_at FROM batches ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        keys = (
            "batch_id",
            "bundle_id",
            "model_id",
            "row_count",
            "eligible_count",
            "review_capacity",
            "abstain_count",
        )
        return [
            {
                **{key: self.decode(row["result"])[key] for key in keys},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_stream(self, state: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO streams VALUES (?, ?, ?)",
                (state["stream_id"], utc_now(), canonical_json(state)),
            )

    def get_stream(self, stream_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT state FROM streams WHERE stream_id = ?", (stream_id,)
            ).fetchone()
        if not row:
            raise ApplicationError("not_found", "Monitoring stream not found.", 404)
        return self.decode(row["state"])

    def update_stream(
        self, stream_id: str, expected_cursor: int, state: dict[str, Any]
    ) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM streams WHERE stream_id = ?", (stream_id,)
            ).fetchone()
            if not row:
                raise ApplicationError("not_found", "Monitoring stream not found.", 404)
            existing = self.decode(row["state"])
            if existing["cursor"] != expected_cursor:
                raise ApplicationError(
                    "cursor_conflict", "Replay advanced already; refresh its current cursor.", 409
                )
            connection.execute(
                "UPDATE streams SET state = ? WHERE stream_id = ?",
                (canonical_json(state), stream_id),
            )
        return state

    def list_streams(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT state FROM streams ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [self.decode(row["state"]) for row in rows]
