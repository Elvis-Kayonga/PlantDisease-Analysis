"""
SQLite persistence layer for predictions, uploads, and training runs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "plant_disease.db"


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                file_name TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL,
                source TEXT DEFAULT 'ui',
                is_low_confidence INTEGER DEFAULT 0,
                probabilities_json TEXT
            );

            CREATE TABLE IF NOT EXISTS upload_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                class_name TEXT,
                uploaded_count INTEGER NOT NULL,
                skipped_count INTEGER NOT NULL,
                total_uploaded_data INTEGER NOT NULL,
                skipped_files_json TEXT
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                total_samples INTEGER,
                num_classes INTEGER,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS class_registry (
                class_name TEXT PRIMARY KEY,
                last_seen_at TEXT NOT NULL,
                source TEXT DEFAULT 'model',
                sample_count INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_upload_logs_created_at ON upload_logs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_training_runs_created_at ON training_runs(created_at DESC);
            """
        )


def upsert_classes(classes: List[str], source: str = "model", db_path: Path | str = DEFAULT_DB_PATH) -> None:
    if not classes:
        return
    now = _utc_now()
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO class_registry (class_name, last_seen_at, source)
            VALUES (?, ?, ?)
            ON CONFLICT(class_name) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                source=excluded.source;
            """,
            [(c, now, source) for c in classes],
        )


def log_prediction(
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    file_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
    source: str = "ui",
    low_conf_threshold: float = 0.55,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    payload = json.dumps(probabilities)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (
                created_at, file_name, predicted_class, confidence, latency_ms,
                source, is_low_confidence, probabilities_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                file_name,
                predicted_class,
                float(confidence),
                float(latency_ms) if latency_ms is not None else None,
                source,
                1 if confidence < low_conf_threshold else 0,
                payload,
            ),
        )


def log_upload(
    class_name: Optional[str],
    uploaded_count: int,
    skipped_files: List[str],
    total_uploaded_data: int,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO upload_logs (
                created_at, class_name, uploaded_count, skipped_count,
                total_uploaded_data, skipped_files_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                class_name,
                int(uploaded_count),
                int(len(skipped_files)),
                int(total_uploaded_data),
                json.dumps(skipped_files),
            ),
        )


def log_training_run(
    status: str,
    message: str,
    total_samples: Optional[int] = None,
    num_classes: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    details: Optional[Dict[str, Any]] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    metrics = metrics or {}
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO training_runs (
                created_at, status, message, total_samples, num_classes,
                accuracy, precision, recall, f1_score, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                status,
                message,
                total_samples,
                num_classes,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                json.dumps(details or {}),
            ),
        )


def get_prediction_history(limit: int = 100, db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT created_at, file_name, predicted_class, confidence, latency_ms, source, is_low_confidence
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_training_history(limit: int = 50, db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT created_at, status, message, total_samples, num_classes, accuracy, precision, recall, f1_score
            FROM training_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_database_status(db_path: Path | str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        predictions = conn.execute("SELECT COUNT(*) AS n FROM prediction_logs").fetchone()["n"]
        uploads = conn.execute("SELECT COUNT(*) AS n FROM upload_logs").fetchone()["n"]
        trainings = conn.execute("SELECT COUNT(*) AS n FROM training_runs").fetchone()["n"]
        classes = conn.execute("SELECT COUNT(*) AS n FROM class_registry").fetchone()["n"]
        latest_prediction = conn.execute(
            "SELECT created_at FROM prediction_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest_training = conn.execute(
            "SELECT created_at FROM training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    db_file = Path(db_path)
    size_bytes = db_file.stat().st_size if db_file.exists() else 0

    return {
        "db_path": str(db_file),
        "db_size_bytes": int(size_bytes),
        "prediction_logs": int(predictions),
        "upload_logs": int(uploads),
        "training_runs": int(trainings),
        "registered_classes": int(classes),
        "last_prediction_at": latest_prediction["created_at"] if latest_prediction else None,
        "last_training_at": latest_training["created_at"] if latest_training else None,
    }


def safe_call(func, *args, **kwargs):
    """Guard database writes so API operations never fail due to logging issues."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.warning("Database operation failed: %s", exc)
        return None
