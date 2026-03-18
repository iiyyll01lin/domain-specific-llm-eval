from __future__ import annotations

from typing import Dict, List


CURRENT_REVIEWER_STATE_SCHEMA_VERSION = 2


def sqlite_migration_statements() -> List[Dict[str, object]]:
    return [
        {
            "version": 1,
            "statements": [
                """
                CREATE TABLE IF NOT EXISTS reviewer_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS reviewer_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    review_id TEXT,
                    reviewer TEXT,
                    tenant_id TEXT,
                    payload_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ],
        },
        {
            "version": 2,
            "statements": [
                "ALTER TABLE review_queue ADD COLUMN tenant_id TEXT",
                "ALTER TABLE reviewer_results ADD COLUMN tenant_id TEXT",
            ],
        },
    ]


def postgres_migration_statements() -> List[Dict[str, object]]:
    return [
        {
            "version": 1,
            "statements": [
                """
                CREATE TABLE IF NOT EXISTS reviewer_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS reviewer_audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    review_id TEXT,
                    reviewer TEXT,
                    tenant_id TEXT,
                    payload_json JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ],
        },
        {
            "version": 2,
            "statements": [
                "ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS tenant_id TEXT",
                "ALTER TABLE reviewer_results ADD COLUMN IF NOT EXISTS tenant_id TEXT",
            ],
        },
    ]