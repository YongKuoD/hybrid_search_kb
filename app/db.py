"""
SQLite 数据库 — 文档 & 分块存储（上传解析后、导入 ES/Milvus 前的中间层）
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from engine.fields import normalize_record_fields

logger = logging.getLogger("db")

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "chunks.db"

_CHUNK_COLS = (
    "编号", "需求模块", "交易功能", "类型", "名称", "内容", "关联"
)


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_chunks_schema(conn: sqlite3.Connection) -> None:
    """为 chunks 表补充「编号」列，并拆分历史合并字段。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "编号" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN 编号 TEXT DEFAULT ''")
        logger.info("chunks 表已添加列: 编号")

    rows = conn.execute(
        "SELECT id, 需求模块 FROM chunks WHERE 编号 IS NULL OR 编号 = ''"
    ).fetchall()
    for row in rows:
        rec = normalize_record_fields({"需求模块": row["需求模块"] or ""})
        if rec.get("编号") or rec.get("需求模块") != (row["需求模块"] or ""):
            conn.execute(
                "UPDATE chunks SET 编号 = ?, 需求模块 = ? WHERE id = ?",
                (rec.get("编号", ""), rec.get("需求模块", ""), row["id"]),
            )


def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            kb_id       TEXT NOT NULL,
            filename    TEXT NOT NULL DEFAULT '',
            format      TEXT NOT NULL DEFAULT 'xmind',
            created_at  TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            imported_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            编号        TEXT DEFAULT '',
            需求模块    TEXT DEFAULT '',
            交易功能    TEXT DEFAULT '',
            类型        TEXT DEFAULT '',
            名称        TEXT DEFAULT '',
            内容        TEXT DEFAULT '',
            关联        TEXT DEFAULT '',
            imported    INTEGER NOT NULL DEFAULT 0,
            es_id       TEXT DEFAULT '',
            UNIQUE(document_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id);
    """)
    _migrate_chunks_schema(conn)
    conn.commit()
    conn.close()
    logger.info("SQLite 数据库初始化完成: %s", DB_PATH)


def _chunk_row_from_record(rec: dict) -> tuple:
    rec = normalize_record_fields(rec)
    return tuple(rec.get(col, "") for col in _CHUNK_COLS)


def _chunk_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    return normalize_record_fields(d)


def insert_document(kb_id: str, filename: str, records: list[dict]) -> str:
    conn = _connect()
    doc_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, format, created_at, chunk_count, imported_count) "
        "VALUES (?, ?, ?, ?, ?, ?, 0)",
        (doc_id, kb_id, filename, "xmind", now, len(records)),
    )

    col_list = ", ".join(_CHUNK_COLS)
    placeholders = ", ".join("?" * len(_CHUNK_COLS))
    for i, rec in enumerate(records):
        vals = _chunk_row_from_record(rec)
        conn.execute(
            f"INSERT INTO chunks (document_id, chunk_index, {col_list}) "
            f"VALUES (?, ?, {placeholders})",
            (doc_id, i + 1, *vals),
        )

    conn.commit()
    conn.close()
    return doc_id


def list_documents(kb_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, kb_id, filename, format, created_at, chunk_count, imported_count "
        "FROM documents WHERE kb_id = ? ORDER BY created_at DESC",
        (kb_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, kb_id, filename, format, created_at, chunk_count, imported_count "
        "FROM documents WHERE id = ?",
        (doc_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_document_chunks(doc_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, document_id, chunk_index, 编号, 需求模块, 交易功能, 类型, 名称, 内容, 关联, "
        "imported, es_id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    conn.close()
    return [_chunk_dict(r) for r in rows]


def delete_document(doc_id: str):
    conn = _connect()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def mark_chunks_imported(doc_id: str, chunk_indices: list[int], es_ids: list[str]):
    conn = _connect()
    for idx, es_id in zip(chunk_indices, es_ids):
        conn.execute(
            "UPDATE chunks SET imported = 1, es_id = ? WHERE document_id = ? AND chunk_index = ?",
            (es_id, doc_id, idx),
        )
    imported = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ? AND imported = 1",
        (doc_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE documents SET imported_count = ? WHERE id = ?",
        (imported, doc_id),
    )
    conn.commit()
    conn.close()


def get_unimported_chunks(doc_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, document_id, chunk_index, 编号, 需求模块, 交易功能, 类型, 名称, 内容, 关联 "
        "FROM chunks WHERE document_id = ? AND imported = 0 ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    conn.close()
    return [_chunk_dict(r) for r in rows]
