"""
混合检索知识库 — Web 管理页面后端

提供 REST API：
  GET  /api/kb              → 列出所有知识库
  POST /api/kb              → 创建知识库  body: {"name": "显示名称"}
  DELETE /api/kb/{kb_id}    → 删除知识库
  POST /api/kb/{kb_id}/upload → 上传 XMind 文件并解析导入
  POST /api/kb/{kb_id}/search → 混合检索
  GET  /api/kb/{kb_id}/stats  → 知识库统计

内部 ID 与 显示名称 分离：
  - 显示名称（name）: 用户输入，可含中文，用于 UI 展示
  - 内部 ID  (id)   : 自动生成，仅小写 ASCII，用于 ES 索引 / Milvus 集合 / API 路由

启动: python server.py   然后浏览器访问 http://localhost:8800
"""

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("server")

from app import db  # SQLite 文档/分块存储
db.init_db()

# ─── 项目根路径 ────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent          # app/
PROJECT_DIR = APP_DIR.parent                        # 项目根目录
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
META_FILE = DATA_DIR / "kb_meta.json"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="混合检索知识库")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── KB 元数据管理 ──────────────────────────────────

def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text("utf-8"))
    return {}


def _save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


# ─── ID 生成 ──────────────────────────────────────

def _generate_kb_id() -> str:
    """生成知识库内部 ID（UUID 短标识），用于 ES 索引名 / Milvus 集合名 / API 路由。"""
    return uuid.uuid4().hex[:12]


def _get_kb_internal(kb_id: str, meta: dict) -> dict | None:
    """用内部 ID 查找知识库元数据（兼容用户可能直接用显示名查）。"""
    if kb_id in meta:
        return meta[kb_id]
    # 兼容：按显示名查找
    for info in meta.values():
        if info.get("name") == kb_id:
            return info
    return None


def _resolve_meta(kb_id: str, meta: dict) -> dict:
    """解析 kb_id → 返回元数据条目与内部 ID。找不到抛 404。"""
    info = _get_kb_internal(kb_id, meta)
    if info is None:
        raise HTTPException(404, f"知识库「{kb_id}」不存在")
    internal_id = info["id"]
    return {"info": info, "internal_id": internal_id}


# ─── 页面路由 ──────────────────────────────────────

def _serve_html(filename: str) -> HTMLResponse:
    """读取 static 目录下的 HTML 文件"""
    path = STATIC_DIR / filename
    if path.exists():
        return HTMLResponse(path.read_text("utf-8"))
    return HTMLResponse(f"<h2>static/{filename} 未找到</h2>", status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    return _serve_html("index.html")


@app.get("/create", response_class=HTMLResponse)
def page_create():
    return _serve_html("create.html")


@app.get("/upload", response_class=HTMLResponse)
def page_upload():
    return _serve_html("upload.html")


@app.get("/search", response_class=HTMLResponse)
def page_search():
    return _serve_html("search.html")


@app.get("/records", response_class=HTMLResponse)
def page_records():
    return _serve_html("records.html")


# ─── KB CRUD ───────────────────────────────────────

@app.get("/api/kb")
def list_kbs():
    """列出所有知识库"""
    meta = _load_meta()
    result = []
    for info in meta.values():
        result.append({
            "id": info["id"],
            "name": info["name"],
            "created_at": info.get("created_at", ""),
            "doc_count": info.get("doc_count", 0),
        })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


@app.post("/api/kb")
def create_kb(body: dict):
    """创建知识库。body: {"name": "显示名称"}"""
    display_name = (body.get("name") or "").strip()
    if not display_name:
        raise HTTPException(400, "知识库名称不能为空")

    kb_id = _generate_kb_id()
    es_index = f"kb_{kb_id}"
    logger.info("创建知识库: display_name=%r → id=%r → es_index=%r", display_name, kb_id, es_index)

    meta = _load_meta()
    if kb_id in meta:
        raise HTTPException(409, f"知识库「{display_name}」已存在（ID: {kb_id}）")

    from engine.es_store import ESStore
    from engine.ingestion import Ingester

    # 先验证 ES 连通性（给明确错误信息）
    try:
        es = ESStore()
        es.ping()
    except Exception as e:
        raise HTTPException(500, f"无法连接 Elasticsearch: {e}")

    try:
        ingester = Ingester(kb_name=kb_id)
        ingester.ensure_stores()
    except Exception as e:
        raise HTTPException(500, f"存储初始化失败 [index={es_index}]: {e}")

    meta[kb_id] = {
        "id": kb_id,
        "name": display_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": 0,
    }
    _save_meta(meta)
    return {"ok": True, "id": kb_id, "name": display_name, "es_index": es_index}


@app.delete("/api/kb/{kb_id}")
def delete_kb(kb_id: str):
    """删除知识库（同时清除 ES 索引和 Milvus Collection）"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    internal_id = resolved["internal_id"]

    from engine.es_store import ESStore
    from engine.milvus_store import MilvusStore

    es = ESStore(index_name=f"kb_{internal_id}")
    mv = MilvusStore(collection_name=f"kb_{internal_id}")
    es.delete_index()
    mv.delete_collection()

    del meta[internal_id]
    _save_meta(meta)
    return {"ok": True}


@app.get("/api/kb/{kb_id}/stats")
def kb_stats(kb_id: str):
    """知识库统计"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    return resolved["info"]


# ─── 文档 & 分块（SQLite 中间层） ────────────────

@app.get("/api/kb/{kb_id}/documents")
def list_documents_endpoint(kb_id: str):
    """列出知识库中所有上传的文档（SQLite）"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    internal_id = resolved["internal_id"]

    docs = db.list_documents(internal_id)
    return [
        {
            "id": d["id"],
            "kb_id": d["kb_id"],
            "filename": d["filename"],
            "format": d["format"],
            "created_at": d["created_at"],
            "chunk_count": d["chunk_count"],
            "imported_count": d["imported_count"],
            "status": "已全部导入" if d["imported_count"] == d["chunk_count"]
                      else "部分导入" if d["imported_count"] > 0
                      else "未导入",
        }
        for d in docs
    ]


@app.get("/api/kb/{kb_id}/documents/{doc_id}")
def get_document_endpoint(kb_id: str, doc_id: str):
    """获取文档详情及其全部分块"""
    meta = _load_meta()
    _resolve_meta(kb_id, meta)  # 校验知识库存在

    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, f"文档「{doc_id}」不存在")

    chunks = db.get_document_chunks(doc_id)
    return {
        "document": doc,
        "chunks": [
            {
                "chunk_index": c["chunk_index"],
                "编号": c["编号"],
                "需求模块": c["需求模块"],
                "交易功能": c["交易功能"],
                "类型": c["类型"],
                "名称": c["名称"],
                "内容": c["内容"],
                "关联": c["关联"],
                "imported": bool(c["imported"]),
                "es_id": c["es_id"],
            }
            for c in chunks
        ],
        "format": doc["format"],
    }


@app.delete("/api/kb/{kb_id}/documents/{doc_id}")
def delete_document_endpoint(kb_id: str, doc_id: str):
    """删除文档及其全部分块（SQLite，不影响已导入 ES/Milvus 的数据）"""
    meta = _load_meta()
    _resolve_meta(kb_id, meta)

    db.delete_document(doc_id)
    return {"ok": True, "id": doc_id}


@app.post("/api/kb/{kb_id}/documents/{doc_id}/import")
def import_chunks(kb_id: str, doc_id: str, body: dict):
    """将选中的分块导入 ES + Milvus"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    internal_id = resolved["internal_id"]

    chunk_indices = body.get("chunk_indices", [])
    if not chunk_indices:
        raise HTTPException(400, "chunk_indices 不能为空")

    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, f"文档「{doc_id}」不存在")

    all_chunks = db.get_document_chunks(doc_id)
    # 筛选出需要导入的分块
    to_import = [c for c in all_chunks if c["chunk_index"] in chunk_indices and not c["imported"]]
    if not to_import:
        raise HTTPException(400, "选中的分块均已导入或不存在")

    from engine.ingestion import Ingester
    ingester = Ingester(kb_name=internal_id)
    ingester.ensure_stores()

    # 稳定唯一 id：同一分块重复导入覆盖同一条 ES/Milvus 文档，不同分块互不覆盖
    records = [
        {
            "id": f"{doc_id}_{c['chunk_index']}",
            "编号": c["编号"],
            "需求模块": c["需求模块"],
            "交易功能": c["交易功能"],
            "类型": c["类型"],
            "名称": c["名称"],
            "内容": c["内容"],
            "关联": c["关联"],
        }
        for c in to_import
    ]

    es_ids = ingester._ingest_records(records)

    # 标记已导入
    db.mark_chunks_imported(doc_id, [c["chunk_index"] for c in to_import], es_ids)

    # 更新元数据 doc_count
    from engine.es_store import ESStore
    es = ESStore(index_name=f"kb_{internal_id}")
    try:
        count_result = es.es.count(index=es.index_name)
        meta[internal_id]["doc_count"] = count_result["count"]
    except Exception:
        meta[internal_id]["doc_count"] = meta[internal_id].get("doc_count", 0) + len(to_import)
    _save_meta(meta)

    return {"ok": True, "imported": len(to_import), "message": f"已导入 {len(to_import)} 条分块"}


@app.get("/api/health")
def health_check():
    """服务健康检查 + ES / Milvus 连通性验证"""
    result = {"status": "ok", "es": "unknown", "milvus": "unknown"}

    from engine.es_store import ESStore
    from engine.milvus_store import MilvusStore

    try:
        es = ESStore(index_name="kb_cabinet")
        es.ping()
        result["es"] = "connected"
    except Exception as e:
        result["es"] = f"error: {e}"
        result["status"] = "degraded"

    try:
        mv = MilvusStore(collection_name="kb_cabinet")
        mv.ping()
        result["milvus"] = "connected"
    except Exception as e:
        result["milvus"] = f"error: {e}"
        result["status"] = "degraded"

    return result


# ─── XMind 上传解析 ────────────────────────────────

@app.post("/api/kb/{kb_id}/upload")
def upload_xmind(kb_id: str, file: UploadFile = File(...)):
    """上传 XMind 文件到指定知识库"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    internal_id = resolved["internal_id"]

    if not file.filename or not file.filename.lower().endswith(".xmind"):
        raise HTTPException(400, "仅支持 .xmind 文件")

    suffix = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from parser.xmind_parser import xmind_to_records
        records = xmind_to_records(str(save_path))
    except Exception as e:
        raise HTTPException(400, f"XMind 解析失败: {e}")

    if not records:
        raise HTTPException(400, "XMind 文件未解析出有效记录")

    # 保存到 SQLite（暂不导入 ES/Milvus）
    doc_id = db.insert_document(internal_id, file.filename, records)

    meta[internal_id]["last_upload"] = datetime.now(timezone.utc).isoformat()
    _save_meta(meta)

    try:
        save_path.unlink()
    except OSError:
        pass

    return {
        "ok": True,
        "document_id": doc_id,
        "records": len(records),
        "preview": [
            {
                "编号": r.get("编号", "")[:20],
                "需求模块": r.get("需求模块", "")[:60],
                "交易功能": r.get("交易功能", "")[:30] or "（无）",
                "类型": r.get("类型", ""),
                "名称": r.get("名称", "")[:50],
                "关联": r.get("关联", "")[:40] or "",
            }
            for r in records[:5]
        ],
    }


# ─── 检索 ──────────────────────────────────────────

@app.post("/api/kb/{kb_id}/search")
def search_kb(kb_id: str, body: dict):
    """混合检索"""
    meta = _load_meta()
    resolved = _resolve_meta(kb_id, meta)
    internal_id = resolved["internal_id"]

    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "查询文本不能为空")

    top_k = body.get("top_k", 10)
    _bm25_raw = body.get("bm25_threshold")
    bm25_threshold = float(_bm25_raw) if _bm25_raw is not None else -1.0
    _vec_raw = body.get("vector_threshold")
    vector_threshold = float(_vec_raw) if _vec_raw is not None else -1.0

    from engine.hybrid_search import HybridSearcher
    searcher = HybridSearcher(kb_name=internal_id)
    search_result = searcher.search(
        query, top_k=top_k,
        bm25_threshold=bm25_threshold,
        vector_threshold=vector_threshold,
        verbose=True,
    )
    results = search_result["results"]
    stats = search_result["stats"]

    return {
        "query": query,
        "total": len(results),
        "stats": stats,
        "results": [
            {
                "rrf_score": r["rrf_score"],
                "bm25_score": r.get("bm25_score"),
                "vector_score": r.get("vector_score"),
                "编号": (r.get("source") or {}).get("编号", ""),
                "需求模块": (r.get("source") or {}).get("需求模块", ""),
                "交易功能": (r.get("source") or {}).get("交易功能", ""),
                "类型": (r.get("source") or {}).get("类型", ""),
                "名称": (r.get("source") or {}).get("名称", ""),
                "内容": (r.get("source") or {}).get("内容", ""),
                "关联": (r.get("source") or {}).get("关联", ""),
                # 兼容旧字段
                "流程": (r.get("source") or {}).get("流程", ""),
                "步骤": (r.get("source") or {}).get("步骤", ""),
                "highlight": r.get("highlight", {}),
            }
            for r in results
        ],
    }


# ─── 启动入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("混合检索知识库 Web 服务启动: http://localhost:8800")
    uvicorn.run(app, host="0.0.0.0", port=8800, log_level="info")
