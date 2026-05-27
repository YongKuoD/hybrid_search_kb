"""
数据摄入 — 同时写入 ES 和 Milvus, 保持 ID 一致
"""

import json
import csv
import uuid
from pathlib import Path
from engine.config import EMBEDDING_FIELDS, EMBEDDING_BATCH
from engine.embedding import embedder
from engine.es_store import ESStore
from engine.fields import normalize_record_fields
from engine.milvus_store import MilvusStore


def build_embedding_text(fields: dict) -> str:
    """
    将结构化字段拼接为 embedding 文本。

    优先级顺序（来自 EMBEDDING_FIELDS，不含「内容」— 内容仅走 ES BM25）:
      需求模块 > 交易功能 > 类型 > 名称

    兼容旧字段（流程/步骤）作为 fallback。
    """
    parts = []
    for field in EMBEDDING_FIELDS:
        value = fields.get(field, "")
        if value:
            parts.append(f"{field}: {value}")
    # 兼容旧字段：如果 名称/内容 没值，回退 流程/步骤
    if not fields.get("名称") and not fields.get("内容"):
        for fallback in ("流程", "步骤"):
            value = fields.get(fallback, "")
            if value:
                parts.append(f"{fallback}: {value}")
    return "\n".join(parts)


def build_doc_id(index: int) -> str:
    """生成统一的文档 ID（仅 CLI 批量导入且无显式 id 时使用，避免批次内冲突）。"""
    return f"doc_{index:06d}"


def new_doc_id() -> str:
    """全局唯一文档 ID，避免多次 Web 导入时 doc_000000 互相覆盖。"""
    return uuid.uuid4().hex


class Ingester:
    """同时写入 ES + Milvus"""

    def __init__(self, kb_name: str | None = None):
        """
        :param kb_name: 知识库名称。若提供，ES 索引 = kb_{kb_name}，Milvus 集合 = kb_{kb_name}。
                        若为 None，使用 config 中的默认值。
        """
        es_index = f"kb_{kb_name}" if kb_name else None
        mv_coll = f"kb_{kb_name}" if kb_name else None
        self.es = ESStore(index_name=es_index)
        self.mv = MilvusStore(collection_name=mv_coll)

    def ensure_stores(self, force: bool = False):
        """确保索引和 collection 已创建（含连通性检查）"""
        self.es.ping()
        self.mv.ping()
        self.es.create_index(force=force)
        self.mv.create_collection(force=force)

    def ingest_json(self, path: str):
        """
        从 JSON 文件摄入。
        格式: [{"需求模块": "...", "交易功能": "...", "流程": "...", "步骤": "..."}, ...]
        """
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self._ingest_records(records)

    def ingest_csv(self, path: str):
        """从 CSV 文件摄入"""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        self._ingest_records(records)

    def ingest_xmind(self, path: str):
        """
        从 XMind 文件摄入。

        XMind 层级结构:
          需求模块(编号-描述) → 交易/功能(可选) → 内容类型(业务流程/业务规则/页面控制…)
          流程 = 业务流程的直接子节点标题
          步骤 = 业务规则的直接子节点标题
        """
        from parser.xmind_parser import xmind_to_records

        records = xmind_to_records(path)
        if not records:
            print(f"XMind 文件未解析出有效记录: {path}")
            return
        print(f"XMind 解析出 {len(records)} 条知识库记录")
        self._ingest_records(records)

    def _ingest_records(self, records: list[dict]) -> list[str]:
        """核心摄入逻辑: 批量写入 ES + Milvus。

        若 record 中包含 'id' 字段则使用，否则用 build_doc_id(i) 自动生成。
        返回实际使用的 ES doc_id 列表。
        """
        records = [normalize_record_fields(r) for r in records]
        total = len(records)
        ids = [r.get("id") or new_doc_id() for r in records]

        # 1. 准备 Milvus 数据: 拼接文本 → 批量 embedding
        texts = [build_embedding_text(rec) for rec in records]
        print(f"正在为 {total} 条记录生成 embedding...")
        embeddings = embedder.encode(texts)

        # 2. 批量写入 ES
        print(f"正在写入 ES ({total} 条)...")
        es_docs = [
            (ids[i], rec) for i, rec in enumerate(records)
        ]
        self.es.insert_batch(es_docs)

        # 3. 批量写入 Milvus
        print(f"正在写入 Milvus ({total} 条)...")
        mv_items = [
            (ids[i], texts[i], embeddings[i]) for i in range(total)
        ]
        self.mv.insert_batch(mv_items)
        self.mv.flush()

        print(f"摄入完成: {total} 条记录已写入 ES + Milvus")
        return ids


def ingest_demo_data(ingester: Ingester):
    """
    写入你示例中的那条数据作为 demo。
    """
    demo = [
        {
            "需求模块": "CP0120025070996-智柜借记卡可读可销卡",
            "交易功能": "",
            "流程": "可读可销卡转账支取流程、可读卡销卡现金支取流程",
            "步骤": (
                "插入借记卡/磁条卡、输入密码、插入身份证、人脸识别、"
                "选择支取方式、签字并提交、柜员指纹检验、出钞、交易后验证"
            ),
        }
    ]
    ingester._ingest_records(demo)
