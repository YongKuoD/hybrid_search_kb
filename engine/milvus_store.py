"""
Milvus 向量存储 — 语义检索引擎
"""

import logging

import numpy as np
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility,
)
from engine.config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    VECTOR_SIZE,
)
from engine.embedding import embedder

logger = logging.getLogger("milvus_store")


class MilvusStore:
    """向量检索引擎"""

    def __init__(self, collection_name: str | None = None):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self._collection: Collection | None = None
        self._collection_name = collection_name or MILVUS_COLLECTION
        logger.info("MilvusStore 初始化: host=%s:%s collection=%s", MILVUS_HOST, MILVUS_PORT, self._collection_name)

    # ================================================================
    # Collection 管理
    # ================================================================

    def ping(self) -> bool:
        """测试 Milvus 连通性。"""
        try:
            # list_collections 会触发实际的 gRPC 调用
            colls = utility.list_collections()
            logger.info("Milvus 连通: %d 个 collection", len(colls))
            return True
        except Exception as e:
            logger.error("Milvus ping 失败 [%s:%s]: %s", MILVUS_HOST, MILVUS_PORT, e)
            raise ConnectionError(f"无法连接 Milvus ({MILVUS_HOST}:{MILVUS_PORT}): {e}") from e

    # ================================================================
    # Collection 管理
    # ================================================================

    def collection_exists(self) -> bool:
        return utility.has_collection(self._collection_name)

    @staticmethod
    def _embedding_dim_of(coll: Collection) -> int | None:
        for field in coll.schema.fields:
            if field.name == "embedding":
                return int(field.params.get("dim", 0))
        return None

    def create_collection(self, force: bool = False):
        """创建 Milvus Collection"""
        target_dim = embedder.dim
        if self.collection_exists():
            existing = Collection(self._collection_name)
            existing_dim = self._embedding_dim_of(existing)
            if force or (existing_dim is not None and existing_dim != target_dim):
                if existing_dim != target_dim:
                    logger.warning(
                        "Milvus collection %s 向量维度 %s 与当前模型 %s 不一致，重建 collection",
                        self._collection_name, existing_dim, target_dim,
                    )
                utility.drop_collection(self._collection_name)
                self._collection = None
            else:
                self._collection = existing
                return

        # Schema
        id_field = FieldSchema(
            name="id", dtype=DataType.VARCHAR,
            is_primary=True, max_length=64,
        )
        emb_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR,
            dim=embedder.dim,
        )
        text_field = FieldSchema(
            name="text", dtype=DataType.VARCHAR,
            max_length=4096,
        )

        schema = CollectionSchema(
            fields=[id_field, emb_field, text_field],
            description="柜面业务知识库 - 向量索引",
        )

        self._collection = Collection(name=self._collection_name, schema=schema)

        # 建 IVF_FLAT 索引
        index_params = {
            "metric_type": "IP",       # inner product (配合归一化向量 = 余弦相似度)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding", index_params=index_params,
        )

    def delete_collection(self):
        """删除整个 Collection（用于删除知识库）。"""
        if self.collection_exists():
            utility.drop_collection(self._collection_name)
            self._collection = None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = Collection(self._collection_name)
        return self._collection

    # ================================================================
    # 数据写入
    # ================================================================

    @staticmethod
    def _vector_row(embedding: np.ndarray, expected_dim: int) -> list[float]:
        """将 numpy 向量规范为一维 list，长度须等于 collection dim。"""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != expected_dim:
            raise ValueError(
                f"向量维度 {vec.shape[0]} 与 Milvus collection dim={expected_dim} 不一致，"
                f"请删除知识库后重建（EMBEDDING_DIM 须与模型输出一致）"
            )
        return vec.tolist()

    def insert(self, doc_id: str, text: str, embedding: np.ndarray):
        """写入单条"""
        dim = self._embedding_dim_of(self.collection) or embedder.dim
        self.collection.insert([
            [doc_id],
            [self._vector_row(embedding, dim)],
            [text],
        ])

    def insert_batch(self, items: list[tuple[str, str, np.ndarray]]):
        """批量写入 [(doc_id, text, embedding), ...]"""
        dim = self._embedding_dim_of(self.collection) or embedder.dim
        ids = [item[0] for item in items]
        texts = [item[1] for item in items]
        embs = [self._vector_row(item[2], dim) for item in items]
        self.collection.insert([ids, embs, texts])

    def flush(self):
        """落盘 + 建索引后加载"""
        self.collection.flush()
        self.collection.load()

    def delete(self, doc_id: str):
        self.collection.delete(f'id == "{doc_id}"')

    # ================================================================
    # 向量检索
    # ================================================================

    def search(self, query_text: str, top_k: int = VECTOR_SIZE) -> list[dict]:
        """向量相似度检索"""
        query_vec = embedder.encode_query(query_text)

        self.collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=[query_vec.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text"],
        )

        return [
            {
                "id": hits.id,
                "score": hits.distance,   # IP score, 越高越相似
                "text": hits.entity.get("text", ""),
            }
            for hits in results[0]
        ]
