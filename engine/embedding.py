"""
Embedding 服务 — 远程 API（OpenAI 兼容 /v1/embeddings）

需配置 EMBEDDING_API_URL、EMBEDDING_API_KEY、EMBEDDING_API_MODEL。
可选 EMBEDDING_DIM：指定向量维度（Milvus schema）；未设置则从 API 探测。
接口：encode() / encode_query() / dim
"""

import logging
import time
from urllib import request, error as urllib_error
import json

import numpy as np
from engine.config import (
    EMBEDDING_BATCH,
    EMBEDDING_DIM,
    EMBEDDING_API_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_API_MODEL,
)

logger = logging.getLogger("embedding")


class RemoteEmbeddingClient:
    """远程 Embedding API 客户端（OpenAI 兼容接口）。

    适用所有提供 /v1/embeddings 端点的服务（硅基流动、OpenAI 等）。
    请求格式: POST {base_url}/embeddings
             {"model": "...", "input": [...], "encoding_format": "float"}
    """

    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _check_vectors(self, vectors: list[np.ndarray]) -> None:
        expected = self.dim
        for i, v in enumerate(vectors):
            if len(v) != expected:
                hint = f"请修正 EMBEDDING_DIM={expected} 为模型实际输出维度 {len(v)}"
                raise RuntimeError(
                    f"Embedding 向量维度不一致: 第 {i + 1} 条为 {len(v)} 维，期望 {expected} 维。{hint}"
                )

    def encode(self, texts: list[str], batch_size: int = EMBEDDING_BATCH) -> np.ndarray:
        """将文本列表编码为向量矩阵 (N, dim)。自动分批。"""
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._call_api(batch)
            self._check_vectors(vectors)
            all_vectors.extend(vectors)
        return self._normalize(np.array(all_vectors, dtype=np.float32))

    def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询文本。"""
        vectors = self._call_api([query])
        self._check_vectors(vectors)
        return self._normalize(vectors[0].astype(np.float32))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """L2 归一化：内积 = 余弦相似度。"""
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def _call_api(self, texts: list[str], max_retries: int = 3) -> list[np.ndarray]:
        """调用远程 embeddings API，带重试。"""
        url = f"{self.base_url}/embeddings"
        body = json.dumps({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }).encode("utf-8")

        req = request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")

        last_err = None
        for attempt in range(max_retries):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                items = sorted(data["data"], key=lambda x: x["index"])
                return [
                    np.array(item["embedding"], dtype=np.float32)
                    for item in items
                ]
            except urllib_error.HTTPError as e:
                last_err = e
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                logger.warning(
                    "远程 API 错误 (attempt %d/%d): HTTP %s %s",
                    attempt + 1, max_retries, e.code, err_body,
                )
                if e.code == 401:
                    raise RuntimeError(
                        "Embedding API Key 无效 (401)。请检查 EMBEDDING_API_KEY 环境变量。"
                    ) from e
                if e.code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if attempt < max_retries - 1:
                    time.sleep(1)
            except urllib_error.URLError as e:
                last_err = e
                logger.warning("远程 API 网络错误 (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        raise ConnectionError(
            f"Embedding API 不可达 ({url}), 已重试 {max_retries} 次。最后错误: {last_err}"
        )


def _create_embedder() -> RemoteEmbeddingClient:
    if not EMBEDDING_API_KEY:
        raise RuntimeError(
            "未设置 EMBEDDING_API_KEY 环境变量。请在 .env 中设置: EMBEDDING_API_KEY=sk-xxxx"
        )
    if not EMBEDDING_API_URL:
        raise RuntimeError("未设置 EMBEDDING_API_URL 环境变量。")
    logger.info(
        "Embedding: url=%s model=%s dim=%d",
        EMBEDDING_API_URL, EMBEDDING_API_MODEL, EMBEDDING_DIM,
    )
    return RemoteEmbeddingClient(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_URL,
        model=EMBEDDING_API_MODEL,
        dim=EMBEDDING_DIM,
    )


# 全局单例（首次 encode/search 时初始化）
_embedder: RemoteEmbeddingClient | None = None


class EmbeddingService:
    """延迟初始化的远程 Embedding 门面。"""

    def _client(self) -> RemoteEmbeddingClient:
        global _embedder
        if _embedder is None:
            _embedder = _create_embedder()
        return _embedder

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._client().encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._client().encode_query(query)

    @property
    def dim(self) -> int:
        return self._client().dim


embedder = EmbeddingService()
