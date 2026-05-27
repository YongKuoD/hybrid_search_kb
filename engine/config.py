"""
混合检索知识库 — 配置中心
ES (BM25) + Milvus (向量) 双路检索

主机地址可通过环境变量覆盖（Docker 部署时使用服务名）：
  ES_HOST, ES_PORT, MILVUS_HOST, MILVUS_PORT
"""

import os
from dataclasses import dataclass, field
from typing import List

# ============================================================
# Elasticsearch
# ============================================================
ES_HOST = os.environ.get("ES_HOST", "localhost")
ES_PORT = int(os.environ.get("ES_PORT", "9200"))
# 建索引（IK + ngram）在容器内可能超过默认 10s，创建知识库时需更长超时
ES_REQUEST_TIMEOUT = int(os.environ.get("ES_REQUEST_TIMEOUT", "120"))
ES_INDEX = "kb_cabinet"  # 柜面业务知识库

# ============================================================
# Milvus
# ============================================================
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = "kb_cabinet"

# ============================================================
# Embedding（远程 API，OpenAI 兼容 /v1/embeddings）
# ============================================================
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_API_MODEL = os.environ.get("EMBEDDING_API_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_BATCH = int(os.environ.get("EMBEDDING_BATCH", "32"))
# 向量维度（Milvus schema + API 校验）；默认 4096，与 Qwen3-Embedding-8B 等模型一致
_dim_raw = os.environ.get("EMBEDDING_DIM", "").strip()
EMBEDDING_DIM: int = int(_dim_raw) if _dim_raw else 4096

# ============================================================
# 字段定义
# ============================================================
# 业务字段及其在 ES BM25 中的 boost 权重
FIELD_BOOST: dict = {
    "编号": 8.0,
    "需求模块": 2.0,
    "交易功能": 1.5,
    "类型": 0.5,
    "名称": 5.0,
    # "内容": 2.0,
    # "关联": 1.0,
}

# cross_fields 主匹配字段（不含需求模块/内容：前者各条相同，后者过长易喧宾夺主）
CROSS_FIELD_BOOST: dict = {
    "名称": 8.0,
    "交易功能": 3.0,
    "需求模块": 2.0,
    "类型": 0.5,
    "关联": 1.0,
}
# 交易功能 / 需求模块单独 match（产品名、模块名常在此，如「丰收缴费通」）
TXN_FUNC_TERM_BOOST = 14.0
MODULE_TERM_BOOST = 10.0
# 名称命中单个检索词（如「人脸识别」）的额外权重
NAME_TERM_BOOST = 28.0
# 整句匹配需求模块（与 cross_fields 互补）
MODULE_MATCH_BOOST = 2.0
CONTENT_MATCH_BOOST = 0.5
# 焦点词（多词查询时更具体的一个，如「人脸识别」）命中名称时 function_score 乘数
NAME_FOCUS_WEIGHT = 80.0
# 焦点词仅出现在「内容」、名称未命中时降权
CONTENT_ONLY_FOCUS_WEIGHT = 0.12
# RRF：名称含焦点词时额外加分系数
RRF_NAME_FOCUS_BONUS = 3.0

# 远程 embedding 拼接字段（不含「内容」）
EMBEDDING_FIELDS: List[str] = ["编号", "需求模块", "交易功能", "类型", "名称"]

# 有 ngram 子字段的 ES 列（「内容」为长文本，不用 ngram）
NGRAM_FIELDS: List[str] = ["需求模块", "交易功能", "名称"]

# ============================================================
# 检索参数
# ============================================================
BM25_SIZE = 30       # BM25 路召回的候选数
VECTOR_SIZE = 30     # 向量路召回的候选数
RRF_K = 60           # RRF 平滑常数
FINAL_TOP_K = 10     # 最终返回给用户的结果数

# cross_fields 查询的最小匹配比例
# 查询词 ≤2 个: 100% 匹配；3-4 个: 75%；5+ 个: 60%
def min_should_match(query: str) -> str:
    token_count = len(query.strip().split())
    if token_count <= 2:
        return "100%"
    elif token_count <= 4:
        return "75%"
    else:
        return "60%"


def pick_focus_term(query: str) -> str:
    """多词查询时取更具体的词（默认最后一个；若更长则取最长）。"""
    terms = [t for t in query.strip().split() if t]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    last, longest = terms[-1], max(terms, key=len)
    return longest if len(longest) > len(last) else last


def cross_field_min_should_match(query: str) -> str:
    """跨字段匹配：2 个词时允许只命中 1 个（避免必须凑齐需求模块里的「可读卡」）。"""
    token_count = len(query.strip().split())
    if token_count <= 1:
        return "100%"
    if token_count == 2:
        return "1"
    return min_should_match(query)
