"""
混合检索引擎 — BM25 + 向量, RRF 融合排序
"""

from engine.config import (
    BM25_SIZE, VECTOR_SIZE, RRF_K, FINAL_TOP_K,
    RRF_NAME_FOCUS_BONUS, pick_focus_term,
)
from engine.es_store import ESStore
from engine.milvus_store import MilvusStore


def rrf_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = RRF_K,
    focus_term: str = "",
) -> list[dict]:
    """
    Reciprocal Rank Fusion — 按排名融合；名称命中焦点词时额外加分。
    """
    scores: dict[str, float] = {}

    def _name_focus_bonus(hit: dict) -> float:
        if not focus_term:
            return 0.0
        src = hit.get("source") or {}
        name = src.get("名称", "")
        if focus_term in name:
            return RRF_NAME_FOCUS_BONUS / (k + 1)
        return 0.0

    for rank, hit in enumerate(bm25_results, start=1):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank) + _name_focus_bonus(hit)

    for rank, hit in enumerate(vector_results, start=1):
        scores[hit["id"]] = scores.get(hit["id"], 0) + 1.0 / (k + rank)

    # 按融合分数降序排列
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{"id": doc_id, "rrf_score": scores[doc_id]} for doc_id in sorted_ids]


class HybridSearcher:
    """混合检索: BM25 + 向量 → RRF 融合"""

    def __init__(self, kb_name: str | None = None):
        """
        :param kb_name: 知识库名称。若提供，ES 索引 = kb_{kb_name}，Milvus 集合 = kb_{kb_name}。
        """
        es_index = f"kb_{kb_name}" if kb_name else None
        mv_coll = f"kb_{kb_name}" if kb_name else None
        self.es = ESStore(index_name=es_index)
        self.mv = MilvusStore(collection_name=mv_coll)

    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        bm25_size: int = BM25_SIZE,
        vector_size: int = VECTOR_SIZE,
        bm25_threshold: float = -1.0,
        vector_threshold: float = -1.0,
        verbose: bool = False,
    ) -> list[dict]:
        """
        执行混合检索:
        1. ES BM25 检索
        2. Milvus 向量检索
        3. 按阈值过滤低分结果
        4. RRF 融合排序
        5. 取 top_k 返回, 附带 ES 原始字段
        """

        # 两路并行检索
        bm25_hits = self.es.search(query, size=bm25_size)
        vector_hits = self.mv.search(query, top_k=vector_size)

        # 按阈值过滤（<= 0 表示不过滤，与检索页「0 = 不过滤」一致）
        raw_bm25 = len(bm25_hits)
        raw_vec = len(vector_hits)
        if bm25_threshold > 0:
            bm25_hits = [h for h in bm25_hits if h["score"] >= bm25_threshold]
        if vector_threshold > 0:
            vector_hits = [h for h in vector_hits if h["score"] >= vector_threshold]

        if verbose:
            print(f"[BM25]  召回 {len(bm25_hits)} 条 (原始 {raw_bm25}, threshold>={bm25_threshold})")
            print(f"[向量]  召回 {len(vector_hits)} 条 (原始 {raw_vec}, threshold>={vector_threshold})")

        # RRF 融合（多词查询时对「名称=焦点词」结果额外加分）
        focus = pick_focus_term(query)
        merged = rrf_fusion(bm25_hits, vector_hits, focus_term=focus)

        # 构建 ID → ES source 的索引 (用于回填结构化字段)
        es_by_id = {hit["id"]: hit for hit in bm25_hits}

        # 对于只出现在向量路的结果，去 ES 按 ID 补取 _source
        missing_ids = [item["id"] for item in merged[:top_k] if item["id"] not in es_by_id]
        if missing_ids:
            try:
                mget_result = self.es.es.mget(index=self.es.index_name, body={"ids": missing_ids})
                for doc in mget_result.get("docs", []):
                    if doc.get("found"):
                        es_by_id[doc["_id"]] = {
                            "id": doc["_id"],
                            "score": None,
                            "source": doc["_source"],
                            "highlight": {},
                        }
            except Exception:
                pass  # mget 失败不影响主流程

        # 组装最终结果
        results = []
        for item in merged[:top_k]:
            doc_id = item["id"]
            es_hit = es_by_id.get(doc_id)
            result = {
                "id": doc_id,
                "rrf_score": round(item["rrf_score"], 4),
                "source": es_hit["source"] if es_hit else None,
                "bm25_score": (
                    round(es_hit["score"], 2)
                    if es_hit and es_hit.get("score") is not None
                    else None
                ),
                "highlight": es_hit.get("highlight") if es_hit else None,
            }

            # 查找向量路信息
            vec_match = next((v for v in vector_hits if v["id"] == doc_id), None)
            if vec_match:
                result["vector_score"] = round(vec_match["score"], 4)

            results.append(result)

        return {
            "results": results,
            "stats": {
                "bm25_total": raw_bm25,
                "bm25_filtered": len(bm25_hits),
                "vector_total": raw_vec,
                "vector_filtered": len(vector_hits),
                "bm25_threshold": bm25_threshold,
                "vector_threshold": vector_threshold,
            },
        }

    def search_pretty(self, query: str) -> str:
        """返回格式化的搜索结果文本"""
        result = self.search(query, verbose=True)
        results = result["results"]

        if not results:
            return "未找到匹配结果。"

        lines = [f"查询: {query}", "=" * 60]
        for i, r in enumerate(results, 1):
            src = r.get("source") or {}
            node_type = src.get("类型", "")
            type_label = f"[{node_type}] " if node_type else ""
            lines.append(f"\n--- 结果 #{i} {type_label}(RRF: {r['rrf_score']}) ---")
            if src.get("编号"):
                lines.append(f"编号: {src['编号']}")
            lines.append(f"需求模块: {src.get('需求模块', '-')}")
            if src.get("交易功能"):
                lines.append(f"交易功能: {src['交易功能']}")
            if src.get("名称"):
                lines.append(f"名称: {src['名称']}")
            if src.get("内容"):
                lines.append(f"内容: {src['内容'][:120]}")
            if src.get("关联"):
                lines.append(f"关联: {src['关联']}")
            # 兼容旧字段
            if not src.get("名称"):
                lines.append(f"流程: {src.get('流程', '-')}")
                lines.append(f"步骤: {src.get('步骤', '-')}")

            if r.get("highlight"):
                hl = r["highlight"]
                for field, snippets in hl.items():
                    for snippet in snippets:
                        lines.append(f"  ⤷ 命中 [{field}]: {snippet}")

        return "\n".join(lines)
