"""
Elasticsearch 索引管理 & BM25 检索
"""

import logging
import re

from elasticsearch import Elasticsearch, helpers
from engine.config import (
    ES_HOST, ES_PORT, ES_INDEX, ES_REQUEST_TIMEOUT,
    NGRAM_FIELDS, BM25_SIZE,
    CROSS_FIELD_BOOST, NAME_TERM_BOOST,
    TXN_FUNC_TERM_BOOST, MODULE_TERM_BOOST,
    MODULE_MATCH_BOOST, CONTENT_MATCH_BOOST,
    NAME_FOCUS_WEIGHT, CONTENT_ONLY_FOCUS_WEIGHT,
    cross_field_min_should_match, pick_focus_term,
)

logger = logging.getLogger("es_store")

# 查询形如 CP0000000000000 时走编号精确 / 前缀匹配
_CP_ID_QUERY_RE = re.compile(r"^[A-Za-z]{2}\d", re.I)
_CP_ID_IN_QUERY_RE = re.compile(r"([A-Za-z]{2}\d{8,})", re.I)


class ESStore:
    """BM25 关键词检索引擎"""

    def __init__(self, index_name: str | None = None):
        self.es = Elasticsearch(
            f"http://{ES_HOST}:{ES_PORT}",
            request_timeout=ES_REQUEST_TIMEOUT,
        )
        self._index = index_name or ES_INDEX
        logger.info("ESStore 初始化: host=%s:%s index=%s", ES_HOST, ES_PORT, self._index)

    @property
    def index_name(self) -> str:
        return self._index

    # ================================================================
    # 索引管理
    # ================================================================

    def ping(self) -> bool:
        """测试 ES 连通性，失败时给出明确错误。"""
        try:
            info = self.es.info()
            logger.info("ES 连通: version=%s cluster=%s", info["version"]["number"], info["cluster_name"])
            return True
        except Exception as e:
            logger.error("ES ping 失败 [%s:%s]: %s", ES_HOST, ES_PORT, e)
            raise ConnectionError(f"无法连接 Elasticsearch ({ES_HOST}:{ES_PORT}): {e}") from e

    def index_exists(self) -> bool:
        try:
            logger.debug("检查索引是否存在: %s", self._index)
            return self.es.indices.exists(index=self._index)
        except Exception as e:
            logger.error("检查索引失败 [index=%s]: %s", self._index, e)
            raise RuntimeError(f"ES 索引检查失败 [index={self._index}]: {e}") from e

    def create_index(self, force: bool = False):
        """创建索引，配置 ik_max_word 分词 + ngram 子字段 + copy_to"""
        if self.index_exists():
            if force:
                self.es.indices.delete(index=self._index)
            else:
                return  # 已存在, 跳过

        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ik_analyzer": {
                            "type": "custom",
                            "tokenizer": "ik_max_word",
                        },
                        "ngram_analyzer": {
                            "type": "custom",
                            "tokenizer": "ngram_tokenizer",
                            "filter": ["lowercase"],
                        },
                    },
                    "tokenizer": {
                        "ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 2,
                            "max_gram": 3,
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "编号": {
                        "type": "keyword",
                    },
                    "需求模块": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "copy_to": "full_text",
                        "fields": {
                            "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        },
                    },
                    "交易功能": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "copy_to": "full_text",
                        "fields": {
                            "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        },
                    },
                    "类型": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "copy_to": "full_text",
                        "fields": {
                            "keyword": {"type": "keyword"},
                        },
                    },
                    "名称": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "copy_to": "full_text",
                        "fields": {
                            "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        },
                    },
                    "内容": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        # 长文本：可 BM25 检索，不 copy_to full_text、不加 ngram
                    },
                    "关联": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "copy_to": "full_text",
                    },
                    "full_text": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                    },
                }
            },
        }

        try:
            logger.info("ES 开始创建索引: %s (timeout=%ss)", self._index, ES_REQUEST_TIMEOUT)
            self.es.indices.create(
                index=self._index,
                body=body,
                request_timeout=ES_REQUEST_TIMEOUT,
            )
            logger.info("ES 索引创建成功: %s", self._index)
        except Exception as e:
            err_msg = str(e)
            # 检测 IK 分词器缺失
            if "ik_max_word" in err_msg or "analyzer" in err_msg.lower():
                logger.error(
                    "ES 索引创建失败 [%s]: analysis-ik 插件可能未安装。"
                    "请检查 ES 容器日志: docker logs es_kb | grep -i ik",
                    self._index,
                )
                raise RuntimeError(
                    f"ES 索引创建失败 [{self._index}]: IK 中文分词插件 (analysis-ik) 未安装。\n"
                    f"请在 ES 容器中手动安装: docker exec -it es_kb ./bin/elasticsearch-plugin install --batch "
                    f"https://get.infini.cloud/elasticsearch/analysis-ik/8.11.0\n"
                    f"原始错误: {e}"
                ) from e
            raise

    def delete_index(self):
        """删除整个索引（用于删除知识库）。"""
        if self.index_exists():
            self.es.indices.delete(index=self._index)

    # ================================================================
    # 数据写入
    # ================================================================

    def insert(self, doc_id: str, fields: dict):
        """写入单条文档"""
        self.es.index(index=self._index, id=doc_id, body=fields)

    def insert_batch(self, docs: list[tuple[str, dict]]):
        """批量写入 [(doc_id, fields), ...]"""
        actions = [
            {"_index": self._index, "_id": doc_id, "_source": fields}
            for doc_id, fields in docs
        ]
        helpers.bulk(self.es, actions)

    def delete(self, doc_id: str):
        self.es.delete(index=self._index, id=doc_id, ignore=[404])

    # ================================================================
    # BM25 检索
    # ================================================================

    def search(self, query: str, size: int = BM25_SIZE) -> list[dict]:
        """
        BM25 多路查询：优先「名称」命中（如「人脸识别」），弱化共性需求模块与长内容。
        """
        q = query.strip()
        terms = [t for t in re.split(r"\s+", q) if t]

        cross_fields = [
            f"{field}^{boost}" for field, boost in CROSS_FIELD_BOOST.items()
        ]
        ngram_fields = [f"{field}.ngram" for field in NGRAM_FIELDS]
        msm = cross_field_min_should_match(q)

        should_clauses: list[dict] = []

        for i, term in enumerate(terms):
            # 靠后的检索词往往是更具体意图（如「可读卡 人脸识别」→ 人脸识别）
            term_boost = NAME_TERM_BOOST + i * 10
            should_clauses.append({
                "match": {"名称": {"query": term, "boost": term_boost}},
            })
            should_clauses.append({
                "match": {
                    "交易功能": {"query": term, "boost": TXN_FUNC_TERM_BOOST + i * 3},
                },
            })
            should_clauses.append({
                "match": {
                    "需求模块": {"query": term, "boost": MODULE_TERM_BOOST + i * 2},
                },
            })
        should_clauses.append({
            "match_phrase": {"名称": {"query": q, "boost": NAME_TERM_BOOST + 5}},
        })
        should_clauses.append({
            "match_phrase": {"交易功能": {"query": q, "boost": TXN_FUNC_TERM_BOOST + 4}},
        })
        should_clauses.append({
            "match_phrase": {"需求模块": {"query": q, "boost": MODULE_TERM_BOOST + 3}},
        })
        should_clauses.append({
            "multi_match": {
                "query": q,
                "type": "cross_fields",
                "fields": cross_fields,
                "operator": "or",
                "minimum_should_match": msm,
            }
        })
        should_clauses.append({
            "match": {"需求模块": {"query": q, "boost": MODULE_MATCH_BOOST}},
        })
        should_clauses.append({
            "match": {"内容": {"query": q, "boost": CONTENT_MATCH_BOOST}},
        })

        code_match = _CP_ID_IN_QUERY_RE.search(q)
        if code_match:
            code = code_match.group(1).upper()
            should_clauses.extend([
                {"term": {"编号": {"value": code, "boost": 20}}},
                {"prefix": {"编号": {"value": code, "boost": 12}}},
                {"wildcard": {"编号": {"value": f"*{code}*", "boost": 6}}},
            ])
        elif _CP_ID_QUERY_RE.match(q):
            code = q.upper()
            should_clauses.extend([
                {"term": {"编号": {"value": code, "boost": 20}}},
                {"prefix": {"编号": {"value": code, "boost": 12}}},
            ])

        should_clauses.append({
            "multi_match": {
                "query": q,
                "type": "cross_fields",
                "fields": ngram_fields,
                "operator": "or",
                "boost": 0.3,
            }
        })

        inner_bool = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        }

        focus = pick_focus_term(q)
        if focus and len(terms) >= 2:
            es_query: dict = {
                "function_score": {
                    "query": inner_bool,
                    "functions": [
                        {
                            "filter": {"match": {"名称": {"query": focus}}},
                            "weight": NAME_FOCUS_WEIGHT,
                        },
                        {
                            "filter": {
                                "bool": {
                                    "must": [{"match": {"内容": {"query": focus}}}],
                                    "must_not": [{"match": {"名称": {"query": focus}}}],
                                }
                            },
                            "weight": CONTENT_ONLY_FOCUS_WEIGHT,
                        },
                    ],
                    "score_mode": "multiply",
                    "boost_mode": "multiply",
                }
            }
        else:
            es_query = inner_bool

        body = {
            "query": es_query,
            "highlight": {
                "fields": {
                    "编号": {},
                    "需求模块": {},
                    "交易功能": {},
                    "类型": {},
                    "名称": {},
                    "内容": {"fragment_size": 120, "number_of_fragments": 2},
                    "关联": {},
                },
                "pre_tags": ["**"],
                "post_tags": ["**"],
            },
            "size": size,
        }

        result = self.es.search(index=self._index, body=body)

        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"],
                "highlight": hit.get("highlight", {}),
            }
            for hit in result["hits"]["hits"]
        ]
