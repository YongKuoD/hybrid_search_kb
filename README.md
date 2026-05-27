# 混合检索知识库 — 技术方案

## 1. 概述

为柜面业务知识库构建检索系统，支持用户通过**跨字段、碎片化、含缩写**的关键词查询，从结构化数据中精准召回原始记录。

**核心策略**：BM25 关键词检索（Elasticsearch）+ 向量语义检索（Milvus）→ RRF 融合排序。

## 2. 数据模型

每条知识库记录包含 4 个结构化字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| 需求模块 | 编号 + 描述 | CP0120025070996-智柜借记卡可读可销卡 |
| 交易功能 | 业务功能分类 | （可为空） |
| 流程 | 业务流程名称 | 可读可销卡转账支取流程、可读卡销卡现金支取流程 |
| 步骤 | 操作步骤序列 | 插入借记卡/磁条卡、输入密码、插入身份证、人脸识别… |

查询特点：用户输入不是字段完整内容，而是**跨字段的碎片关键词**，如"可读卡人脸识别"、"销卡流程"等。

## 3. 系统架构

```
                         ┌─────────────────┐
                         │   用户查询文本    │
                         └────────┬────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
          ┌───────▼───────┐               ┌───────▼───────┐
          │  BM25 检索     │               │  向量检索      │
          │ (Elasticsearch)│               │  (Milvus)     │
          │               │               │               │
          │ · ik_max_word │               │ · bge-large   │
          │ · ngram 兜底   │               │   -zh-v1.5    │
          │ · 字段加权     │               │ · IP 相似度    │
          │ · cross_fields│               │               │
          └───────┬───────┘               └───────┬───────┘
                  │                               │
                  │  top 30                       │  top 30
                  └───────────────┬───────────────┘
                                  │
                          ┌───────▼───────┐
                          │   RRF 融合     │
                          │               │
                          │ score = Σ     │
                          │ 1/(60+rank)   │
                          └───────┬───────┘
                                  │
                          ┌───────▼───────┐
                          │   top 10       │
                          │   返回原始记录  │
                          └───────────────┘
```

## 4. 存储设计

### 4.1 为什么要两套存储

| | Elasticsearch | Milvus |
|---|---|---|
| 存储内容 | 原始字段 + 倒排索引 | 语义向量 |
| 擅长 | 字面精确匹配、字段级权重控制、编号匹配 | 同义改写泛化（"刷脸"→"人脸识别"） |
| 不擅长 | 同义词、语义近似 | 精确编号匹配、字段级可控性 |
| 返回内容 | 原始结构化字段（展示用） | 向量相似度分数 |

两者互补，无法互相替代。同一份数据同时写入两套存储，用统一的 `doc_id` 关联。

### 4.2 Elasticsearch 索引设计

**分词策略**：

- 主分词器：`ik_max_word`（细粒度切分，确保"可读卡"能切出"可读"+"卡"）
- 兜底分词器：`ngram`（2-gram/3-gram，字符级保底匹配缩写）
- 每个字段配置：主分词 + `copy_to` 全文检索字段 + ngram 子字段

**索引 Mapping**：

```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "ik_analyzer": { "type": "custom", "tokenizer": "ik_max_word" },
        "ngram_analyzer": {
          "type": "custom",
          "tokenizer": "ngram_tokenizer",
          "filter": ["lowercase"]
        }
      },
      "tokenizer": {
        "ngram_tokenizer": { "type": "ngram", "min_gram": 2, "max_gram": 3 }
      }
    }
  },
  "mappings": {
    "properties": {
      "需求模块": {
        "type": "text", "analyzer": "ik_analyzer", "copy_to": "full_text",
        "fields": { "ngram": { "type": "text", "analyzer": "ngram_analyzer" } }
      },
      "交易功能": {
        "type": "text", "analyzer": "ik_analyzer", "copy_to": "full_text",
        "fields": { "ngram": { "type": "text", "analyzer": "ngram_analyzer" } }
      },
      "流程": {
        "type": "text", "analyzer": "ik_analyzer", "copy_to": "full_text",
        "fields": { "ngram": { "type": "text", "analyzer": "ngram_analyzer" } }
      },
      "步骤": {
        "type": "text", "analyzer": "ik_analyzer", "copy_to": "full_text",
        "fields": { "ngram": { "type": "text", "analyzer": "ngram_analyzer" } }
      },
      "full_text": { "type": "text", "analyzer": "ik_analyzer" }
    }
  }
}
```

### 4.3 Milvus Collection 设计

```
字段:
  id         VARCHAR(64)  — 主键，与 ES 共享 ID
  embedding  FLOAT_VECTOR(1024)  — bge-large-zh-v1.5 输出维度
  text       VARCHAR(4096) — 拼接后的全文（便于排查）

索引:
  metric_type: IP (inner product，等价于余弦相似度)
  index_type:  IVF_FLAT, nlist=128
```

**拼接格式**（用于 embedding）：

```
需求模块: CP0120025070996-智柜借记卡可读可销卡
交易功能: 
流程: 可读可销卡转账支取流程、可读卡销卡现金支取流程
步骤: 插入借记卡/磁条卡、输入密码、插入身份证、人脸识别…
```

带字段标签的拼接让模型在编码时能感知字段语义，比纯拼接更准确。

## 5. 检索策略

### 5.1 BM25 查询（Elasticsearch）

单次查询内三路并行评估，bool 组合：

| 路数 | 类型 | 说明 | 权重 |
|------|------|------|------|
| 第 1 路 | `cross_fields` | 在所有字段上做跨字段匹配，字段按权重 boost（步骤×5 > 流程×3 > 需求模块×2 > 交易功能×1.5），`minimum_should_match` 动态调整（短查询 100%，长查询 60%） | 主体 |
| 第 2 路 | `match_phrase` | 在步骤字段上精确短语匹配，完整命中如"人脸识别"大幅加分 | ×10 |
| 第 3 路 | `match_phrase` | 在流程字段上精确短语匹配 | ×5 |
| 第 4 路 | `cross_fields` + ngram | 在 ngram 子字段上字符级兜底，捕获 ik 分词切不开的缩写 | ×0.3 |

`minimum_should_match` 动态策略：避免"卡"单字命中返回大量噪音。

```
查询词 ≤ 2 个 → 100%（全命中）
查询词 3-4 个 → 75%
查询词 ≥ 5 个 → 60%
```

### 5.2 向量检索（Milvus）

- 查询文本过 bge-large-zh-v1.5 得到 1024 维向量
- 在 Milvus 中做 ANN 检索（nprobe=16）
- 使用 IP（内积）度量，配合归一化向量等价于余弦相似度

### 5.3 RRF 融合

不对原始分数做归一化，直接按排名融合：

```
RRF(doc) = 1/(60 + BM25_rank) + 1/(60 + Vector_rank)
```

- k=60 是经验最优的平滑常数
- 不依赖两路分数的量纲和分布
- 只有同时出现在两路 Top 30 中的文档才会获得高融合分

**效果**：BM25 精确匹配排前列 + 向量语义补充 = 互补式融合。

## 6. 数据流

### 6.1 摄入流程

```
 JSON/CSV 文件
      │
      ▼
 读取记录列表
      │
      ├──────────────────┐
      ▼                  ▼
 拼接 embedding 文本    提取结构化字段
      │                  │
      ▼                  ▼
 bge-large-zh-v1.5     ES bulk index
      │
      ▼
 Milvus insert
      │
      ▼
 flush + load
```

**关键保证**：两路使用相同的 `doc_id`，RRF 融合后通过 ID 从 ES 回填原始字段。

### 6.2 检索流程

```
 用户输入 "可读卡人脸识别"
      │
      ├──────────────┐
      ▼              ▼
 ES.search()     mv.search()
 top 30          top 30
      │              │
      └──────┬───────┘
             ▼
      RRF 融合排序
             │
             ▼
   取 Top 10，通过 ID 查 ES 原始字段
             │
             ▼
      返回带高亮的结果
```

## 7. 关键技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 关键词引擎 | Elasticsearch 8.x | BM25 原生支持、ik 中文分词成熟、cross_fields 跨字段查询 |
| 分词器 | ik_max_word | 细粒度切分适配中文碎片化查询（"可读卡"→"可读"+"卡"） |
| 向量数据库 | Milvus 2.3 | 国产开源、社区活跃、IVF_FLAT 在中规模数据集上检索效率高 |
| 向量模型 | bge-large-zh-v1.5 | 中文 SOTA、1024 维、MTEB 中文榜排名前 3 |
| 融合算法 | RRF | 无需归一化、不依赖分数分布、工程实现简单可靠 |

## 8. 检索示例

**查询："可读卡人脸识别"**

| 排名 | 需求模块 | 命中字段 | 为什么 |
|------|----------|----------|--------|
| #1 | CP0120025070996-智柜借记卡可读可销卡 | 步骤:"人脸识别"、需求模块:"可读" | 步骤 match_phrase 高权重命中 + cross_fields 命中 3/4 字段 |
| #2 | CP0120042093005-智柜借记卡开户 | 步骤:"人脸识别" | 步骤命中但"可读卡"缺失，cross_fields 只命中 1 词 |
| #3 | CP0120031082001-智柜信用卡可读卡 | 需求模块:"可读卡"、步骤:"人脸识别" | 需求模块权重偏低，且缺少"销卡" |

**查询："销卡流程"**

命中文档 #1 和 #4（流程或需求模块含"销卡"），#1 因步骤字段丰富度更高排在前。

## 9. 部署与运行

### 9.1 基础设施

```bash
docker compose up -d  # 启动 ES + Milvus + etcd + MinIO
```

### 9.2 Python 环境

```bash
pip install -r requirements.txt
```

依赖：elasticsearch、pymilvus、sentence-transformers、numpy。

### 9.3 使用步骤

```bash
# 初始化
python main.py build           # 创建 ES 索引 + Milvus Collection

# 导入数据
python main.py ingest data.json
python main.py demo            # 或使用内置示例数据

# 检索
python main.py search "可读卡人脸识别"
python main.py search "销卡 签字"
python main.py interactive     # 交互模式
```

### 9.4 代码集成

```python
from hybrid_search import HybridSearcher

searcher = HybridSearcher()
results = searcher.search("可读卡人脸识别", top_k=10)

for r in results:
    print(f"RRF: {r['rrf_score']} | BM25: {r['bm25_score']} | Vec: {r.get('vector_score')}")
    print(f"  步骤: {r['source']['步骤']}")
```

## 10. 调优参数

所有可调参数集中在 `config.py`：

```python
FIELD_BOOST = {            # 字段权重
    "需求模块": 2.0,
    "交易功能": 1.5,
    "流程": 3.0,
    "步骤": 5.0,          # 操作步骤权重最高
}

BM25_SIZE = 30             # BM25 路召回数
VECTOR_SIZE = 30           # 向量路召回数
RRF_K = 60                 # RRF 平滑常数
FINAL_TOP_K = 10           # 最终返回数

match_phrase_boost = 10    # 步骤精确短语命中权重
ngram_boost = 0.3          # ngram 兜底权重
```

调参建议：先用现有参数跑，观察哪些类型查询召回不佳，再针对性调整对应的权重或召回数。

## 11. 项目文件

```
hybrid_search_kb/
├── config.py           # 配置中心（连接、字段权重、检索参数）
├── embedding.py        # sentence-transformers 封装
├── es_store.py         # ES 索引管理 + BM25 检索
├── milvus_store.py     # Milvus Collection + 向量检索
├── ingestion.py        # 数据摄入（同时写入 ES + Milvus）
├── hybrid_search.py    # RRF 融合引擎
├── main.py             # CLI 入口（build/ingest/demo/search/interactive）
├── docker-compose.yml  # 基础设施一键部署
├── data_sample.json    # 4 条示例数据
└── requirements.txt    # Python 依赖
```
