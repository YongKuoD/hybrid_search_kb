# Elasticsearch analysis-ik 插件（离线）

与 `docker.elastic.co/elasticsearch/elasticsearch:8.11.0` 配套的 IK 中文分词插件。

| 文件 | 说明 |
|------|------|
| `elasticsearch-analysis-ik-8.11.0.zip` | 启动时由 `docker compose` 挂载进容器并本地安装，**无需外网** |

## 有网时重新下载

```bash
curl -fsSL -o plugins/elasticsearch/elasticsearch-analysis-ik-8.11.0.zip \
  https://get.infini.cloud/elasticsearch/analysis-ik/8.11.0
```

## 已安装到数据卷后

插件会写入 `es_data` 卷，后续启动会跳过安装。若需重装，先删除卷再 `docker compose up`：

```bash
docker compose down -v   # 会清空 ES 数据
```
