"""
混合检索知识库 — CLI 入口

用法:
  python cli.py build              # 初始化 ES 索引 + Milvus collection
  python cli.py ingest <文件>      # 导入 JSON/CSV/XMind 数据
  python cli.py demo               # 写入示例数据
  python cli.py search <查询文本>  # 混合检索
"""

import argparse
import sys
from pathlib import Path


def cmd_build(args):
    """创建 ES 索引 + Milvus collection"""
    from engine.ingestion import Ingester

    ingester = Ingester()
    ingester.ensure_stores(force=args.force)
    print("ES 索引和 Milvus Collection 已就绪。")


def cmd_ingest(args):
    """导入数据文件"""
    from engine.ingestion import Ingester

    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    ingester = Ingester()
    ingester.ensure_stores()

    suffix = path.suffix.lower()
    if suffix == ".json":
        ingester.ingest_json(str(path))
    elif suffix == ".csv":
        ingester.ingest_csv(str(path))
    elif suffix == ".xmind":
        ingester.ingest_xmind(str(path))
    else:
        print(f"不支持的文件格式: {suffix}, 请使用 .json、.csv 或 .xmind")
        sys.exit(1)


def cmd_demo(args):
    """写入示例数据 (你问题中的那条)"""
    from engine.ingestion import Ingester, ingest_demo_data

    ingester = Ingester()
    ingester.ensure_stores()
    ingest_demo_data(ingester)


def cmd_search(args):
    """混合检索"""
    from engine.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    print(searcher.search_pretty(args.query))


def cmd_interactive(args):
    """交互式搜索模式"""
    from engine.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    print("混合检索知识库 — 交互模式")
    print("输入查询文本, 输入 q 退出\n")

    while True:
        try:
            query = input("查询 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            break

        print(searcher.search_pretty(query))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="混合检索知识库 — ES(BM25) + Milvus(向量) + RRF 融合"
    )
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="创建 ES 索引 + Milvus collection")
    p_build.add_argument("--force", action="store_true", help="强制重建已有索引")
    p_build.set_defaults(func=cmd_build)

    # ingest
    p_ingest = sub.add_parser("ingest", help="导入数据文件")
    p_ingest.add_argument("file", help="JSON 或 CSV 文件路径")
    p_ingest.set_defaults(func=cmd_ingest)

    # demo
    p_demo = sub.add_parser("demo", help="写入示例数据")
    p_demo.set_defaults(func=cmd_demo)

    # search
    p_search = sub.add_parser("search", help="混合检索")
    p_search.add_argument("query", help="查询文本")
    p_search.set_defaults(func=cmd_search)

    # interactive
    p_int = sub.add_parser("interactive", help="交互式搜索")
    p_int.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
