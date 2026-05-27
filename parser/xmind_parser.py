"""XMind 结构化解析。

识别思维导图中的三层层次：
  「编号-需求模块」→「交易/功能」→「内容类型」

内容类型包含：业务流程、业务规则、页面控制、数据验证、部署 等。
兼容「交易/功能」缺失场景（内容类型直接挂在「编号-需求模块」下）。

解析结果以 ``XMindSection`` 表示，每个切片对应一个「交易/功能」维度，
包含 编号、需求模块、交易/功能、按类型组织的内容 等结构化字段。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── 常量 ────────────────────────────────────────

KNOWN_CONTENT_TYPES = frozenset({"业务流程", "业务规则", "页面控制", "数据验证", "部署"})
"""XMind 中已知的内容类型节点标题（与导图常用栏目一致，空节点也保留占位）。"""

# 编号-描述 同一标题：支持半角/全角连字符及两侧空格
KB_MODULE_RE = re.compile(
    r"^([A-Za-z]{2}\d{8,})\s*[-—―－]\s*(.+)$"
)
# 仅编号标题（描述在下一级子节点）
KB_ID_ONLY_RE = re.compile(r"^([A-Za-z]{2}\d{8,})$")

# 子节点标题前缀序号：``1 开户流程`` → 序号 ``1`` + 名称 ``开户流程``
# 分隔符仅限 [.、．,，] 及空格，避免把名称里的 ``-``（如 产品管理-合同文本）误判为分隔符
_NODE_NUM_PREFIX_RE = re.compile(r"^(\d+)(?:\s+|[\.\、．,，]\s*)(.+)$")
_DIGITS_ONLY_RE = re.compile(r"^\d+$")

MARKER_LABEL_MAP: dict[str, str] = {
    "symbol-question": "预期结果",
    "symbol-right": "正例",
    "symbol-wrong": "反例",
}


# ─── 数据结构 ────────────────────────────────────


@dataclass
class FlowNode:
    """单个业务流程节点。"""

    name: str = ""
    """节点名称，如 ``1 产品管理-合同文本配置-产品配置页面增加配置项``。"""

    content: str = ""
    """节点详细内容（子节点文本）。"""


@dataclass
class RuleNode:
    """单个业务规则节点。"""

    name: str = ""
    """规则名称（与对应流程节点名称一致）。"""

    content: str = ""
    """规则详细内容（子节点文本）。"""

    linked_flow: str = ""
    """关联的流程节点名称。"""


@dataclass
class XMindSection:
    """单个「交易/功能」维度下的 XMind 解析结果。"""

    kb_id: str = ""
    """编号，如 ``CP0000000000000``。"""

    kb_module: str = ""
    """需求模块，如 ``新增丽水地区承诺书需求``。"""

    transaction: str = ""
    """交易/功能，可为空字符串（当内容类型直接挂在需求模块下时）。"""

    content_by_type: dict[str, str] = field(default_factory=dict)
    """
    按内容类型组织的文本。
    键为类型名称（业务流程／业务规则／页面控制／数据验证／部署／自定义），
    值为该类型下所有子节点的 Markdown 文本。
    """

    flow_nodes: list[FlowNode] = field(default_factory=list)
    """拆分后的业务流程节点列表，每个数字编号节点一条。"""

    rule_nodes: list[RuleNode] = field(default_factory=list)
    """拆分后的业务规则节点列表，每个规则一条，含关联流程。"""

    # ── 向后兼容（旧接口使用）─────────────────────────

    @property
    def processes(self) -> str:
        """扁平流程名列表（向后兼容）。"""
        return "、".join(n.name for n in self.flow_nodes)

    @property
    def steps(self) -> str:
        """扁平步骤列表（向后兼容）。"""
        return "、".join(n.name for n in self.rule_nodes)

    # ── 主要接口 ─────────────────────────────────────

    @property
    def full_path(self) -> str:
        """全路径字符串：「需求模块 > 交易/功能」或仅「需求模块」。"""
        if self.transaction:
            return f"{self.kb_module} > {self.transaction}"
        return self.kb_module

    def to_meta(self) -> dict[str, Any]:
        """转换为向量库 payload 的 meta 子字典。"""
        return {
            "编号": self.kb_id,
            "需求模块": self.kb_module,
            "交易/功能": self.transaction,
            "全路径": self.full_path,
        }

    def to_record(self) -> dict[str, str]:
        """向后兼容：旧版四字段扁平记录。"""
        return {
            "编号": self.kb_id,
            "需求模块": self.kb_module,
            "交易功能": self.transaction,
            "流程": self.processes,
            "步骤": self.steps,
        }

    def to_records(self) -> list[dict[str, str]]:
        """拆分为知识库记录列表 — 每个流程节点/规则节点各一条。

        每条记录包含：
        - 编号、需求模块、交易功能（共用）
        - 类型: "业务流程" 或 "业务规则"
        - 名称: 节点自身标题
        - 内容: 节点详细内容
        - 关联: 规则→流程的匹配关系（仅业务规则有值）

        :return: 扁平记录列表
        """
        records: list[dict[str, str]] = []

        # 收集流程节点名称集合（用于规则关联匹配）
        flow_names = {n.name for n in self.flow_nodes}

        for node in self.flow_nodes:
            # 找出关联此流程的规则名列表
            linked_rules = [r.name for r in self.rule_nodes if r.linked_flow == node.name]
            records.append({
                "编号": self.kb_id,
                "需求模块": self.kb_module,
                "交易功能": self.transaction,
                "类型": "业务流程",
                "名称": node.name,
                "内容": node.content,
                "关联": "、".join(linked_rules) if linked_rules else "",
            })

        for node in self.rule_nodes:
            records.append({
                "编号": self.kb_id,
                "需求模块": self.kb_module,
                "交易功能": self.transaction,
                "类型": "业务规则",
                "名称": node.name,
                "内容": node.content,
                "关联": node.linked_flow,
            })

        return records

    def to_markdown(self) -> str:
        """将结构化数据渲染为 Markdown 文本（用于向量化检索）。"""
        parts: list[str] = [
            f"# {self.kb_id} {self.kb_module}",
        ]
        if self.transaction:
            parts.append(f"## {self.transaction}")
        for ctype, content in self.content_by_type.items():
            content = content.strip()
            if content:
                parts.append(f"\n### {ctype}\n{content}")
        return "\n".join(parts)

    def summary(self) -> str:
        """简短描述，用于日志 / 调试。"""
        tx = self.transaction or "（无）"
        types = "、".join(self.content_by_type.keys()) or "（无）"
        chars = sum(len(v) for v in self.content_by_type.values())
        n_flows = len(self.flow_nodes)
        n_rules = len(self.rule_nodes)
        return (
            f"[{self.kb_id}] {self.kb_module} / {tx} "
            f"({len(self.content_by_type)} 类, {chars} 字符, "
            f"{n_flows} 流程节点, {n_rules} 规则节点)"
        )


# ─── 标记映射 ────────────────────────────────────


def _get_marker_label(makers: list[Any]) -> str | None:
    """从 markers 列表中提取中文标签（兼容旧版公用接口）。"""
    if not makers:
        return None
    for m in makers:
        token = ""
        if isinstance(m, str):
            token = m
        elif isinstance(m, dict):
            token = m.get("makerId") or m.get("type") or m.get("name") or ""
        if token in MARKER_LABEL_MAP:
            return MARKER_LABEL_MAP[token]
    return None


# ─── 结构化解析 ──────────────────────────────────


def parse_xmind(path: str | Path) -> list[XMindSection]:
    """解析 XMind 文件，返回结构化切片列表。

    :param path: .xmind 或 .xmind → .json 导出的文件路径
    :return: 按「交易/功能」维度切分的结构化结果列表
    """
    from xmindparser import xmind_to_dict

    path = Path(path)
    logger.info("解析 XMind: %s", path.name)
    sheets = xmind_to_dict(str(path))
    logger.info("XMind sheets 数: %s", len(sheets))
    if not sheets:
        logger.warning("XMind 文件无 sheet: %s", path.name)
        return []

    sections: list[XMindSection] = []
    for i, sheet in enumerate(sheets):
        if not isinstance(sheet, dict):
            logger.debug("Sheet %s 非 dict, 跳过", i)
            continue
        root = sheet.get("topic") or {}
        if not isinstance(root, dict):
            logger.debug("Sheet %s 无 topic, 跳过", i)
            continue
        root_title = (root.get("title") or "") or ""
        root_children = root.get("topics", []) or []
        if not isinstance(root_children, list):
            root_children = []
        logger.info(
            "Sheet %s 根标题=%r 子节点数=%s",
            i, root_title[:60], len(root_children),
        )
        # 列出每个子节点的标题（帮助诊断不匹配原因）
        for ci, c in enumerate(root_children):
            if isinstance(c, dict):
                ct = (c.get("title") or "") or ""
                matched = _identify_kb_module_from_topic(c)
                logger.info(
                    "  子节点 %s: %r → %s",
                    ci, ct[:80],
                    f"编号={matched[0]}, 模块={matched[1]}" if matched else "不匹配（非编号-模块格式）",
                )
        _parse_root_children(root, sections)
    logger.info("XMind 解析完成: %s 个切片", len(sections))
    return sections


# ─── 知识库字段转换 ────────────────────────────────


def xmind_to_records(path: str | Path) -> list[dict[str, str]]:
    """解析 XMind 文件并转换为知识库扁平记录列表。

    每个流程节点（数字编号开头的节点）和每个规则节点各拆为一条独立记录。
    每条记录包含：「编号」「需求模块」「交易功能」「类型」「名称」「内容」「关联」。

    :param path: .xmind 文件路径
    :return: 扁平记录列表
    """
    sections = parse_xmind(path)
    records: list[dict[str, str]] = []
    for s in sections:
        records.extend(s.to_records())
    logger.info("XMind → 知识库记录: %s 条（%s 个 Section）", len(records), len(sections))

    # 按类型统计
    flow_count = sum(1 for r in records if r.get("类型") == "业务流程")
    rule_count = sum(1 for r in records if r.get("类型") == "业务规则")
    logger.info("  业务流程: %s 条, 业务规则: %s 条", flow_count, rule_count)

    for i, r in enumerate(records):
        logger.debug(
            "  记录 %s: [%s] %s | 交易=%s | 关联=%s",
            i + 1,
            r.get("类型", "?"),
            r["名称"][:60],
            r["交易功能"] or "（无）",
            r.get("关联", "")[:40] or "（无）",
        )
    return records


def _parse_root_children(
    topic: dict[str, Any],
    out: list[XMindSection],
) -> None:
    """扫描根 topic 及其子节点，识别并解析「编号-需求模块」节点。

    支持两种结构：
    1. 根节点本身即「编号-需求模块」（子节点直接为内容类型/交易）
    2. 根节点为容器，其子节点为「编号-需求模块」
    """
    # ── 先检查根节点自身是否就是模块节点 ──
    root_title = (topic.get("title") or "") or ""
    root_match = _identify_kb_module_from_topic(topic)
    if root_match:
        _parse_module_node(topic, root_match, out)
        # 根节点已是模块节点，也继续扫描子节点中可能存在的其他模块
        # （部分 XMind 文件存在同级多个模块的情况）

    # ── 扫描子节点 ──
    children = topic.get("topics", []) or []
    if not isinstance(children, list):
        return
    for child in children:
        if not isinstance(child, dict):
            continue
        title = (child.get("title") or "") or ""
        if not title:
            continue
        kb_match = _identify_kb_module_from_topic(child)
        if not kb_match:
            continue
        _parse_module_node(child, kb_match, out)


def _identify_kb_module(title: str) -> tuple[str, str] | None:
    """从标题中提取 编号 和 需求模块（同一节点内「编号-描述」格式）。"""
    title = (title or "").strip()
    if not title:
        return None
    m = KB_MODULE_RE.match(title)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _identify_kb_module_from_topic(topic: dict[str, Any]) -> tuple[str, str] | None:
    """从 topic 识别「编号-需求模块」。

    支持：
    1. 标题 ``CP0000000000000-新增丽水地区承诺书需求``（含全角连字符、空格）
    2. 标题仅 ``CP0000000000000``，描述在第一个非内容类型子节点
    """
    title = (topic.get("title") or "").strip()
    matched = _identify_kb_module(title)
    if matched:
        return matched

    m_id = KB_ID_ONLY_RE.match(title)
    if not m_id:
        return None

    kb_id = m_id.group(1).strip()
    children = topic.get("topics", []) or []
    if not isinstance(children, list):
        return kb_id, ""

    for child in children:
        if not isinstance(child, dict):
            continue
        child_title = (child.get("title") or "").strip()
        if not child_title or _is_content_type(child_title):
            continue
        return kb_id, child_title

    return kb_id, ""


def _is_content_type(title: str) -> bool:
    """判断标题是否为已知内容类型。"""
    return title in KNOWN_CONTENT_TYPES


def _parse_module_node(
    topic: dict[str, Any],
    kb_match: tuple[str, str],
    out: list[XMindSection],
) -> None:
    """解析单个「编号-需求模块」节点下的所有「交易/功能」切片。

    兼容两种子结构：
    1. 【标准】子节点为「交易/功能」→ 其下为内容类型
    2. 【省略】子节点直接为内容类型 → 交易/功能 记为 ""

    对「业务流程」和「业务规则」下的每个数字编号节点拆分，
    并通过名称匹配建立流程↔规则的关联关系。
    """
    kb_id, kb_module = kb_match
    children = topic.get("topics", []) or []
    if not isinstance(children, list):
        return

    # 分离直接内容类型节点 与 交易/功能节点
    direct_ct: list[dict[str, Any]] = []
    tx_nodes: list[dict[str, Any]] = []

    for child in children:
        if not isinstance(child, dict):
            continue
        title = (child.get("title") or "") or ""
        if not title:
            continue
        if _is_content_type(title):
            direct_ct.append(child)
        else:
            tx_nodes.append(child)

    # ── 场景 1：直接内容类型 → 交易/功能="" ──
    if direct_ct:
        section = XMindSection(kb_id=kb_id, kb_module=kb_module, transaction="")
        for node in direct_ct:
            ct_title = (node.get("title") or "") or ""
            content = _collect_node_content(node)
            if content.strip() or ct_title:
                section.content_by_type[ct_title] = content.strip()
            # 收集拆分后的流程/规则节点
            if ct_title == "业务流程":
                raw = _collect_numbered_nodes(node, strip_rule_index=False)
                section.flow_nodes = [FlowNode(name=n["name"], content=n["content"]) for n in raw]
            elif ct_title == "业务规则":
                raw = _collect_numbered_nodes(node, prefix_title=ct_title, strip_rule_index=True)
                section.rule_nodes = [
                    RuleNode(name=n["name"], content=n["content"])
                    for n in raw
                ]
        if section.content_by_type or section.flow_nodes or section.rule_nodes:
            _link_rules_to_flows(section)
            out.append(section)

    # ── 场景 2：标准交易/功能节点 ──
    for node in tx_nodes:
        tx_title = (node.get("title") or "") or ""
        grandchildren = node.get("topics", []) or []
        if not isinstance(grandchildren, list):
            grandchildren = []

        section = XMindSection(
            kb_id=kb_id,
            kb_module=kb_module,
            transaction=tx_title,
        )

        any_content = False
        for gc in grandchildren:
            if not isinstance(gc, dict):
                continue
            gc_title = (gc.get("title") or "") or ""
            if not gc_title:
                continue
            content = _collect_node_content(gc)
            if content.strip():
                section.content_by_type[gc_title] = content.strip()
                any_content = True
            elif _is_content_type(gc_title):
                section.content_by_type[gc_title] = ""
                any_content = True

            # 收集拆分后的流程/规则节点
            if gc_title == "业务流程":
                raw = _collect_numbered_nodes(gc, strip_rule_index=False)
                section.flow_nodes = [FlowNode(name=n["name"], content=n["content"]) for n in raw]
                any_content = any_content or bool(raw)
            elif gc_title == "业务规则":
                raw = _collect_numbered_nodes(gc, prefix_title=gc_title, strip_rule_index=True)
                section.rule_nodes = [
                    RuleNode(name=n["name"], content=n["content"])
                    for n in raw
                ]
                any_content = any_content or bool(raw)

        if any_content:
            _link_rules_to_flows(section)
            out.append(section)


def _link_rules_to_flows(section: XMindSection) -> None:
    """建立规则→流程的关联关系。

    匹配策略：
    1. 精确匹配：规则名 == 流程节点名
    2. 前缀匹配：流程节点名以规则名开头（或反之）
    3. 模糊匹配：规则名包含流程节点名的关键部分
    """
    flow_names = {n.name for n in section.flow_nodes}
    if not flow_names:
        return

    for rule in section.rule_nodes:
        # 策略 1: 精确匹配
        if rule.name in flow_names:
            rule.linked_flow = rule.name
            continue

        # 策略 2: 前缀/包含匹配
        matched = None
        for fname in flow_names:
            if rule.name.startswith(fname) or fname.startswith(rule.name):
                matched = fname
                break
            # 提取纯中文部分做匹配（去掉编号前缀）
            rule_clean = _extract_chinese_core(rule.name)
            flow_clean = _extract_chinese_core(fname)
            if rule_clean and flow_clean and len(rule_clean) > 4:
                if rule_clean in flow_clean or flow_clean in rule_clean:
                    matched = fname
                    break

        if matched:
            rule.linked_flow = matched
        # 策略 3: 如果能推断出关联，记录日志
        # （未匹配的不设 linked_flow，保持空字符串）


def _extract_chinese_core(text: str) -> str:
    """提取文本中的中文核心部分（去数字、标点、英文前缀）。"""
    import re
    # 去掉开头数字+空格/标点
    text = re.sub(r'^[\d\s\.\_\-\—]+', '', text)
    # 只保留中文、字母、数字
    chinese = re.sub(r'[^一-鿿\w]', '', text)
    return chinese


# ─── 内容提取 ────────────────────────────────────


def _collect_flat_titles(topic: dict[str, Any]) -> str:
    """提取 topic 直接子节点的标题，用顿号拼接（向后兼容，旧接口使用）。

    :param topic: 内容类型节点（如「业务流程」「业务规则」）
    :return: 顿号分隔的标题字符串
    """
    children = topic.get("topics", []) or []
    if not isinstance(children, list) or not children:
        return ""
    titles: list[str] = []
    for child in children:
        if isinstance(child, dict):
            title = (child.get("title") or "") or ""
            if title:
                titles.append(title)
    return "、".join(titles)


def _strip_rule_index_prefix(title: str) -> str:
    """业务规则名称：去掉前导 ``数字+空格/标点``，保留名称主体。

    ``1 产品管理-合同文本`` → ``产品管理-合同文本``
    """
    title = (title or "").strip()
    m = _NODE_NUM_PREFIX_RE.match(title)
    if m:
        return m.group(2).strip()
    return title


def _resolve_rule_node_name_and_content_root(
    title: str,
    child: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """业务规则节点名（无序号）及内容采集起点。"""
    title = (title or "").strip()
    if not title:
        return "", child

    if _DIGITS_ONLY_RE.match(title):
        subs = child.get("topics", []) or []
        if isinstance(subs, list):
            for sub in subs:
                if not isinstance(sub, dict):
                    continue
                sub_title = (sub.get("title") or "").strip()
                if sub_title:
                    return _strip_rule_index_prefix(sub_title), sub
        return "", child

    return _strip_rule_index_prefix(title), child


def _collect_numbered_nodes(
    topic: dict[str, Any],
    prefix_title: str = "",
    *,
    strip_rule_index: bool = False,
) -> list[dict[str, str]]:
    """收集 topic 下流程/规则子节点，每条 {name, content}。

    :param strip_rule_index: True=业务规则（去掉名称前序号）；False=业务流程（保留完整标题）
    """
    children = topic.get("topics", []) or []
    if not isinstance(children, list) or not children:
        return []

    result: list[dict[str, str]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        title = (child.get("title") or "") or ""
        if not title:
            continue

        if strip_rule_index:
            name, content_root = _resolve_rule_node_name_and_content_root(title, child)
            if not name:
                continue
            child_content = _collect_node_content(content_root)
        else:
            # 业务流程：沿用原逻辑，标题原样作为名称
            name = title.strip()
            child_content = _collect_node_content(child)

        if prefix_title:
            child_content = f"{prefix_title}: {child_content}" if child_content else prefix_title

        result.append({
            "name": name,
            "content": child_content.strip(),
        })
    return result


def _collect_node_content(topic: dict[str, Any]) -> str:
    """递归收集 topic 子节点的 Markdown 文本（不含 topic 本身标题）。

    - topic 没有 ``topics`` 键 → 视为叶子节点，返回自身标题
    - topic 有 ``topics`` 键但为空列表 → 无子内容，返回空字符串
    - topic 有子节点 → 递归收集子节点 Markdown
    """
    has_topics_key = "topics" in topic
    children = topic.get("topics", []) or []
    if not has_topics_key:
        # 真正的叶子节点：没有 topics 键，自身标题就是内容
        return topic.get("title", "") or ""
    if not isinstance(children, list) or not children:
        return ""

    lines: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        title = (child.get("title") or "") or ""
        if not title:
            continue

        grand_children = child.get("topics", []) or []
        if not isinstance(grand_children, list):
            grand_children = []

        if not grand_children:
            # 叶子子节点，直接添加
            lines.append(f"- {title}")
        else:
            # 有下层结构：标题作小标题，缩进收集子孙
            lines.append(title)
            _append_indented(grand_children, 1, lines)

    return "\n".join(lines)


def _append_indented(
    children: list[dict[str, Any]],
    depth: int,
    lines: list[str],
) -> None:
    """递归收集 children 并以指定缩进层级追加到 lines。"""
    indent = "  " * depth
    for child in children:
        if not isinstance(child, dict):
            continue
        title = (child.get("title") or "") or ""
        if not title:
            continue
        lines.append(f"{indent}- {title}")
        grandchildren = child.get("topics", []) or []
        if isinstance(grandchildren, list) and grandchildren:
            _append_indented(grandchildren, depth + 1, lines)


# ─── 向后兼容 ────────────────────────────────────


def xmind_to_markdown(path: str | Path) -> str:
    """旧版接口：XMind → 平面 Markdown（保留供外部脚本使用）。

    新代码应使用 :func:`parse_xmind` 获取结构化结果。
    """
    lines: list[str] = []
    from xmindparser import xmind_to_dict

    path = Path(path)
    sheets = xmind_to_dict(str(path))
    if not sheets:
        return ""

    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue
        root = sheet.get("topic") or {}
        if not isinstance(root, dict):
            continue
        lines.extend(_topic_to_markdown(root))

    return "\n\n---\n\n".join(lines)


def _topic_to_markdown(topic: dict[str, Any], depth: int = 0) -> list[str]:
    """递归将 topic 转为 Markdown 行（旧版平面转换）。"""
    result: list[str] = []
    title = topic.get("title", "") or ""

    markers = topic.get("makers") or topic.get("markers") or []
    if not isinstance(markers, list):
        markers = []
    label = _get_marker_label(markers)

    if label:
        display = f"{label}：{title}"
    else:
        display = title

    if depth == 0:
        result.append(f"# {display}")
    elif depth == 1:
        result.append(f"## {display}")
    elif depth == 2:
        result.append(f"### {display}")
    else:
        prefix = "  " * (depth - 1) + "- "
        result.append(f"{prefix}{display}")

    children = topic.get("topics", []) or []
    if not isinstance(children, list):
        children = []

    for child in children:
        if isinstance(child, dict) and child.get("title"):
            result.extend(_topic_to_markdown(child, depth + 1))

    return result


def _sheet_to_markdown(sheet: dict[str, Any]) -> str:
    """单张画布转 Markdown（旧版内部）。"""
    root = sheet.get("topic") or {}
    if not isinstance(root, dict):
        return ""
    return "\n".join(_topic_to_markdown(root))



# ─── 简易 CLI 调试 ───────────────────────────────


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        sections = parse_xmind(p)
        print(f"\n{'=' * 60}")
        print(f"文件: {p}")
        print(f"切片数: {len(sections)}")
        print("=" * 60)
        for i, sec in enumerate(sections, 1):
            print(f"\n--- 切片 {i} ---")
            print(sec.summary())
            print()
            print(sec.to_markdown())
            print()
