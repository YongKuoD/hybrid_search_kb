"""业务字段归一化（编号 / 需求模块拆分）。"""

import re

_LEGACY_MODULE_RE = re.compile(
    r"^([A-Za-z]{2}\d{8,})\s*[-—―－]\s*(.+)$"
)
_ID_ONLY_RE = re.compile(r"^([A-Za-z]{2}\d{8,})$")


def normalize_record_fields(rec: dict) -> dict:
    """确保记录含独立的「编号」「需求模块」字段。

    兼容旧数据：需求模块存为 ``CP0000000000000-新增丽水地区承诺书需求``。
    """
    out = dict(rec)
    code = (out.get("编号") or "").strip()
    module = (out.get("需求模块") or "").strip()

    if code:
        out["编号"] = code
        out["需求模块"] = module
        return out

    if not module:
        out.setdefault("编号", "")
        out.setdefault("需求模块", "")
        return out

    m = _LEGACY_MODULE_RE.match(module)
    if m:
        out["编号"] = m.group(1).strip()
        out["需求模块"] = m.group(2).strip()
        return out

    m_id = _ID_ONLY_RE.match(module)
    if m_id:
        out["编号"] = m_id.group(1).strip()
        out["需求模块"] = ""
        return out

    out["编号"] = ""
    out["需求模块"] = module
    return out
