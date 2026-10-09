"""ASS 对白 / 顶注样式。SRT/VTT 带不走字体颜色，只有 ASS 会按这里落盘。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .models import LANG_FONT_SIZES

SAFE_FONTS = (
    "Arial",
    "Microsoft YaHei",
    "PingFang SC",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
    "Helvetica",
)

DEFAULT_ASS_STYLE: Dict[str, Any] = {
    "font_name": "Arial",
    "primary_color": "#FFFFFF",
    "outline_color": "#000000",
    "back_color": "#000000",
    "outline": 2,
    "shadow": 2,
    "bold": False,
    "italic": False,
    "sizes": [22, 17, 14],
    "note_size": 16,
    "note_color": "#B4E0FF",
    "margin_v": 20,
    "line_gap": 28,
}

HEX = re.compile(r"^#?[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$")


def hex_to_ass(value: str, *, alpha: int = 0) -> str:
    """CSS #RRGGBB → ASS &HAABBGGRR。alpha 0 不透明。"""
    text = str(value or "").strip()
    if not HEX.match(text):
        text = "#FFFFFF"
    raw = text.lstrip("#")
    if len(raw) == 8:
        rr, gg, bb, aa = raw[0:2], raw[2:4], raw[4:6], raw[6:8]
        return f"&H{aa}{bb}{gg}{rr}".upper()
    rr, gg, bb = raw[0:2], raw[2:4], raw[4:6]
    return f"&H{int(alpha):02X}{bb}{gg}{rr}".upper()


def safe_font(name: str) -> str:
    cleaned = re.sub(r"[,\\]", "", str(name or "").strip())
    return cleaned or "Arial"


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _hex(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if HEX.match(text):
        return text if text.startswith("#") else f"#{text}"
    return default


def normalize_ass_style(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    sizes_in = raw.get("sizes") if isinstance(raw.get("sizes"), (list, tuple)) else DEFAULT_ASS_STYLE["sizes"]
    sizes = []
    defaults = list(DEFAULT_ASS_STYLE["sizes"])
    for index in range(3):
        fallback = defaults[index] if index < len(defaults) else LANG_FONT_SIZES[-1]
        incoming = sizes_in[index] if index < len(sizes_in) else fallback
        sizes.append(_clamp_int(incoming, fallback, 8, 72))
    return {
        "font_name": safe_font(raw.get("font_name") or DEFAULT_ASS_STYLE["font_name"]),
        "primary_color": _hex(raw.get("primary_color"), DEFAULT_ASS_STYLE["primary_color"]),
        "outline_color": _hex(raw.get("outline_color"), DEFAULT_ASS_STYLE["outline_color"]),
        "back_color": _hex(raw.get("back_color"), DEFAULT_ASS_STYLE["back_color"]),
        "outline": _clamp_int(raw.get("outline"), DEFAULT_ASS_STYLE["outline"], 0, 8),
        "shadow": _clamp_int(raw.get("shadow"), DEFAULT_ASS_STYLE["shadow"], 0, 8),
        "bold": bool(raw.get("bold")),
        "italic": bool(raw.get("italic")),
        "sizes": sizes,
        "note_size": _clamp_int(raw.get("note_size"), DEFAULT_ASS_STYLE["note_size"], 8, 48),
        "note_color": _hex(raw.get("note_color"), DEFAULT_ASS_STYLE["note_color"]),
        "margin_v": _clamp_int(raw.get("margin_v"), DEFAULT_ASS_STYLE["margin_v"], 0, 160),
        "line_gap": _clamp_int(raw.get("line_gap"), DEFAULT_ASS_STYLE["line_gap"], 8, 80),
    }


def style_sizes(style: Dict[str, Any] | None, count: int = 3) -> List[int]:
    normalized = normalize_ass_style(style)
    sizes = list(normalized["sizes"])
    if count <= 0:
        return []
    while len(sizes) < count:
        sizes.append(sizes[-1])
    return sizes[:count]


def style_from_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    return normalize_ass_style((config or {}).get("ass_style"))


def ass_style_line(
    name: str,
    style: Dict[str, Any],
    *,
    size: int,
    color: str,
    alignment: int,
    margin_v: int,
) -> str:
    spec = normalize_ass_style(style)
    bold = -1 if spec["bold"] else 0
    italic = -1 if spec["italic"] else 0
    primary = hex_to_ass(color)
    secondary = "&H000000FF"
    outline = hex_to_ass(spec["outline_color"])
    back = hex_to_ass(spec["back_color"], alpha=0x64)
    return (
        f"Style: {name},{spec['font_name']},{size},{primary},{secondary},{outline},{back},"
        f"{bold},{italic},0,0,100,100,0,0,1,{spec['outline']},{spec['shadow']},"
        f"{alignment},10,10,{margin_v},1"
    )
