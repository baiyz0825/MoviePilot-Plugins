"""多行路径解析。

设计合同：`watch_paths` / `strm_paths` 存真实换行，一行一条容器内绝对路径。
海拉鲁曾把字面 `\\n` 画进输入框；这里解析时只按真实换行切，并丢掉空行。
不要把用户写进框里的两个字符 `\\` + `n` 当成分隔符，以免路径被拦腰切断。
"""

from __future__ import annotations

from typing import Any, Iterable, List


def parse_multiline_paths(value: Any) -> List[str]:
    """把配置里的多行目录拆成路径列表。

    - 接受 str / 已是 list / None。
    - 只按真实 `\\n` / `\\r\\n` 切，不把字面 ``\\n`` 当分隔。
    - 每行 trim；空行丢掉。保序、去重（后出现的丢掉）。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_lines = [str(item) for item in value]
    else:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = text.split("\n")
    seen = set()
    paths: List[str] = []
    for line in raw_lines:
        item = line.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        paths.append(item)
    return paths


def join_multiline_paths(paths: Iterable[str]) -> str:
    """回写成真实换行字符串，给 VTextarea 绑定。"""
    return "\n".join(parse_multiline_paths(list(paths)))
