"""插件 API 统一信封。

宿主不会替插件再包一层。写操作和列表都走 `{success, message, data}`，
这样同一套 Vue 在 V2 / V3 只解析一次。
空查询必须 success=true，用 data 表示空，避免前端当失败 Toast。
message 禁止写堆栈、Cookie、令牌、内部路径。
预览视频 / 预览 ASS 是原生流，不要走这个信封。
"""

from __future__ import annotations

from typing import Any, Dict


def ok(data: Any = None, message: str = "") -> Dict[str, Any]:
    return {"success": True, "message": str(message or ""), "data": {} if data is None else data}


def fail(message: str, data: Any = None) -> Dict[str, Any]:
    return {"success": False, "message": str(message or "操作失败"), "data": {} if data is None else data}
