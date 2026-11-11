"""dataio — 结果读写层。

* :class:`ResultStore`：run 目录 + 原子提交 + 加载；
* :class:`SilvacoLogWriter` / :class:`SilvacoLogReader`：SSF 2.0 (.log) 互验。
"""

from .result_store import LAYOUT_VERSION, ResultStore, RunHandle
from .silvaco_log import LogTable, SilvacoLogReader, SilvacoLogWriter

__all__ = [
    "ResultStore", "RunHandle", "LAYOUT_VERSION",
    "SilvacoLogReader", "SilvacoLogWriter", "LogTable",
]
