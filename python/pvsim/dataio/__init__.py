"""dataio — 结果读写层。

* :class:`ResultStore`：run 目录 + 原子提交 + 加载；
* :class:`SilvacoLogWriter` / :class:`SilvacoLogReader`：SSF 2.0 (.log) 互验。
"""

from .reference import ComparisonReport, MetricDelta, compare, load_reference
from .result_store import LAYOUT_VERSION, ResultStore, RunHandle
from .scaps_export import ScapsLayer, export_scaps, layers_of, parameter_table, scaps_script
from .silvaco_log import (
    CSV_HEADER,
    LogTable,
    SilvacoLogReader,
    SilvacoLogWriter,
    jv_to_csv,
    write_jv_csv,
)

__all__ = [
    "ResultStore", "RunHandle", "LAYOUT_VERSION",
    "SilvacoLogReader", "SilvacoLogWriter", "LogTable",
    "CSV_HEADER", "jv_to_csv", "write_jv_csv",
    "load_reference", "compare", "ComparisonReport", "MetricDelta",
    "ScapsLayer", "layers_of", "scaps_script", "parameter_table", "export_scaps",
]
