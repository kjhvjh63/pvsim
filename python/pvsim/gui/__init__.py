"""gui — Tkinter 交互界面（只组装，不做计算）。

    python -m pvsim.gui

``App`` 依赖显示器；无显示环境下仍可导入纯逻辑部分（``LogBuffer``、
``plot_menu_tree``、``SolveWorker``、``read_deck/write_deck``）。
"""

from .app import App, main
from .editor import DeckEditor, read_deck, write_deck
from .log_panel import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARN,
    LogBuffer,
    LogPanel,
    LogRecord,
    format_record,
)
from .menu_builder import (
    build_menu_bar,
    menu_label,
    plot_menu_tree,
    populate_plot_menu,
)
from .worker import (
    EVT_DONE,
    EVT_ERROR,
    EVT_LOG,
    EVT_POINT,
    SolveEvent,
    SolveWorker,
)

__all__ = [
    "App", "main",
    "DeckEditor", "read_deck", "write_deck",
    "LogBuffer", "LogPanel", "LogRecord", "format_record",
    "LEVEL_DEBUG", "LEVEL_INFO", "LEVEL_WARN", "LEVEL_ERROR",
    "build_menu_bar", "populate_plot_menu", "plot_menu_tree", "menu_label",
    "SolveWorker", "SolveEvent",
    "EVT_LOG", "EVT_POINT", "EVT_DONE", "EVT_ERROR",
]
