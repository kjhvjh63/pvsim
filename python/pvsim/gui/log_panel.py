"""日志面板：环形缓冲 + 按级别着色的只读文本框。

``LogBuffer`` 不依赖 Tk，可以单独测试；``LogPanel`` 只是它的显示层。
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from tkinter import ttk
from typing import Deque, Optional, Tuple

LEVEL_DEBUG = 0
LEVEL_INFO = 1
LEVEL_WARN = 2
LEVEL_ERROR = 3

LEVEL_NAMES = {LEVEL_DEBUG: "DEBUG", LEVEL_INFO: "INFO",
               LEVEL_WARN: "WARN", LEVEL_ERROR: "ERROR"}

_TAG_OF = {LEVEL_DEBUG: "debug", LEVEL_INFO: "info",
           LEVEL_WARN: "warn", LEVEL_ERROR: "error"}
_TAG_COLOR = {"debug": "#7a7a7a", "info": "#1f4e79",
              "warn": "#b26a00", "error": "#b03030"}


@dataclass(frozen=True)
class LogRecord:
    level:   int
    message: str
    stamp:   str


def format_record(record: LogRecord) -> str:
    return f"[{record.stamp}] {LEVEL_NAMES.get(record.level, 'LOG'):5s} {record.message}"


class LogBuffer:
    """有上限的日志缓冲（默认保留 1000 条）。"""

    def __init__(self, maxlen: int = 1000):
        self._items: Deque[LogRecord] = deque(maxlen=maxlen)

    def append(self, level: int, message: str,
               now: Optional[datetime] = None) -> LogRecord:
        stamp = (now or datetime.now()).strftime("%H:%M:%S")
        record = LogRecord(level=int(level), message=str(message), stamp=stamp)
        self._items.append(record)
        return record

    def clear(self) -> None:
        self._items.clear()

    @property
    def records(self) -> Tuple[LogRecord, ...]:
        return tuple(self._items)

    def text(self) -> str:
        return "\n".join(format_record(r) for r in self._items)

    def __len__(self) -> int:
        return len(self._items)


class LogPanel(ttk.Frame):
    """只读日志视图；``append`` 同时写缓冲和文本框。"""

    def __init__(self, master: tk.Misc, height: int = 10,
                 maxlen: int = 1000):
        super().__init__(master)
        self.buffer = LogBuffer(maxlen=maxlen)

        self._text = tk.Text(self, height=height, wrap="word",
                             state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL,
                               command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        for tag, color in _TAG_COLOR.items():
            self._text.tag_configure(tag, foreground=color)

    def append(self, level: int, message: str) -> LogRecord:
        record = self.buffer.append(level, message)
        tag = _TAG_OF.get(record.level, "info")
        self._text.configure(state="normal")
        self._text.insert("end", format_record(record) + "\n", tag)
        self._text.configure(state="disabled")
        self._text.see("end")
        return record

    def clear(self) -> None:
        self.buffer.clear()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
