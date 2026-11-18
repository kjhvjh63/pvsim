"""deck 编辑器：文本框 + 打开/保存 + 轻量语法高亮，纯文件读写函数单独可测。"""

from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

# 与 parser._HANDLERS 对齐的语句关键字（高亮用；新增语句记得同步）
KEYWORDS = frozenset({
    "title", "mesh", "region", "doping", "contact", "electrode", "material",
    "models", "model", "optics", "method", "solve", "log", "save", "output",
    "extract", "comment", "go", "quit", "plot", "tonyplot",
})

_TOKEN_RE = re.compile(
    r"(?P<comment>#[^\n]*)"
    r"|(?P<string>\"[^\"\n]*\"|'[^'\n]*')"
    r"|(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_.\-/]*)"
)


def read_deck(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_deck(path: Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def classify(text: str):
    """把 deck 文本切成 ``(tag, start, end)`` 三元组；纯函数，便于测试。

    tag ∈ keyword / param / number / string / comment；其余不产出（保持默认样式）。
    """
    spans = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "comment":
            spans.append(("comment", m.start(), m.end()))
        elif kind == "string":
            spans.append(("string", m.start(), m.end()))
        elif kind == "number":
            spans.append(("number", m.start(), m.end()))
        elif kind == "word":
            word = m.group(0)
            if word.lower() in KEYWORDS:
                spans.append(("keyword", m.start(), m.end()))
            elif text[m.end():m.end() + 1] == "=":
                spans.append(("param", m.start(), m.end()))
    return spans


class DeckEditor(ttk.Frame):
    """deck 文本编辑器：行号式滚动条 + DeckBuild 风格语法高亮。"""

    HIGHLIGHT_DELAY_MS = 120

    def __init__(self, master: tk.Misc, path: Optional[Path] = None):
        super().__init__(master)
        self._path: Optional[Path] = Path(path) if path else None
        self._hl_job: Optional[str] = None

        self.text = tk.Text(self, wrap="none", undo=True,
                            font=("Consolas", 10))
        yscroll = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                command=self.text.yview)
        xscroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                                command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set,
                            xscrollcommand=xscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._setup_tags()
        self.text.bind("<KeyRelease>", self._schedule_highlight)

        if self._path is not None and self._path.exists():
            self.load(self._path)

    # ----- 高亮 ---------------------------------------------------------
    def _setup_tags(self) -> None:
        self.text.tag_configure("keyword", foreground="#0b5394",
                                font=("Consolas", 10, "bold"))
        self.text.tag_configure("param", foreground="#7a1fa2")
        self.text.tag_configure("number", foreground="#0f7b0f")
        self.text.tag_configure("string", foreground="#b45309")
        self.text.tag_configure("comment", foreground="#8a8a8a")

    def _schedule_highlight(self, _event: Optional[tk.Event] = None) -> None:
        if self._hl_job is not None:
            self.after_cancel(self._hl_job)
        self._hl_job = self.after(self.HIGHLIGHT_DELAY_MS, self.highlight)

    def highlight(self) -> None:
        self._hl_job = None
        for tag in ("keyword", "param", "number", "string", "comment"):
            self.text.tag_remove(tag, "1.0", "end")
        content = self.text.get("1.0", "end-1c")
        for tag, start, end in classify(content):
            self.text.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

    # ----- 文件 ---------------------------------------------------------
    @property
    def path(self) -> Optional[Path]:
        return self._path

    def load(self, path: Path) -> None:
        self._path = Path(path)
        self.set_text(read_deck(self._path))

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("还没有目标路径")
        self._path = write_deck(target, self.get_text())
        return self._path

    # ----- 文本 ---------------------------------------------------------
    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.highlight()
