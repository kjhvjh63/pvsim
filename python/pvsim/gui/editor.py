"""deck 编辑器：文本框 + 打开/保存，纯文件读写函数单独可测。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional


def read_deck(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_deck(path: Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


class DeckEditor(ttk.Frame):
    """带行号式滚动条的 deck 文本编辑器（不做语法高亮，保持轻量）。"""

    def __init__(self, master: tk.Misc, path: Optional[Path] = None):
        super().__init__(master)
        self._path: Optional[Path] = Path(path) if path else None

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

        if self._path is not None and self._path.exists():
            self.load(self._path)

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
