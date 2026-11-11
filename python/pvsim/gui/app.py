"""App：把编辑器、求解线程、绘图层、日志面板组装成一个 Tk 窗口。

只做组装与事件路由，不持有任何物理/数值状态：

* 求解在 :class:`SolveWorker` 线程里跑，经 ``queue`` + ``after()`` 回主循环；
* 图由 ``PlotRegistry`` 动态生成菜单与下拉框；缺数据时自动走
  ``viz.fallback``（图上带"参考参数"水印）或给出缺失说明；
* ``Workspace`` / ``Simulator`` / ``PlotRegistry`` / ``DeckCompiler`` 全部
  由构造函数注入，方便测试与替换。
"""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..param.schema import SpecError
from ..script import DeckCompiler, DeckSyntaxError
from ..solver import Simulator
from ..viz import REGISTRY, MissingData, fallback
from ..viz.registry import PlotRegistry
from .editor import DeckEditor
from .log_panel import LogPanel
from .menu_builder import build_menu_bar
from .worker import (
    EVT_DONE,
    EVT_ERROR,
    EVT_LOG,
    EVT_POINT,
    SolveEvent,
    SolveWorker,
)

_LOG_DEBUG = 0
_LOG_INFO = 1
_LOG_WARN = 2
_LOG_ERROR = 3


def _default_workspace() -> Path:
    # python/pvsim/gui/app.py → 项目根
    return Path(__file__).resolve().parents[3]


class App(tk.Tk):
    """主窗口。构造完即可 ``mainloop()``；测试里也可以只 ``update()``。"""

    POLL_MS = 50

    def __init__(self, workspace: Optional[Path] = None,
                 simulator: Any = None,
                 registry: Optional[PlotRegistry] = None,
                 compiler: Any = None):
        super().__init__()
        self.title("pvsim — 钙钛矿太阳能电池仿真平台")
        self.geometry("1220x780")
        self.minsize(900, 560)

        self.workspace = Path(workspace) if workspace is not None else _default_workspace()
        self.registry = registry or REGISTRY
        self.simulator = simulator or Simulator()
        self.compiler = compiler or DeckCompiler()

        self.result: Any = None       # 最近一次求解结果（仅用于绘图）
        self.spec: Any = None         # 最近一次编译出的 ProblemSpec
        self._worker: Optional[SolveWorker] = None
        self._events: "queue.Queue[SolveEvent]" = queue.Queue()
        self._plot_ids: List[str] = list(self.registry.ids())

        self._build()
        self._bind_keys()
        self._load_default_deck()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.option_add("*tearOff", False)

        menubar = build_menu_bar(
            self, self.registry,
            on_open=self.open_deck, on_save=self.save_deck,
            on_run=self.run_current, on_plot=self.draw,
            on_quit=self.close,
        )
        self.configure(menu=menubar)

        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="打开", command=self.open_deck).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="保存", command=self.save_deck).pack(side=tk.LEFT, padx=4)
        self._run_btn = ttk.Button(toolbar, text="运行 (F5)", command=self.run_current)
        self._run_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="图：").pack(side=tk.LEFT, padx=(16, 2))
        labels = self._plot_labels()
        self._plot_var = tk.StringVar(value=labels[0] if labels else "")
        self._plot_box = ttk.Combobox(toolbar, textvariable=self._plot_var,
                                      values=labels, state="readonly", width=30)
        self._plot_box.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="绘图", command=self.draw_selected).pack(side=tk.LEFT, padx=4)

        status = ttk.Frame(self)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self._status = tk.StringVar(value="就绪")
        ttk.Label(status, textvariable=self._status, anchor="w").pack(
            side=tk.LEFT, padx=6, pady=3)
        self._progress = ttk.Progressbar(status, mode="determinate", length=220)
        self._progress.pack(side=tk.RIGHT, padx=6, pady=3)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        self.editor = DeckEditor(left)
        self.editor.pack(fill=tk.BOTH, expand=True)
        paned.add(left, weight=2)

        right = ttk.Notebook(paned)
        plot_tab = ttk.Frame(right)
        self.figure = Figure(figsize=(6.4, 4.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._nav = NavigationToolbar2Tk(self.canvas, plot_tab)
        self._nav.update()
        right.add(plot_tab, text="绘图")

        log_tab = ttk.Frame(right)
        self.log_panel = LogPanel(log_tab, height=12)
        self.log_panel.pack(fill=tk.BOTH, expand=True)
        right.add(log_tab, text="日志")
        paned.add(right, weight=3)

    def _bind_keys(self) -> None:
        self.bind("<F5>", lambda _e: self.run_current())
        self.bind("<Control-o>", lambda _e: self.open_deck())
        self.bind("<Control-s>", lambda _e: self.save_deck())

    def _plot_labels(self) -> List[str]:
        return [f"{self.registry.get(pid).group} · {self.registry.get(pid).title}"
                for pid in self._plot_ids]

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------
    def _load_default_deck(self) -> None:
        candidate = self.workspace / "decks" / "device_example.tst"
        if candidate.exists():
            try:
                self.editor.load(candidate)
                self.log(_LOG_INFO, f"已加载示例 deck：{candidate}")
            except OSError as exc:
                self.log(_LOG_WARN, f"读取示例 deck 失败：{exc}")
        else:
            self.log(_LOG_WARN, f"未找到示例 deck：{candidate}")

    def open_deck(self) -> None:
        path = filedialog.askopenfilename(
            title="打开 deck", filetypes=[("Atlas deck", "*.tst"), ("所有文件", "*.*")],
            initialdir=str(self.workspace / "decks"),
        )
        if not path:
            return
        try:
            self.editor.load(Path(path))
            self.log(_LOG_INFO, f"打开：{path}")
        except OSError as exc:
            self.log(_LOG_ERROR, f"打开失败：{exc}")

    def save_deck(self) -> None:
        path = self.editor.path
        if path is None:
            chosen = filedialog.asksaveasfilename(
                title="保存 deck", defaultextension=".tst",
                filetypes=[("Atlas deck", "*.tst"), ("所有文件", "*.*")],
                initialdir=str(self.workspace / "decks"),
            )
            if not chosen:
                return
            path = Path(chosen)
        try:
            self.editor.save(path)
            self.log(_LOG_INFO, f"已保存：{path}")
        except OSError as exc:
            self.log(_LOG_ERROR, f"保存失败：{exc}")

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def run_current(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.log(_LOG_WARN, "上一次求解还没结束")
            return

        try:
            self.spec = self.compiler.compile_text(self.editor.get_text())
        except (SpecError, DeckSyntaxError) as exc:
            self.log(_LOG_ERROR, f"编译失败：{exc}")
            for detail in getattr(exc, "details", ()):
                self.log(_LOG_ERROR, f"  - {detail}")
            messagebox.showerror("编译失败", str(exc))
            return

        n_points = max(1, int(self.spec.sweep.n_points))
        self.log(_LOG_INFO,
                 f"编译通过：{len(self.spec.device.layers)} 层 / "
                 f"{n_points} 个偏压点 / {self.spec.output.run_name}")
        self._progress.configure(maximum=n_points, value=0)
        self._status.set("求解中…")
        self._run_btn.state(["disabled"])

        self._events = queue.Queue()
        self._worker = SolveWorker(self.spec, self.simulator, self._events)
        self._worker.start()
        self.after(self.POLL_MS, self._poll_events)

    def _poll_events(self) -> None:
        drained = 0
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            drained += 1
            self._handle(event)

        alive = self._worker is not None and self._worker.is_alive()
        if alive or drained:
            self.after(self.POLL_MS, self._poll_events)
        elif self._worker is not None:
            self._run_btn.state(["!disabled"])
            self._worker = None

    def _handle(self, event: SolveEvent) -> None:
        if event.kind == EVT_LOG:
            self.log(event.level, str(event.payload))
        elif event.kind == EVT_POINT:
            self._progress.configure(value=event.index + 1)
            self._status.set(f"求解中… {event.index + 1}/{event.total}  {event.payload}")
        elif event.kind == EVT_DONE:
            self.result = event.payload
            self._progress.configure(
                value=float(self._progress.cget("maximum")))
            summary = getattr(event.payload, "summary", None)
            text = summary() if callable(summary) else str(event.payload)
            self.log(_LOG_INFO, text)
            self._status.set(f"完成：{text}")
            self._run_btn.state(["!disabled"])
            if "cell_jv_curve" in self._plot_ids:
                self.select_plot("cell_jv_curve")
                self.draw("cell_jv_curve")
        elif event.kind == EVT_ERROR:
            self.log(_LOG_ERROR, f"求解失败：{event.payload!r}")
            self._status.set("求解失败")
            self._run_btn.state(["!disabled"])
            messagebox.showerror("求解失败", str(event.payload))

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def select_plot(self, plot_id: str) -> None:
        if plot_id in self._plot_ids:
            self._plot_box.current(self._plot_ids.index(plot_id))

    def draw_selected(self) -> None:
        index = self._plot_box.current()
        if index < 0:
            return
        self.draw(self._plot_ids[index])

    def draw(self, plot_id: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.result is None and self.spec is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "先运行一次求解", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.5")
            self._status.set("无数据可画")
            self.canvas.draw_idle()
            return

        if self.result is not None:
            try:
                self.registry.render(plot_id, self.result, ax=ax)
                self._status.set(f"已绘制：{self.registry.get(plot_id).title}")
                self.canvas.draw_idle()
                return
            except MissingData as exc:
                self._fallback_or_note(plot_id, ax, exc)
        else:
            self._fallback_or_note(plot_id, ax, None)

        self.canvas.draw_idle()

    def _fallback_or_note(self, plot_id: str, ax: Any,
                          exc: Optional[MissingData]) -> None:
        target = self.result if self.result is not None else self.spec
        if fallback.supports(plot_id):
            fallback.render_reference(plot_id, target, ax=ax)
            self.log(_LOG_WARN, f"{plot_id}: 使用参考参数（非仿真结果）")
            self._status.set("参考参数兜底（非仿真结果）")
            return

        ax.set_axis_off()
        need = exc.need if exc is not None else "仿真结果"
        ax.text(0.5, 0.5, f"缺少数据：{need}\n请先运行求解或补充输出场量",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.45")
        self.log(_LOG_WARN, f"{plot_id}: 缺少数据 {need}，无法出图")
        self._status.set(f"缺少数据：{need}")

    # ------------------------------------------------------------------
    # 其它
    # ------------------------------------------------------------------
    def log(self, level: int, message: str) -> None:
        self.log_panel.append(level, message)

    def close(self) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            self.log(_LOG_WARN, "后台求解仍在运行，关闭窗口不会中断它（守护线程）")
        self.destroy()


def main() -> None:
    App().mainloop()
