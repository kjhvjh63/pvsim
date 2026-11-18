"""从 ``PlotRegistry`` 动态生成菜单：新增一张图 = 注册一个函数。

``plot_menu_tree`` 是纯函数（可无显示测试），``populate_plot_menu`` /
``build_menu_bar`` 才碰 Tk。
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional, Sequence, Tuple

from ..viz.registry import PlotItem, PlotRegistry


def plot_menu_tree(registry: PlotRegistry) -> Tuple[Tuple[str, Tuple[PlotItem, ...]], ...]:
    """按注册表顺序返回 ``(分组, 图项元组)``，直接对应菜单结构。"""
    return tuple(registry.by_group().items())


def menu_label(item: PlotItem) -> str:
    return item.title


def populate_plot_menu(menu: tk.Menu, registry: PlotRegistry,
                       command: Callable[[str], None]) -> None:
    """把每个分组做成一个级联子菜单，命令参数是 plot_id。"""
    for group, items in plot_menu_tree(registry):
        submenu = tk.Menu(menu, tearoff=0)
        for item in items:
            submenu.add_command(
                label=menu_label(item),
                command=lambda pid=item.id: command(pid),
            )
        menu.add_cascade(label=group, menu=submenu)


def build_menu_bar(root: tk.Misc, registry: PlotRegistry, *,
                   on_open: Callable[[], None],
                   on_save: Callable[[], None],
                   on_run: Callable[[], None],
                   on_plot: Callable[[str], None],
                   on_quit: Callable[[], None],
                   on_mesh: Optional[Callable[[], None]] = None,
                   on_physics: Optional[Callable[[], None]] = None,
                   on_solve: Optional[Callable[[], None]] = None,
                   on_output: Optional[Callable[[], None]] = None,
                   on_compile: Optional[Callable[[], None]] = None,
                   on_export: Optional[Callable[[], None]] = None,
                   examples: Sequence[Tuple[str, str]] = (),
                   on_example: Optional[Callable[[str], None]] = None) -> tk.Menu:
    """文件 / 网格 / 物理 / 求解 / 输出 / 绘图 / 帮助。

    顶层分类沿用 DeckBuild 习惯（Mesh/Physics/Solve/Output），让熟悉 Atlas
    的用户能按同样的位置找到参数对话框。
    """
    bar = tk.Menu(root)

    file_menu = tk.Menu(bar, tearoff=0)
    file_menu.add_command(label="打开 deck…", command=on_open, accelerator="Ctrl+O")
    file_menu.add_command(label="保存 deck", command=on_save, accelerator="Ctrl+S")
    file_menu.add_separator()
    if on_compile is not None:
        file_menu.add_command(label="编译检查", command=on_compile,
                              accelerator="Ctrl+K")
    file_menu.add_command(label="运行", command=on_run, accelerator="F5")
    if on_export is not None:
        file_menu.add_command(label="导出结果 (.log/.csv)…", command=on_export)
    file_menu.add_separator()
    file_menu.add_command(label="退出", command=on_quit)
    bar.add_cascade(label="文件", menu=file_menu)

    for label, handler, hint in (("网格", on_mesh, "Mesh"),
                                 ("物理", on_physics, "Physics"),
                                 ("求解", on_solve, "Solve"),
                                 ("输出", on_output, "Output")):
        sub = tk.Menu(bar, tearoff=0)
        if handler is None:
            sub.add_command(label=f"{hint} 设置…", state="disabled")
        else:
            sub.add_command(label=f"{hint} 设置…", command=handler)
        bar.add_cascade(label=label, menu=sub)

    plot_menu = tk.Menu(bar, tearoff=0)
    populate_plot_menu(plot_menu, registry, on_plot)
    bar.add_cascade(label="绘图", menu=plot_menu)

    help_menu = tk.Menu(bar, tearoff=0)
    if examples and on_example is not None:
        ex_menu = tk.Menu(help_menu, tearoff=0)
        for label, path in examples:
            ex_menu.add_command(
                label=label,
                command=lambda p=path: on_example(p),
            )
        help_menu.add_cascade(label="示例 deck", menu=ex_menu)
        help_menu.add_separator()
    help_menu.add_command(label="共 %d 张图" % len(registry), state="disabled")
    bar.add_cascade(label="帮助", menu=help_menu)
    return bar
