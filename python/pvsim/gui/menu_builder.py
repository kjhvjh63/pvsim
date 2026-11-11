"""从 ``PlotRegistry`` 动态生成菜单：新增一张图 = 注册一个函数。

``plot_menu_tree`` 是纯函数（可无显示测试），``populate_plot_menu`` /
``build_menu_bar`` 才碰 Tk。
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Tuple

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
                   on_quit: Callable[[], None]) -> tk.Menu:
    """文件 / 绘图 / 帮助 三项菜单栏。"""
    bar = tk.Menu(root)

    file_menu = tk.Menu(bar, tearoff=0)
    file_menu.add_command(label="打开 deck…", command=on_open, accelerator="Ctrl+O")
    file_menu.add_command(label="保存 deck", command=on_save, accelerator="Ctrl+S")
    file_menu.add_separator()
    file_menu.add_command(label="运行", command=on_run, accelerator="F5")
    file_menu.add_separator()
    file_menu.add_command(label="退出", command=on_quit)
    bar.add_cascade(label="文件", menu=file_menu)

    plot_menu = tk.Menu(bar, tearoff=0)
    populate_plot_menu(plot_menu, registry, on_plot)
    bar.add_cascade(label="绘图", menu=plot_menu)

    help_menu = tk.Menu(bar, tearoff=0)
    help_menu.add_command(label="共 %d 张图" % len(registry), state="disabled")
    bar.add_cascade(label="帮助", menu=help_menu)
    return bar
