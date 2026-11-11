"""PlotRegistry — 绘图函数注册表。

设计要点
--------
* 一张图 = 一个 ``PlotItem``：id、分组、标题、所需数据（``needs``）、
  绘制函数。GUI 菜单只读注册表，新增图 = 注册一个函数。
* 绘制函数签名固定为 ``fn(result, cfg, ax) -> None``，自己拿到
  ``Axes`` 后画；标题/轴标签/范围等"外观"由 :class:`PlotConfig`
  统一覆盖，绘图函数里不再散落这些参数。
* ``needs`` 声明数据依赖，``render`` 前统一校验；缺数据抛
  :class:`MissingData`，GUI 据此走兜底（参考参数）而不是画空图。
* 不持有任何求解结果，本模块只登记函数。

进程内只有一个 ``REGISTRY``，它在 import 时被 ``viz.plots`` 填充。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

PlotFn = Callable[[Any, "PlotConfig", Any], None]


# ---------------------------------------------------------------------------
# 错误与配置
# ---------------------------------------------------------------------------

class MissingData(RuntimeError):
    """绘图所需的数据不在 ``SimResult`` 里（例如没输出 CB/VB）。

    ``need`` 是具体缺的那一项（``"jv"`` / ``"fields:CB"`` / ``"optics"``），
    GUI 可以据此判断是"换数据"还是"用参考参数兜底"。
    """

    def __init__(self, plot_id: str, need: str, message: str):
        super().__init__(message)
        self.plot_id = plot_id
        self.need = need


@dataclass(frozen=True)
class PlotConfig:
    """单张图的外观覆盖项。所有字段留空即用绘图函数的默认值。"""

    title:      Optional[str] = None
    xlabel:     Optional[str] = None
    ylabel:     Optional[str] = None
    xlim:       Optional[Tuple[float, float]] = None
    ylim:       Optional[Tuple[float, float]] = None
    logx:       bool = False
    logy:       bool = False
    grid:       bool = True
    annotate:   bool = True          # 是否画指标注释框（J-V / P-V）
    point:      int = -1             # 剖面图取第几个偏压点（-1 = 最后一个）
    figsize:    Tuple[float, float] = (6.4, 4.2)
    dpi:        int = 150

    def resolve_point(self, n_points: int) -> int:
        """把 ``point``（支持负数）解析成 ``[0, n_points)`` 内的下标。"""
        if n_points <= 0:
            raise ValueError("场表为空，无法取剖面")
        idx = self.point + n_points if self.point < 0 else self.point
        if not (0 <= idx < n_points):
            raise IndexError(
                f"PlotConfig.point={self.point} 超出范围，共 {n_points} 个偏压点"
            )
        return idx


@dataclass(frozen=True)
class PlotItem:
    """注册表里的一项。``fn`` 保持原始函数，便于测试里直接调用。"""

    id:          str
    group:       str
    title:       str
    needs:       Tuple[str, ...]
    fn:          PlotFn
    description: str = ""


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

# needs 里的合法取值：jv / fields / optics / scan / compare / external
_KNOWN_NEEDS = ("jv", "fields", "optics", "scan", "compare", "external")


class PlotRegistry:
    """有序注册表（插入顺序 = 菜单顺序）。"""

    def __init__(self) -> None:
        self._items: Dict[str, PlotItem] = {}

    # ----- 登记 ---------------------------------------------------------
    def register(self, id: str, group: str, *, needs: Sequence[str] = ("jv",),
                 title: Optional[str] = None, description: str = ""):
        """装饰器：``@REGISTRY.register(id=..., group=..., needs=(...))``。"""
        for need in needs:
            if need not in _KNOWN_NEEDS:
                raise ValueError(f"未知的 needs 项 {need!r}，合法值 {_KNOWN_NEEDS}")

        def deco(fn: PlotFn) -> PlotFn:
            item = PlotItem(
                id=id, group=group, title=title or fn.__name__,
                needs=tuple(needs), fn=fn, description=description,
            )
            self.add(item)
            fn.plot_item = item          # type: ignore[attr-defined]
            return fn

        return deco

    def add(self, item: PlotItem) -> None:
        if item.id in self._items:
            raise ValueError(f"重复注册的图 id: {item.id!r}")
        self._items[item.id] = item

    # ----- 查询 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, plot_id: object) -> bool:
        return plot_id in self._items

    def __iter__(self) -> Iterator[PlotItem]:
        return iter(self._items.values())

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._items)

    def groups(self) -> Tuple[str, ...]:
        """按首次出现顺序返回分组名（菜单一级标题）。"""
        seen: List[str] = []
        for item in self._items.values():
            if item.group not in seen:
                seen.append(item.group)
        return tuple(seen)

    def by_group(self) -> Dict[str, Tuple[PlotItem, ...]]:
        out: Dict[str, List[PlotItem]] = {g: [] for g in self.groups()}
        for item in self._items.values():
            out[item.group].append(item)
        return {g: tuple(v) for g, v in out.items()}

    def get(self, plot_id: str) -> PlotItem:
        try:
            return self._items[plot_id]
        except KeyError:
            raise KeyError(
                f"未注册的图 id {plot_id!r}；可用：{', '.join(self._items) or '(空)'}"
            ) from None

    # ----- 数据校验 -----------------------------------------------------
    def check(self, item: PlotItem, result: Any) -> None:
        """缺数据时抛 :class:`MissingData`。"""
        for need in item.needs:
            if not self._has(result, need):
                raise MissingData(
                    item.id, need,
                    f"图 {item.id!r}（{item.title}）需要 {need}，"
                    f"当前结果不包含该数据",
                )

    @staticmethod
    def _has(result: Any, need: str) -> bool:
        if need == "jv":
            return getattr(result.jv, "V", None) is not None and result.jv.V.size > 0
        if need == "fields":
            return bool(result.fields.names) and result.fields.n_points > 0
        if need == "optics":
            return "G_OPT" in result.fields.names
        if need == "scan":
            return len(getattr(result, "runs", ())) > 0
        if need == "compare":
            return len(getattr(result, "entries", ())) > 0
        if need == "external":
            return getattr(result, "source", None) is not None
        return False

    # ----- 出图 ---------------------------------------------------------
    def render(self, plot_id: str, result: Any,
               cfg: Optional[PlotConfig] = None, ax: Any = None) -> Any:
        """在 ``ax`` 上画一张图；``ax=None`` 时新建 Figure 并返回 Axes。"""
        from . import axes_style

        item = self.get(plot_id)
        cfg = cfg or PlotConfig()
        self.check(item, result)

        axes_style.ensure()
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)

        item.fn(result, cfg, ax)
        self._apply(cfg, item, ax)
        return ax

    def figure(self, result: Any, plot_ids: Sequence[str],
               cfg: Optional[PlotConfig] = None, cols: Optional[int] = None) -> Any:
        """把多张图拼成一张 Figure（子图顺序 = ``plot_ids`` 顺序）。"""
        from . import axes_style

        ids = tuple(plot_ids)
        if not ids:
            raise ValueError("plot_ids 不能为空")
        cfg = cfg or PlotConfig()
        n = len(ids)
        cols = cols or min(2, n)
        rows = (n + cols - 1) // cols

        import matplotlib.pyplot as plt
        axes_style.ensure()
        fig, grid = plt.subplots(rows, cols, figsize=(cfg.figsize[0] * cols,
                                                     cfg.figsize[1] * rows),
                                 dpi=cfg.dpi, squeeze=False)
        flat = list(grid.ravel())
        for ax, pid in zip(flat, ids):
            item = self.get(pid)
            self.check(item, result)
            item.fn(result, cfg, ax)
            self._apply(cfg, item, ax)
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig

    def savefig(self, result: Any, plot_ids: Sequence[str], path: Any,
                cfg: Optional[PlotConfig] = None,
                cols: Optional[int] = None, dpi: Optional[int] = None) -> Path:
        """出图并落盘（父目录自动创建），关闭 Figure 释放内存。"""
        import matplotlib.pyplot as plt

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = self.figure(result, plot_ids, cfg=cfg, cols=cols)
        try:
            fig.savefig(out, dpi=dpi or (cfg or PlotConfig()).dpi,
                        bbox_inches="tight")
        finally:
            plt.close(fig)
        return out

    # ----- 内部 ---------------------------------------------------------
    @staticmethod
    def _apply(cfg: PlotConfig, item: PlotItem, ax: Any) -> None:
        """把 PlotConfig 的外观覆盖项落到 Axes 上（绘图函数之后执行）。"""
        if cfg.title is not None:
            ax.set_title(cfg.title)
        elif not ax.get_title():
            ax.set_title(item.title)
        if cfg.xlabel is not None:
            ax.set_xlabel(cfg.xlabel)
        if cfg.ylabel is not None:
            ax.set_ylabel(cfg.ylabel)
        if cfg.logx:
            ax.set_xscale("log")
        if cfg.logy:
            ax.set_yscale("log")
        if cfg.xlim is not None:
            ax.set_xlim(*cfg.xlim)
        if cfg.ylim is not None:
            ax.set_ylim(*cfg.ylim)
        if cfg.grid:
            ax.grid(True, alpha=0.3, lw=0.6)
        else:
            ax.grid(False)   # 不能带 lw/alpha，否则 matplotlib 视为"要开网格"


# 进程级唯一注册表
REGISTRY = PlotRegistry()
