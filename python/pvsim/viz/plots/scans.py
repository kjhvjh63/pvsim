"""扫描汇总图：把一次 ``SweepPlan`` 的多个 ``SimResult`` 汇总成趋势图。

``Simulator.run_plan`` 只返回 ``Tuple[PlanRun, ...]``，标签是给人看的字符串，
没有数值坐标。扫描图统一吃 :class:`ScanSet`：它在 runs 之外补上"扫描参数"
这一维（x 值、轴标签、是否对数轴）。

    runs = Simulator().run_plan(plan, base_spec)
    scan = ScanSet.from_plan(plan, runs)          # x 值直接从 plan 取
    REGISTRY.figure(scan, ["scan_jsc_vs_x", "scan_voc_vs_x",
                           "scan_ff_vs_x", "scan_pce_vs_x"], cols=2)

外部导入的 run（没有 plan 对象）用 ``ScanSet.from_runs(runs, x, xlabel)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence, Tuple

import numpy as np

from ...param.schema import (
    BiasSweep,
    DefectScan,
    DopingScan,
    IntensitySweep,
    SweepPlan,
    ThicknessScan,
)
from ..axes_style import COLORS
from ..registry import REGISTRY, MissingData, PlotConfig

if TYPE_CHECKING:
    from ...solver.simulator import PlanRun

GROUP = "扫描"

#: 指标键 → (y 轴标签, 颜色)
METRICS = {
    "Jsc": ("短路电流 Jsc (mA/cm²)", COLORS["main"]),
    "Voc": ("开路电压 Voc (V)", COLORS["secondary"]),
    "FF":  ("填充因子 FF", COLORS["tertiary"]),
    "PCE": ("效率 PCE (%)", COLORS["accent"]),
}


# ---------------------------------------------------------------------------
# ScanSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanSet:
    """扫描结果 + 数值坐标。

    ``runs`` 与 ``x`` 一一对应；``logx`` 用于掺杂/缺陷这类跨数量级的扫描，
    由 :meth:`from_plan` 自动判定。
    """

    runs:   Tuple["PlanRun", ...]
    x:      np.ndarray
    xlabel: str
    logx:   bool = False
    title:  str = ""

    def __post_init__(self):
        if not isinstance(self.runs, tuple):
            object.__setattr__(self, "runs", tuple(self.runs))
        if not isinstance(self.x, np.ndarray):
            object.__setattr__(self, "x", np.asarray(self.x, dtype=float))
        if len(self.runs) == 0:
            raise ValueError("ScanSet 至少需要 1 个 PlanRun")
        if self.x.size != len(self.runs):
            raise ValueError(
                f"ScanSet.x 长度 {self.x.size} 与 runs 数量 {len(self.runs)} 不一致"
            )

    # ----- 基本访问 -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator["PlanRun"]:
        return iter(self.runs)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(r.label for r in self.runs)

    @property
    def results(self) -> Tuple[Any, ...]:
        return tuple(r.result for r in self.runs)

    def metric(self, key: str) -> np.ndarray:
        """取某个 JV 指标随扫描变化的数组（NaN 原样保留）。"""
        if key not in METRICS:
            raise KeyError(f"未知指标 {key!r}；可选 {tuple(METRICS)}")
        return np.asarray([getattr(r.result.jv, key) for r in self.runs], dtype=float)

    # ----- 构造 ---------------------------------------------------------
    @classmethod
    def from_runs(cls, runs: Sequence["PlanRun"], x: Sequence[float], xlabel: str,
                  *, logx: bool = False, title: str = "") -> "ScanSet":
        """手工指定坐标（结果来自 ResultStore 或外部数据时用）。"""
        return cls(runs=tuple(runs), x=np.asarray(x, dtype=float),
                   xlabel=xlabel, logx=logx, title=title)

    @classmethod
    def from_plan(cls, plan: SweepPlan, runs: Sequence["PlanRun"],
                  *, title: str = "") -> "ScanSet":
        """从 plan 自身的字段取扫描轴，杜绝"x 与 run 对不上"。"""
        runs = tuple(runs)
        logx = False

        if isinstance(plan, ThicknessScan):
            x = plan.thick_nm_values
            xlabel = f"{plan.layer_name} 厚度 (nm)"
        elif isinstance(plan, DopingScan):
            x = plan.values_cm3
            xlabel = f"{plan.layer_name} {plan.kind} 掺杂 (cm$^{{-3}}$)"
            logx = True
        elif isinstance(plan, DefectScan):
            x = plan.Nt_values_cm3
            xlabel = f"{plan.layer_name} 陷阱浓度 Nt (cm$^{{-3}}$)"
            logx = True
        elif isinstance(plan, IntensitySweep):
            x = plan.sun_values
            xlabel = "光强 (sun)"
        elif isinstance(plan, BiasSweep):
            x = tuple(r.result.spec.sweep.v_stop for r in runs)
            xlabel = "偏压上限 (V)"
        else:
            raise TypeError(
                f"ScanSet.from_plan 不支持 {type(plan).__name__}；"
                f"请改用 ScanSet.from_runs(runs, x, xlabel)"
            )

        return cls(runs=runs, x=np.asarray(x, dtype=float), xlabel=xlabel,
                   logx=logx, title=title)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _require_scan(result: Any) -> ScanSet:
    if not isinstance(result, ScanSet):
        raise MissingData(
            "<scan>", "scan",
            f"扫描汇总图需要 ScanSet（Simulator.run_plan 的结果 + 扫描坐标），"
            f"当前传入 {type(result).__name__}",
        )
    return result


def _set_xlabel(ax: Any, scan: ScanSet) -> None:
    ax.set_xlabel(scan.xlabel)
    if scan.logx:
        ax.set_xscale("log")


def _no_data_note(ax: Any, what: str) -> None:
    ax.text(0.5, 0.5, f"{what} 全部为 NaN", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=COLORS["muted"])


# ---------------------------------------------------------------------------
# 指标趋势：Jsc / Voc / FF / PCE
# ---------------------------------------------------------------------------

def _plot_metric(result: Any, cfg: PlotConfig, ax: Any, key: str) -> None:
    scan = _require_scan(result)
    ylabel, color = METRICS[key]
    y = scan.metric(key)
    finite = np.isfinite(y)

    ax.plot(scan.x, y, "-", color=color, lw=1.4, zorder=2)
    ax.plot(scan.x[finite], y[finite], "o", ms=4.5, mfc="white",
            mec=color, mew=1.2, zorder=3)
    _set_xlabel(ax, scan)
    ax.set_ylabel(ylabel)
    if not np.any(finite):
        _no_data_note(ax, key)
    elif scan.title:
        ax.set_title(f"{scan.title} · {key}")


@REGISTRY.register(id="scan_jsc_vs_x", group=GROUP, needs=("scan",),
                   title="Jsc 随扫描参数",
                   description="短路电流密度 vs 扫描参数")
def plot_scan_jsc(result: Any, cfg: PlotConfig, ax: Any) -> None:
    _plot_metric(result, cfg, ax, "Jsc")


@REGISTRY.register(id="scan_voc_vs_x", group=GROUP, needs=("scan",),
                   title="Voc 随扫描参数",
                   description="开路电压 vs 扫描参数（无过零点时为 NaN）")
def plot_scan_voc(result: Any, cfg: PlotConfig, ax: Any) -> None:
    _plot_metric(result, cfg, ax, "Voc")


@REGISTRY.register(id="scan_ff_vs_x", group=GROUP, needs=("scan",),
                   title="FF 随扫描参数",
                   description="填充因子 vs 扫描参数")
def plot_scan_ff(result: Any, cfg: PlotConfig, ax: Any) -> None:
    _plot_metric(result, cfg, ax, "FF")


@REGISTRY.register(id="scan_pce_vs_x", group=GROUP, needs=("scan",),
                   title="PCE 随扫描参数",
                   description="光电转换效率 vs 扫描参数")
def plot_scan_pce(result: Any, cfg: PlotConfig, ax: Any) -> None:
    _plot_metric(result, cfg, ax, "PCE")


# ---------------------------------------------------------------------------
# J-V 家族
# ---------------------------------------------------------------------------

@REGISTRY.register(id="scan_jv_family", group=GROUP, needs=("scan",),
                   title="J-V 家族曲线",
                   description="扫描中每条 J-V 叠画，图例为各步标签")
def plot_scan_jv_family(result: Any, cfg: PlotConfig, ax: Any) -> None:
    import matplotlib.pyplot as plt

    scan = _require_scan(result)
    cmap = plt.get_cmap("viridis")
    n = len(scan)
    for i, run in enumerate(scan.runs):
        color = cmap(i / max(n - 1, 1))
        ax.plot(run.result.jv.V, -run.result.jv.J, color=color, lw=1.2,
                label=run.label)

    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    ax.set_xlabel("电压 V (V)")
    ax.set_ylabel("电流密度 J (mA/cm²)")
    if cfg.annotate and n <= 12:
        ax.legend(loc="best", frameon=False, fontsize=6.5, ncol=2 if n > 6 else 1)


def scan_summary_ids() -> Tuple[str, ...]:
    """GUI 一键出 2×2 指标汇总图时用。"""
    return ("scan_jsc_vs_x", "scan_voc_vs_x", "scan_ff_vs_x", "scan_pce_vs_x")
