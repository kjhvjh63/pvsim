"""电学类绘图：J-V、P-V、能带、电场、载流子浓度、净复合率。

统一的 x 轴是 nm（与 ``FieldTable.x_nm`` 一致）；剖面图默认取最后一个
偏压点，可用 ``PlotConfig(point=i)`` 指定其它点。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

from ..axes_style import COLORS, annotate_metrics, layer_span
from ..registry import REGISTRY, MissingData, PlotConfig

if TYPE_CHECKING:               # 只是类型提示，不在 import 期拉起求解层
    from ..solver.simulator import SimResult

GROUP = "电学"


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _require_fields(result: "SimResult", plot_id: str, *names: str) -> None:
    missing = [n for n in names if n not in result.fields.names]
    if missing:
        raise MissingData(
            plot_id, "fields:" + ",".join(missing),
            f"结果缺少场量 {'/'.join(missing)}；当前输出："
            f"{', '.join(result.fields.names) or '(空)'}。"
            f"请在 OutputSpec.fields 中加入后重跑。",
        )


def _profile(result: "SimResult", cfg: PlotConfig, plot_id: str,
             name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """取某个偏压点上的一个场量剖面 → (x_nm, values, point_index)。"""
    _require_fields(result, plot_id, name)
    idx = cfg.resolve_point(result.fields.n_points)
    x = np.asarray(result.fields.x_nm, dtype=float)
    y = np.asarray(result.fields[name][idx, :], dtype=float)
    return x, y, idx


def _at(result: "SimResult", idx: int, base: str) -> str:
    """剖面图标题带上偏压，避免多张图混在一起时分不清。"""
    V = float(result.V[idx]) if result.points else float("nan")
    return f"{base}  @ V = {V:.3f} V"


# ---------------------------------------------------------------------------
# J-V / P-V
# ---------------------------------------------------------------------------

@REGISTRY.register(id="cell_jv_curve", group=GROUP, needs=("jv",),
                   title="J-V 曲线",
                   description="光照 J-V：纵轴为交付电流 −J（短路时 >0），"
                               "标注 Jsc/Voc/FF/PCE")
def plot_jv_curve(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    V = np.asarray(result.jv.V, dtype=float)
    J = -np.asarray(result.jv.J, dtype=float)   # 交付电流：与常规 J-V 象限一致
    ax.plot(V, J, "-", color=COLORS["main"], label="光照 J-V")
    ax.plot(V, J, "o", ms=3.0, mfc="white", mec=COLORS["main"], mew=1.0)
    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    ax.axvline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    ax.set_xlabel("电压 V (V)")
    ax.set_ylabel("电流密度 J (mA/cm²)")
    if cfg.annotate:
        annotate_metrics(ax, result.jv, loc="lower left")


@REGISTRY.register(id="cell_pv_curve", group=GROUP, needs=("jv",),
                   title="P-V 曲线",
                   description="输出功率 P = −J·V，并标出最大功率点")
def plot_pv_curve(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    V = np.asarray(result.jv.V, dtype=float)
    P = -np.asarray(result.jv.J, dtype=float) * V
    ax.plot(V, P, "-", color=COLORS["main"], label="P = −J·V")
    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    ax.set_xlabel("电压 V (V)")
    ax.set_ylabel("功率密度 P (mW/cm²)")

    if cfg.annotate and P.size and np.any(np.isfinite(P)):
        k = int(np.nanargmax(P))
        if math.isfinite(P[k]) and P[k] > 0:
            ax.plot(V[k], P[k], "o", ms=5, color=COLORS["accent"], zorder=4)
            ax.annotate(f"Pmax = {P[k]:.2f} mW/cm²\n@ V = {V[k]:.3f} V",
                        xy=(V[k], P[k]), xytext=(10, -18),
                        textcoords="offset points", fontsize=8.5,
                        color=COLORS["accent"],
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color=COLORS["accent"]))


# ---------------------------------------------------------------------------
# 能带
# ---------------------------------------------------------------------------

@REGISTRY.register(id="cell_band_diagram", group=GROUP, needs=("fields",),
                   title="能带图",
                   description="导带底/价带顶随位置的剖面（需要输出 CB/VB）")
def plot_band_diagram(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    _require_fields(result, "cell_band_diagram", "CB", "VB")
    idx = cfg.resolve_point(result.fields.n_points)
    x = np.asarray(result.fields.x_nm, dtype=float)
    Ec = np.asarray(result.fields["CB"][idx, :], dtype=float)
    Ev = np.asarray(result.fields["VB"][idx, :], dtype=float)

    ax.plot(x, Ec, color=COLORS["main"], label="E_C")
    ax.plot(x, Ev, color=COLORS["secondary"], label="E_V")
    ax.fill_between(x, Ev, Ec, color=COLORS["main"], alpha=0.08, lw=0)

    # 准费米能级：有就画，没有不报错
    if "PHIN" in result.fields.names:
        ax.plot(x, result.fields["PHIN"][idx, :], "--", lw=1.2,
                color=COLORS["tertiary"], label="E_Fn")
    if "PHIP" in result.fields.names:
        ax.plot(x, result.fields["PHIP"][idx, :], ":", lw=1.2,
                color=COLORS["accent"], label="E_Fp")

    layer_span(ax, result.spec.device)
    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("能量 (eV)")
    ax.set_title(_at(result, idx, "能带图"))
    ax.legend(loc="best", frameon=False)


# ---------------------------------------------------------------------------
# 剖面：电场 / 载流子 / 复合
# ---------------------------------------------------------------------------

@REGISTRY.register(id="cell_efield_profile", group=GROUP, needs=("fields",),
                   title="电场分布",
                   description="电场 E(x) 剖面，层界面以底色区分")
def plot_efield_profile(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    x, E, idx = _profile(result, cfg, "cell_efield_profile", "EFIELD")
    ax.plot(x, E, color=COLORS["main"])
    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    layer_span(ax, result.spec.device)
    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("电场 E (V/cm)")
    ax.set_title(_at(result, idx, "电场分布"))


@REGISTRY.register(id="cell_carrier_profile", group=GROUP, needs=("fields",),
                   title="载流子浓度",
                   description="电子/空穴浓度对数剖面（需要输出 N/P）")
def plot_carrier_profile(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    _require_fields(result, "cell_carrier_profile", "N", "P")
    idx = cfg.resolve_point(result.fields.n_points)
    x = np.asarray(result.fields.x_nm, dtype=float)
    n = np.maximum(np.asarray(result.fields["N"][idx, :], dtype=float), 1e-30)
    p = np.maximum(np.asarray(result.fields["P"][idx, :], dtype=float), 1e-30)

    ax.semilogy(x, n, color=COLORS["main"], label="电子 n")
    ax.semilogy(x, p, color=COLORS["secondary"], label="空穴 p")
    layer_span(ax, result.spec.device)
    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("浓度 (cm$^{-3}$)")
    ax.set_title(_at(result, idx, "载流子浓度"))
    ax.legend(loc="best", frameon=False)


@REGISTRY.register(id="cell_recomb_profile", group=GROUP, needs=("fields",),
                   title="净复合率",
                   description="SRH 净复合率 R(x)（需要输出 R_NET）")
def plot_recomb_profile(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    x, R, idx = _profile(result, cfg, "cell_recomb_profile", "R_NET")
    ax.plot(x, R, color=COLORS["tertiary"])
    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    layer_span(ax, result.spec.device)
    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("净复合率 R (cm$^{-3}$ s$^{-1}$)")
    ax.set_title(_at(result, idx, "净复合率"))
