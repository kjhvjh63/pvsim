"""结构类绘图：器件层堆叠示意图、平带能带对齐。

这两张图只依赖 ``result.spec.device``，因此 ``needs=()`` —— 不跑求解也能出图，
求解结果为空（field 表为空）时同样可用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from matplotlib.patches import Rectangle

from ..axes_style import COLORS, layer_span
from ..registry import REGISTRY, PlotConfig

if TYPE_CHECKING:
    from ..solver.simulator import SimResult

GROUP = "结构"

# 非电极、非吸收层的填充色（循环使用）
_FILL_CYCLE = ("#cfe3f2", "#dcecd3", "#eadcf2", "#d8e8e4", "#f2e6cf")
_CONTACT_FILL = "#d5d5d5"
_ABSORBER_FILL = "#f7e0c2"

# 堆叠示意图的纵向布局（0~1 相对坐标）：上部层块 + 下部参数表
_Y_LO, _Y_HI = 0.62, 1.00
_TABLE_H = 0.56


def _doping_text(layer: Any) -> str:
    """Nd / Na → 简短的掺杂标注；都没有就写"本征"。"""
    parts = []
    if layer.Nd_cm3 > 0:
        parts.append(f"$N_D$={layer.Nd_cm3:.1g}")
    if layer.Na_cm3 > 0:
        parts.append(f"$N_A$={layer.Na_cm3:.1g}")
    return "  ".join(parts) or "本征"


def _role(layer: Any) -> str:
    if layer.is_contact:
        return "电极"
    return "吸收层" if _is_absorber(layer) else "传输层"


def _cum_edges(device: Any) -> Tuple[float, ...]:
    edges = [0.0]
    for layer in device.layers:
        edges.append(edges[-1] + float(layer.thick_nm))
    return tuple(edges)


def _is_absorber(layer: Any) -> bool:
    return layer.is_gen and not layer.is_contact


# ---------------------------------------------------------------------------
# 层堆叠示意图
# ---------------------------------------------------------------------------

@REGISTRY.register(id="structure_stack", group=GROUP, needs=(),
                   title="器件结构示意图",
                   description="按真实厚度比例的层堆叠 + 厚度/掺杂/Eg 参数表")
def plot_structure_stack(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    device = result.spec.device
    edges = _cum_edges(device)
    total = edges[-1]

    for i, layer in enumerate(device.layers):
        if layer.is_contact:
            face, hatch = _CONTACT_FILL, "//"
        elif _is_absorber(layer):
            face, hatch = _ABSORBER_FILL, None
        else:
            face, hatch = _FILL_CYCLE[i % len(_FILL_CYCLE)], None

        left, width = edges[i], edges[i + 1] - edges[i]
        ax.add_patch(Rectangle((left, _Y_LO), width, _Y_HI - _Y_LO,
                               facecolor=face, edgecolor="0.25", lw=0.9,
                               hatch=hatch, zorder=2))
        # 窄层（如 40 nm 的 TiO2）竖排层名，宽层横排
        ax.text(left + 0.5 * width, 0.5 * (_Y_LO + _Y_HI), layer.name,
                rotation=90 if width < 0.12 * total else 0,
                ha="center", va="center", fontsize=8.5, color="0.1", zorder=3)

    # 入射光方向与两端极性（标注在层块上方，不占层内空间）
    pin = result.spec.optics.intensity_mW_cm2
    ax.annotate("", xy=(0.5 * total, _Y_HI + 0.01),
                xytext=(0.5 * total, _Y_HI + 0.09),
                arrowprops=dict(arrowstyle="->", lw=1.3,
                                color=COLORS["accent"]),
                annotation_clip=False)
    ax.text(0.5 * total, _Y_HI + 0.11, f"AM1.5G  {pin:g} mW/cm²",
            ha="center", va="bottom", fontsize=8, color=COLORS["accent"],
            clip_on=False)
    ax.text(0.0, _Y_HI + 0.02, "阳极侧 / 入射面", ha="left", va="bottom",
            fontsize=7.5, color="0.4", clip_on=False)
    ax.text(total, _Y_HI + 0.02, "阴极侧", ha="right", va="bottom",
            fontsize=7.5, color="0.4", clip_on=False)

    # 参数表：窄层的文字放在行里，不跟层宽抢空间
    rows = [[l.name, f"{l.thick_nm:g}", f"{l.Eg_eV:g}", _doping_text(l),
             _role(l)] for l in device.layers]
    table = ax.table(
        cellText=rows,
        colLabels=("层", "厚度 (nm)", "Eg (eV)", "掺杂 (cm$^{-3}$)", "角色"),
        colWidths=(0.22, 0.15, 0.14, 0.29, 0.20),
        cellLoc="center", colLoc="center",
        bbox=(0.0, 0.0, 1.0, _TABLE_H),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("0.8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")

    ax.set_xlim(0.0, total)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("位置 x (nm)")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# 平带能带对齐
# ---------------------------------------------------------------------------

@REGISTRY.register(id="structure_band_align", group=GROUP, needs=(),
                   title="平带能带对齐",
                   description="由各层 Eg/χ 推算的平带阶梯能带（不含内建电势）")
def plot_band_align(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    device = result.spec.device
    edges = _cum_edges(device)

    xs, ec, ev = [], [], []
    for i, layer in enumerate(device.layers):
        e_c = -float(layer.chi_eV)               # 真空能级为 0
        e_v = e_c - float(layer.Eg_eV)
        xs.extend((edges[i], edges[i + 1]))
        ec.extend((e_c, e_c))
        ev.extend((e_v, e_v))

    ax.plot(xs, ec, color=COLORS["main"], label="E_C")
    ax.plot(xs, ev, color=COLORS["secondary"], label="E_V")
    layer_span(ax, device, alpha=0.05)

    # 只给吸收层标 Eg 双箭头，避免五层都标导致拥挤
    for i, layer in enumerate(device.layers):
        if not _is_absorber(layer):
            continue
        center = 0.5 * (edges[i] + edges[i + 1])
        e_c = -float(layer.chi_eV)
        e_v = e_c - float(layer.Eg_eV)
        ax.annotate("", xy=(center, e_c), xytext=(center, e_v),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.3"))
        ax.text(center, 0.5 * (e_c + e_v), f"Eg = {layer.Eg_eV:g} eV",
                ha="center", va="center", fontsize=8, color="0.15",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.8", alpha=0.9))

    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("能量 (eV，相对真空)")
    ax.text(0.01, 0.02, "平带近似：由 Eg/χ 推算，非求解结果",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=COLORS["muted"])
    ax.legend(loc="best", frameon=False)
