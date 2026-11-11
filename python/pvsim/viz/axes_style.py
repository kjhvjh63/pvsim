"""统一绘图外观：字体、配色、层界面底色、指标注释框。

约定
----
* 调用方不需要手动设字体；:func:`ensure` 在首次出图时挑一个可用的中文字体
  写进 rcParams，并关闭 ``unicode_minus``（否则负号变方框）。
* 剖面图统一用 :func:`layer_span` 画层界面（交替浅灰）+ 顶部层名，
  这样电场/载流子/产生率等图的 x 轴语义一致。
* 配色集中在 ``COLORS``，新增图不再各写各的色值。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Tuple

COLORS = {
    "main":      "#1f6fb4",   # 主曲线（J-V、电场）
    "secondary": "#d1495b",   # 次曲线（空穴、价带）
    "tertiary":  "#2a9d8f",   # 第三条（复合、产生率）
    "accent":    "#e9a13b",   # 关键点（Pmax、Voc）
    "muted":     "#8c8c8c",
}

# 按优先级排序：Windows / Linux / macOS 常见中文字体
_FONT_CANDIDATES: Tuple[str, ...] = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans",
)

_STYLE_APPLIED = False


def _pick_font() -> Optional[str]:
    """在 matplotlib 已发现的字体里挑第一个可用的中文族。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return None


def ensure() -> None:
    """幂等地把统一风格写进 rcParams（首次调用时探测字体）。"""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib as mpl

    font = _pick_font()
    if font:
        mpl.rcParams["font.sans-serif"] = [font] + list(mpl.rcParams["font.sans-serif"])
        mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams.update({
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.autolayout": False,
        "lines.linewidth":   1.6,
    })
    _STYLE_APPLIED = True


def label(ax: Any, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    """一次性设置轴标签（可选标题）。"""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def layer_span(ax: Any, device: Any, *, alpha: float = 0.07,
               labels: bool = True, label_fontsize: float = 7.0) -> Tuple[float, float]:
    """按 ``device.layers`` 的厚度画层界面底色，返回 (x0, x1)。

    x 轴单位必须是 nm（与 ``FieldTable.x_nm`` 一致）；奇数层填浅灰以区分
    相邻层，层名竖排在顶部。
    """
    from matplotlib.transforms import blended_transform_factory

    edges = [0.0]
    for layer in device.layers:
        edges.append(edges[-1] + float(layer.thick_nm))

    for i in range(len(device.layers)):
        if i % 2 == 1:
            ax.axvspan(edges[i], edges[i + 1], color="0.5", alpha=alpha,
                       lw=0, zorder=0)

    if labels:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for i, layer in enumerate(device.layers):
            mid = 0.5 * (edges[i] + edges[i + 1])
            ax.text(mid, 0.985, layer.name, transform=trans, rotation=90,
                    ha="center", va="top", fontsize=label_fontsize,
                    color="0.35", clip_on=False, zorder=1)

    ax.set_xlim(edges[0], edges[-1])
    return edges[0], edges[-1]


def _fmt(value: float, spec: str, unit: str) -> str:
    return "—" if not math.isfinite(value) else f"{value:{spec}}{unit}"


def annotate_metrics(ax: Any, jv: Any, *, loc: str = "lower left",
                     fontsize: float = 8.5) -> None:
    """在轴内画 Jsc/Voc/FF/PCE 指标框；NaN 显示为 "—"。"""
    vertical, _, horizontal = loc.partition(" ")
    y, va = (0.96, "top") if vertical == "upper" else (0.05, "bottom")
    x, ha = {"left": (0.03, "left"),
             "right": (0.97, "right")}.get(horizontal or "left", (0.5, "center"))

    text = "\n".join((
        f"Jsc = {_fmt(jv.Jsc, '.3f', ' mA/cm²')}",
        f"Voc = {_fmt(jv.Voc, '.3f', ' V')}",
        f"FF  = {_fmt(jv.FF, '.3f', '')}",
        f"PCE = {_fmt(jv.PCE, '.2f', ' %')}",
    ))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, family="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75",
                      alpha=0.92))


def save(fig: Any, path: Any, *, dpi: int = 150) -> Path:
    """保存并关闭 Figure，父目录自动创建。"""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out
