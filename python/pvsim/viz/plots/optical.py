"""光学类绘图：光学产生率剖面。

吸收谱 / 光学常数 / 多程反射等依赖 n,k 数据的图，等 optics 层的
``pvk_optics_compute`` 接上以后再补；当前只覆盖求解结果里已有的 G_OPT。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..axes_style import COLORS, layer_span
from ..registry import REGISTRY, MissingData, PlotConfig

if TYPE_CHECKING:
    from ..solver.simulator import SimResult

GROUP = "光学"


@REGISTRY.register(id="cell_gen_profile", group=GROUP, needs=("optics",),
                   title="光学产生率",
                   description="G_opt(x) 剖面：入射光在器件内的吸收分布")
def plot_gen_profile(result: "SimResult", cfg: PlotConfig, ax: Any) -> None:
    if "G_OPT" not in result.fields.names:
        raise MissingData(
            "cell_gen_profile", "optics",
            f"结果缺少 G_OPT；当前输出：{', '.join(result.fields.names) or '(空)'}。"
            f"请在 OutputSpec.fields 中加入 G_OPT 后重跑。",
        )

    idx = cfg.resolve_point(result.fields.n_points)
    x = np.asarray(result.fields.x_nm, dtype=float)
    G = np.maximum(np.asarray(result.fields["G_OPT"][idx, :], dtype=float), 1e-30)

    ax.plot(x, G, color=COLORS["accent"])
    ax.fill_between(x, G, 1e-30, color=COLORS["accent"], alpha=0.15, lw=0)
    ax.set_yscale("log")
    layer_span(ax, result.spec.device)

    V = float(result.V[idx]) if result.points else float("nan")
    ax.set_xlabel("位置 x (nm)")
    ax.set_ylabel("光学产生率 G (cm$^{-3}$ s$^{-1}$)")
    ax.set_title(f"光学产生率  @ V = {V:.3f} V")
