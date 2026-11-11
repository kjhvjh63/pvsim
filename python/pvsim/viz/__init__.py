"""viz — 绘图引擎（注册表模式）。

用法::

    from pvsim.viz import REGISTRY, PlotConfig
    ax = REGISTRY.render("cell_jv_curve", result)
    fig = REGISTRY.figure(result, ["cell_jv_curve", "cell_pv_curve"])

扫描汇总（多个 SimResult）先包成 ``ScanSet``::

    from pvsim.viz import ScanSet
    scan = ScanSet.from_plan(plan, Simulator().run_plan(plan, base_spec))
    REGISTRY.figure(scan, scan_summary_ids(), cols=2)

论文对照先包成 ``ThesisSet``::

    from pvsim.viz import ThesisSet, load_curve
    ts = ThesisSet(result, [load_curve("refs/paper_a.dat", label="Paper A")])
    REGISTRY.figure(ts, ["compare_jv", "compare_pce_bar"])

真实数据不可用时用 ``fallback`` 兜底（图上带"参考参数"水印）::

    from pvsim.viz import fallback
    if fallback.needs_fallback(result):
        fallback.render_reference("cell_jv_curve", result)

import 本包时 ``viz.plots`` 会自动把已实现的图注册进 ``REGISTRY``。
"""

from .registry import REGISTRY, MissingData, PlotConfig, PlotItem, PlotRegistry
from . import axes_style
from . import plots  # noqa: F401  导入即注册绘图函数
from . import fallback

from .plots.scans import ScanSet, scan_summary_ids
from .plots.thesis_compare import (
    LiteratureEntry,
    ThesisSet,
    load_curve,
    load_dat,
    load_xlsx,
)

__all__ = [
    "REGISTRY", "MissingData", "PlotConfig", "PlotItem", "PlotRegistry",
    "ScanSet", "scan_summary_ids",
    "ThesisSet", "LiteratureEntry", "load_curve", "load_dat", "load_xlsx",
    "axes_style", "fallback", "plots",
]
