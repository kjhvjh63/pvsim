"""参考参数兜底：真实数据不可用时，用参考指标合成"示意"J-V。

使用边界
--------
这是**兜底**，不是仿真结果：

* 只有 J-V 派生的图（``needs ⊆ {"jv"}``）能兜底，剖面/扫描图必须真实数据；
* 合成结果的 ``run_id`` 固定为 :data:`REFERENCE_RUN_ID`，``converged=False``，
  图上会打水印，:func:`is_reference` 可判定，避免被当成计算结果汇报。

合成模型
--------
单指数 J-V 形状 ``j(v) = (1 - exp(k(v/Voc - 1))) / (1 - exp(-k))``，
由目标填充因子 FF 反解形状因子 ``k``（二分），因此曲线严格经过
``(0, Jsc)`` 与 ``(Voc, 0)``，且 FF 与给定值一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..solver.simulator import FieldTable, JVData, SimResult, SolvePoint
from .registry import REGISTRY, MissingData, PlotConfig

REFERENCE_RUN_ID = "参考参数（示意）"


# ---------------------------------------------------------------------------
# 参考指标
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceMetrics:
    """一组参考指标（来自历史数据、文献或人工设定）。"""

    Jsc:   float = 22.0
    Voc:   float = 1.10
    FF:    float = 0.75
    label: str = "参考参数"

    def validate(self) -> None:
        if self.Jsc <= 0:
            raise ValueError(f"ReferenceMetrics.Jsc={self.Jsc} must be > 0")
        if self.Voc <= 0:
            raise ValueError(f"ReferenceMetrics.Voc={self.Voc} must be > 0")
        if not (0.25 < self.FF < 0.90):
            raise ValueError(
                f"ReferenceMetrics.FF={self.FF} 超出单指数形状可实现范围 (0.25, 0.90)"
            )


# ---------------------------------------------------------------------------
# 形状因子反解
# ---------------------------------------------------------------------------

def _fill_factor(k: float) -> float:
    """给定形状因子 k，返回归一化曲线的 FF（纯数值，无解析近似）。"""
    v = np.linspace(0.0, 1.0, 401)
    j = (1.0 - np.exp(k * (v - 1.0))) / (1.0 - math.exp(-k))
    return float(np.max(v * j))


def _k_for_ff(ff: float, iters: int = 80) -> float:
    """二分反解：FF 随 k 单调递增（k→0 为三角形，k→∞ 为矩形）。"""
    lo, hi = 0.5, 40.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _fill_factor(mid) < ff:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reference_jv(metrics: ReferenceMetrics, pin_mW_cm2: float,
                 n: int = 161) -> JVData:
    """由参考指标合成 J-V（``J`` 仍遵守内核符号：光照短路时 J < 0）。"""
    metrics.validate()
    if n < 5:
        raise ValueError("n 至少为 5")
    k = _k_for_ff(metrics.FF)
    V = np.linspace(0.0, metrics.Voc, n)
    j = (1.0 - np.exp(k * (V / metrics.Voc - 1.0))) / (1.0 - math.exp(-k))
    J = -metrics.Jsc * np.clip(j, 0.0, None)   # 交付电流 >0 → 原始 J < 0
    return JVData.from_points(V, J, pin_mW_cm2=pin_mW_cm2)


def reference_metrics_for(spec: Any) -> ReferenceMetrics:
    """按吸收层带隙粗估一组参考指标。

    只是"让图有东西可画"的占位值，**不得**当作计算结果引用；正式对标
    请显式传 ``ReferenceMetrics``。
    """
    absorbers = spec.device.absorber_layers
    eg = min((l.Eg_eV for l in absorbers), default=1.6)
    return ReferenceMetrics(
        Jsc=22.0, Voc=max(eg - 0.40, 0.4), FF=0.75,
        label=f"参考参数（按 Eg={eg:g} eV 粗估）",
    )


# ---------------------------------------------------------------------------
# 合成 SimResult
# ---------------------------------------------------------------------------

def reference_result(spec: Any,
                     metrics: Optional[ReferenceMetrics] = None) -> SimResult:
    """用参考参数造一个仅有 J-V 的 SimResult（场表为空）。"""
    m = metrics or reference_metrics_for(spec)
    jv = reference_jv(m, spec.optics.intensity_mW_cm2)
    points = tuple(
        SolvePoint(index=i, V=float(v), J_mA_cm2=float(j),
                   converged=False, residual=float("nan"), iterations=0)
        for i, (v, j) in enumerate(zip(jv.V, jv.J))
    )
    fields = FieldTable(
        x_nm=np.array([0.0, float(spec.device.total_thick_nm)]),
        data={},
    )
    return SimResult(run_id=REFERENCE_RUN_ID, spec=spec,
                     points=points, fields=fields, jv=jv)


def is_reference(result: Any) -> bool:
    return getattr(result, "run_id", None) == REFERENCE_RUN_ID


# ---------------------------------------------------------------------------
# 判定与出图
# ---------------------------------------------------------------------------

def needs_fallback(result: Any) -> bool:
    """真实数据是否"等于没有"：无 J-V、全 NaN、或 Jsc≈0（如内核 stub）。

    暗态（光强 0）扫描本就该是零电流，不触发兜底。
    """
    spec = getattr(result, "spec", None)
    if spec is not None and spec.optics.intensity_mW_cm2 <= 0:
        return False
    jv = getattr(result, "jv", None)
    if jv is None or np.asarray(jv.V).size < 2:
        return True
    if not np.any(np.isfinite(jv.J)):
        return True
    return not (math.isfinite(jv.Jsc) and abs(jv.Jsc) > 1e-9)


def supports(plot_id: str) -> bool:
    """该图能否用参考曲线替代（只有纯 J-V 派生的图可以）。"""
    return set(REGISTRY.get(plot_id).needs) <= {"jv"}


def render_reference(plot_id: str, target: Any,
                     metrics: Optional[ReferenceMetrics] = None,
                     cfg: Optional[PlotConfig] = None, ax: Any = None) -> Any:
    """用参考参数出图并打水印；``target`` 可以是 SimResult 或 ProblemSpec。"""
    if not supports(plot_id):
        raise MissingData(
            plot_id, "jv",
            f"图 {plot_id!r} 需要真实求解数据，不能用参考参数兜底",
        )
    spec = getattr(target, "spec", target)
    ref = reference_result(spec, metrics)
    ax = REGISTRY.render(plot_id, ref, cfg=cfg, ax=ax)
    _watermark(ax, ref)
    return ax


def _watermark(ax: Any, result: SimResult) -> None:
    if not is_reference(result):
        return
    ax.text(0.99, 0.97, f"{REFERENCE_RUN_ID} · 非仿真结果",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#b03030", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#b03030", lw=0.6, alpha=0.9))
