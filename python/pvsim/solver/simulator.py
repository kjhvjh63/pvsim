"""Simulator — 把 ProblemSpec / SweepPlan 串成一次完整求解。

三层职责划分：

* ``ProblemSpec``：一次求解的全部输入（本次研究的一个"点"）；
* ``SweepPlan``：一次研究的展开规则，产出若干 ``PlanStep``；
* ``Simulator``：只负责"给定一个 spec，调内核，装出 SimResult"，
  外加 ``run_plan`` 把 plan 的每个 step 依次跑一遍。

当前实现按偏压网格逐点调 ``pvk_solve_dc``（而不是一次性
``pvk_sweep_dc``），原因是：

1. 内核的 ctx 只保留最后一次求解的场量，逐点调才能在 Python 侧
   累积"点 × 网格"的二维场表；
2. 每个点都能立即回调进度钩子（GUI 进度条数据源）；
3. 等内核内部 continuation 落地后，这里可以无缝切换到
   ``pvk_sweep_dc`` 而不影响上层 API。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..kernel import structs as st
from ..param.schema import ProblemSpec, SpecError, SweepPlan

# 与 pvk.h 的 pvk_log_level 对齐
LOG_DEBUG = 0
LOG_INFO = 1
LOG_WARN = 2
LOG_ERROR = 3


# ---------------------------------------------------------------------------
# 进度钩子
# ---------------------------------------------------------------------------

class ProgressHook(Protocol):
    """GUI / CLI 观察求解进度的接口。两个方法都按需实现即可。"""

    def on_point(self, index: int, total: int, point: "SolvePoint") -> None:
        """每完成一个偏压点回调一次（index 从 0 起）。"""

    def on_log(self, level: int, message: str) -> None:
        """日志/警告回调。"""


def _emit(hook: Optional[ProgressHook], method: str, *args: Any) -> None:
    """安全地调用钩子方法——钩子可以只实现其中一部分。"""
    if hook is None:
        return
    fn = getattr(hook, method, None)
    if fn is not None:
        fn(*args)


def _emit_log(hook: Optional[ProgressHook], level: int, message: str) -> None:
    _emit(hook, "on_log", level, message)


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolvePoint:
    """单个偏压点的求解摘要（不持有 ctypes 对象）。"""

    index:      int
    V:          float
    J_mA_cm2:   float
    converged:  bool
    residual:   float
    iterations: int

    def __str__(self) -> str:
        return (f"V={self.V:.4f} V, J={self.J_mA_cm2:.4g} mA/cm², "
                f"conv={self.converged}, it={self.iterations}")


@dataclass(frozen=True)
class FieldTable:
    """场量表：``data[name]`` 形状 ``(n_points, n_node)``。

    ``x_nm`` 是网格坐标（nm）。数组由 Simulator 填好后不再修改；
    dataclass 本身 frozen，但内部是 ndarray —— 调用方按只读约定使用。
    """

    x_nm: np.ndarray
    data: Mapping[str, np.ndarray]

    @property
    def n_points(self) -> int:
        if not self.data:
            return 0
        return int(next(iter(self.data.values())).shape[0])

    @property
    def n_node(self) -> int:
        return int(self.x_nm.size)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.data.keys())

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    def get(self, name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self.data.get(name, default)

    def at(self, point_index: int) -> Dict[str, np.ndarray]:
        """取某个偏压点上的全部剖面。"""
        return {name: arr[point_index, :] for name, arr in self.data.items()}


@dataclass(frozen=True)
class JVData:
    """J-V 曲线 + 四个常用指标。

    符号约定（与漂移扩散代码一致）：``J`` 是 anode→cathode 方向的电流
    密度，光照下短路时 ``J < 0``，所以对外报的 ``Jsc = -J(V=0) > 0``，
    功率 ``P = -J·V``。

    过零缺失（例如全零场、或者扫描没覆盖到 Voc）时 ``Voc = nan``；
    ``Voc`` 或 ``Jsc`` 非正时 ``FF = nan``；``Pmax <= 0`` 时
    ``PCE = nan``。绝不拿扫描终点冒充 Voc。
    """

    V:            np.ndarray
    J:            np.ndarray
    Jsc:          float
    Voc:          float
    FF:           float
    PCE:          float
    Pmax_mW_cm2:  float
    pin_mW_cm2:   float

    @classmethod
    def from_points(cls, V: Sequence[float], J: Sequence[float],
                    pin_mW_cm2: float) -> "JVData":
        Varr = np.asarray(V, dtype=float).ravel()
        Jarr = np.asarray(J, dtype=float).ravel()
        nan = float("nan")
        pin = float(pin_mW_cm2)

        if Varr.size == 0 or Varr.size != Jarr.size:
            return cls(Varr, Jarr, nan, nan, nan, nan, nan, pin)

        order = np.argsort(Varr)
        Vs, Js = Varr[order], Jarr[order]

        # Jsc：V=0 处线性插值，取 −J
        if Vs[0] <= 0.0 <= Vs[-1]:
            j0 = float(np.interp(0.0, Vs, Js))
        else:
            j0 = float(Js[int(np.argmin(np.abs(Vs)))])
        Jsc = 0.0 if j0 == 0.0 else -j0   # 避免 -0.0 影响显示

        # Voc：J 由负穿越到正的第一个交点（线性插值）
        Voc = nan
        for i in range(Vs.size - 1):
            if Js[i] < 0.0 <= Js[i + 1]:
                d = Js[i + 1] - Js[i]
                t = (-Js[i] / d) if d != 0.0 else 0.5
                Voc = float(Vs[i] + t * (Vs[i + 1] - Vs[i]))
                break

        # Pmax：只在 V>0 区取 -J·V 的最大值
        Pmax = nan
        mask = Vs > 0.0
        if np.any(mask):
            Pmax = float(np.max(-Js[mask] * Vs[mask]))

        FF = nan
        if (math.isfinite(Voc) and Voc > 0.0
                and Jsc > 0.0
                and math.isfinite(Pmax) and Pmax > 0.0):
            FF = Pmax / (Jsc * Voc)

        PCE = nan
        if pin > 0.0 and math.isfinite(Pmax) and Pmax > 0.0:
            PCE = Pmax / pin * 100.0

        return cls(Vs, Js, float(Jsc), float(Voc), float(FF),
                   float(PCE), float(Pmax), pin)

    def summary(self) -> str:
        def _f(x: float, unit: str = "") -> str:
            return "—" if not math.isfinite(x) else f"{x:.4g}{unit}"
        return (f"Jsc={_f(self.Jsc, ' mA/cm²')}, Voc={_f(self.Voc, ' V')}, "
                f"FF={_f(self.FF)}, PCE={_f(self.PCE, '%')}")


@dataclass(frozen=True)
class SimResult:
    """一次求解的完整产物（viz / dataio / gui 只认这个类型）。"""

    run_id:  str
    spec:    ProblemSpec
    points:  Tuple[SolvePoint, ...]
    fields:  FieldTable
    jv:      JVData

    @property
    def converged(self) -> bool:
        return all(p.converged for p in self.points)

    @property
    def V(self) -> np.ndarray:
        return np.asarray([p.V for p in self.points], dtype=float)

    @property
    def J(self) -> np.ndarray:
        return np.asarray([p.J_mA_cm2 for p in self.points], dtype=float)

    def summary(self) -> str:
        n_conv = sum(1 for p in self.points if p.converged)
        return (f"{self.run_id}: {len(self.points)} 点（收敛 {n_conv}） "
                f"{self.jv.summary()}")


@dataclass(frozen=True)
class PlanRun:
    """扫描计划里的一步 + 它的求解结果。"""

    label:  str
    result: SimResult


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """把 ``ProblemSpec`` 交给 C 内核跑完，装成 ``SimResult``。

    参数
    ----
    lib : PvkLib | None
        已加载的内核绑定对象。``None`` 时首次求解自动加载
        （走 ``PVK_LIB`` 环境变量或源码树 ``csrc/build``）。

    线程模型
    --------
    每次 ``run`` 都新建并销毁一个 ctx，实例本身无可变状态，因此可以
    在 GUI 工作线程里安全使用（同一 ctx 不跨线程）。
    """

    def __init__(self, lib: Optional[Any] = None):
        self._lib = lib

    @property
    def lib(self):
        """惰性加载 libpvk：只有真正要算的时候才碰 DLL。"""
        if self._lib is None:
            from ..kernel.abi import PvkLib
            self._lib = PvkLib()
        return self._lib

    # ----- 单次求解 ------------------------------------------------------
    def run(self, spec: ProblemSpec,
            hook: Optional[ProgressHook] = None,
            run_id: Optional[str] = None) -> SimResult:
        """按 ``spec.sweep`` 的偏压网格逐点求解，返回 ``SimResult``。

        致命配置错误在碰内核之前就以 ``SpecError`` 抛出；warning 级
        问题通过 ``hook.on_log(LOG_WARN, ...)`` 上报后继续。
        """
        fatal, warnings = spec.validate_notes()
        if fatal:
            raise SpecError("ProblemSpec 未通过校验", details=fatal + warnings)
        for w in warnings:
            _emit_log(hook, LOG_WARN, w)

        # ctypes 结构里的字符串挂在 Layer 对象上，必须活到本次 run 结束
        dev, _layers_keepalive = st.device_from_spec(spec.device)
        opt = st.optics_from_spec(spec.optics)
        sopts = st.solver_opts_from_spec(spec.sweep.solver_opts)

        n_pts = max(1, int(spec.sweep.n_points))
        v_grid = np.linspace(float(spec.sweep.v_start),
                             float(spec.sweep.v_stop), n_pts)
        names = tuple(spec.output.fields)

        ctx = self.lib.create()
        try:
            self.lib.set_solver_opts(ctx, sopts)
            self.lib.build_mesh(ctx, dev)
            n_node = self.lib.node_count(ctx)
            if n_node < 3:
                raise RuntimeError(f"内核建出的网格只有 {n_node} 个节点")

            x_nm = np.asarray(self.lib.get_x(ctx, n_node), dtype=float) * 1e7
            arrays = {name: np.empty((n_pts, n_node), dtype=float)
                      for name in names}

            points = []
            for k, v in enumerate(v_grid):
                bias = st.bias_from_spec(spec.sweep, float(v))
                info = self.lib.solve_dc(ctx, dev, opt, bias)

                for name in names:
                    arrays[name][k, :] = self.lib.get_field(
                        ctx, st.FIELD_ID[name], n_node)

                point = SolvePoint(
                    index=k,
                    V=float(v),
                    J_mA_cm2=float(info.J_total_mA_cm2),
                    converged=bool(info.converged),
                    residual=float(info.residual),
                    iterations=int(info.iterations),
                )
                points.append(point)
                _emit(hook, "on_point", k, n_pts, point)
        finally:
            self.lib.destroy(ctx)

        jv = JVData.from_points(v_grid, [p.J_mA_cm2 for p in points],
                                pin_mW_cm2=spec.optics.intensity_mW_cm2)
        rid = run_id or f"{spec.output.run_name}_{datetime.now():%Y%m%d_%H%M%S}"
        result = SimResult(
            run_id=rid,
            spec=spec,
            points=tuple(points),
            fields=FieldTable(x_nm=x_nm, data=arrays),
            jv=jv,
        )
        _emit_log(hook, LOG_INFO, result.summary())
        if not result.converged:
            _emit_log(hook, LOG_WARN, f"{rid}: 有偏压点未收敛")
        return result

    # ----- 扫描计划 ------------------------------------------------------
    def run_plan(self, plan: SweepPlan, base_spec: ProblemSpec,
                 hook: Optional[ProgressHook] = None) -> Tuple[PlanRun, ...]:
        """展开 ``plan`` 并逐 step 求解，返回与 step 一一对应的结果。"""
        steps = plan.expand(base_spec)
        out = []
        for i, step in enumerate(steps):
            _emit_log(hook, LOG_INFO, f"[{i + 1}/{len(steps)}] {step.label}")
            out.append(PlanRun(label=step.label, result=self.run(step.spec, hook=hook)))
        return tuple(out)
