"""外部软件（SCAPS / Silvaco）J-V 参考曲线读取与量化对照。

用途
----
P2 外部对照需要把别的仿真器算出的 J-V 与本平台结果逐点比对。本模块只做
两件事：

1. :func:`load_reference` —— 把外部文件读成 :class:`JVData`；
2. :func:`compare` —— 用同一套指标（Jsc/Voc/FF/PCE + 曲线 RMSE）给出
   逐项偏差与通过/不通过判定。

符号约定（重要）
----------------
外部流程（SCAPS、Silvaco Atlas、文献）在短路处报**正**电流 ``Jsc > 0``；
本平台内核按阳极→阴极方向记电流，光照短路时 ``J < 0``。
:class:`JVData` 的 ``Jsc = -J(V=0)``，所以读出的参考曲线必须换成
**内核惯例（短路 J < 0）**，指标才会是正常正数。:func:`load_reference`
按 V=0 附近电流的符号自动归一，仿真与参考因此可以直接逐点相减。

支持格式
--------
* ``.log`` / ``.ssf``：Silvaco SSF，交给 :class:`SilvacoLogReader`（带单位换算）；
* 其他文本（``.iv`` / ``.dat`` / ``.txt`` / ``.csv``）：空白/逗号/分号分列，
  ``#`` 之后为注释；有表头时按列名挑电压/电流列，没表头就取前两列。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..solver.simulator import JVData
from .silvaco_log import SilvacoLogReader

_SSF_SUFFIXES = (".log", ".ssf")
_XLSX_SUFFIXES = (".xlsx", ".xlsm")
_SPLIT_RE = re.compile(r"[,\s;]+")


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def _parse_float(tok: str) -> Optional[float]:
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _pick_column(header: Sequence[str], hints: Sequence[str],
                 default: int) -> int:
    for i, name in enumerate(header):
        low = name.lower()
        if any(h in low for h in hints):
            return i
    return default


def read_silvaco_xy(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """读 TonyPlot ``EXPORTED XY-GRAPH DATA``；列序常是「电流在前、电压在后」。

    头部形如::

        EXPORTED  XY-GRAPH  DATA
        41  2  2
        Anode   Current
        Cathode Voltage
        <npoints 行数值>
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    if not lines or "EXPORTED" not in lines[0]:
        raise ValueError(f"{p}: 不是 EXPORTED XY-GRAPH DATA 格式")

    # 计数行：跳过空行后第一行是 "npoints ncols [ncurves]"
    idx = 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    counts = lines[idx].split()
    try:
        n_points, n_cols = int(counts[0]), int(counts[1])
    except (IndexError, ValueError):
        raise ValueError(f"{p}: 头部计数行无法解析：{lines[idx]!r}") from None
    idx += 1

    names: List[str] = []
    for _ in range(n_cols):
        if idx >= len(lines):
            raise ValueError(f"{p}: 列名行不足 {n_cols} 行")
        names.append(" ".join(lines[idx].split()))
        idx += 1

    rows: List[List[float]] = []
    while idx < len(lines) and len(rows) < n_points:
        toks = lines[idx].split()
        idx += 1
        if not toks:
            continue
        vals: List[float] = []
        for tok in toks[:n_cols]:
            val = _parse_float(tok)
            if val is None:
                vals = []
                break
            vals.append(val)
        if len(vals) == n_cols:
            rows.append(vals)
    if len(rows) < 2:
        raise ValueError(f"{p}: 数值行不足 2 行")

    arr = np.asarray(rows, dtype=float)
    header = tuple(names)
    vi = _pick_column(header, ("volt", "bias"), 1)
    ji = _pick_column(header, ("curr", "dens"), 0)
    if vi == ji:
        vi, ji = 1, 0
    return arr[:, vi], arr[:, ji], header


def read_xlsx_curve(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """读 Excel J-V：按表头找电压列与电流密度列（A 列会要求面积换算）。"""
    try:
        import openpyxl
    except ImportError as exc:                      # pragma: no cover - 依赖缺失
        raise RuntimeError("读取 .xlsx 需要 openpyxl（pip install openpyxl）") from exc

    p = Path(path)
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = None
        for row in it:
            if row and any(isinstance(c, str) for c in row):
                header = tuple(str(c) if c is not None else "" for c in row)
                break
        if header is None:
            raise ValueError(f"{p}: 没找到表头行")

        vi = _pick_column(header, ("voltage",), -1)
        # 优先电流密度（mA/cm²），其次电流（A，需要面积，直接报错更诚实）
        ji = _pick_column(header, ("current density",), -1)
        if ji < 0:
            ji = _pick_column(header, ("current",), -1)
            if ji >= 0:
                raise ValueError(
                    f"{p}: 只有电流(A)列，缺电流密度；请在 header 含 "
                    f"'Current Density (mA/cm^2)'，或用 compare_reference --area 换算")
        if vi < 0 or ji < 0:
            raise ValueError(f"{p}: 表头里找不到电压/电流列：{header}")

        rows = []
        for row in it:
            if row is None or len(row) <= max(vi, ji):
                continue
            v, j = row[vi], row[ji]
            if isinstance(v, (int, float)) and isinstance(j, (int, float)):
                rows.append((float(v), float(j)))
    finally:
        wb.close()

    if len(rows) < 2:
        raise ValueError(f"{p}: 数值行不足 2 行")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], header


def read_text_curve(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """读两列 (V, J) 文本；返回 ``(V, J, 表头)``，表头可能为空元组。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    header_lines: List[str] = []
    rows: List[List[float]] = []

    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        tokens = [t for t in _SPLIT_RE.split(body) if t]
        vals = [_parse_float(t) for t in tokens]
        if any(v is None for v in vals):
            if not rows:
                header_lines.append(body)   # 数值行之前的非数值行当表头候选
            continue
        rows.append([float(v) for v in vals])

    if len(rows) < 2:
        raise ValueError(f"{p}: 至少需要 2 行数值 (V, J)，实际 {len(rows)} 行")

    # 数据列宽取众数；表头必须按同样的列宽切开（单位里的空格不算分列）
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    rows = [r for r in rows if len(r) >= width]
    header: Tuple[str, ...] = ()
    for body in header_lines:
        for pattern in (r"[,\t;]+", None):
            toks = ([t.strip() for t in re.split(pattern, body) if t.strip()]
                    if pattern else body.split())
            if len(toks) == width:
                header = tuple(toks)
                break
        if header:
            break

    vi = _pick_column(header, ("volt", "bias"), 0) if header else 0
    ji = _pick_column(header, ("curr", "dens", " j", "j("), 1) if header else 1
    if header and (vi == ji or max(vi, ji) >= width):
        vi, ji = 0, 1                          # 表头不可信时退回前两列
    arr = np.asarray(rows, dtype=float)
    return arr[:, vi], arr[:, ji], header


def _short_circuit_current(V: np.ndarray, J: np.ndarray) -> float:
    """V=0 附近的电流；区间不覆盖 0 时取端点中离 0 最近的那个。"""
    order = np.argsort(V)
    Vs, Js = V[order], J[order]
    if Vs[0] <= 0.0 <= Vs[-1]:
        return float(np.interp(0.0, Vs, Js))
    return float(Js[0] if abs(Vs[0]) <= abs(Vs[-1]) else Js[-1])


def load_reference(path: Path, *, pin_mW_cm2: float = 100.0) -> JVData:
    """读外部 J-V 文件，返回交付电流惯例（短路处 J > 0）的 :class:`JVData`。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"参考曲线不存在：{p}")

    if p.suffix.lower() in _SSF_SUFFIXES:
        jv = SilvacoLogReader().read_jv(p, pin_mW_cm2=pin_mW_cm2)
        # SSF 可能是交付惯例（短路为正），统一翻成内核惯例（短路为负）
        if jv.Jsc < 0:
            jv = JVData.from_points(jv.V, -jv.J, pin_mW_cm2)
        return jv

    suffix = p.suffix.lower()
    if suffix in _XLSX_SUFFIXES:
        V, J, _ = read_xlsx_curve(p)
    else:
        head = p.read_text(encoding="utf-8", errors="replace").lstrip()
        if head.startswith("EXPORTED"):
            V, J, _ = read_silvaco_xy(p)
        else:
            V, J, _ = read_text_curve(p)
    if _short_circuit_current(V, J) > 0:
        J = -J                                   # 交付惯例 → 内核惯例
    return JVData.from_points(V, J, pin_mW_cm2)


# ---------------------------------------------------------------------------
# 对照
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricDelta:
    """单项指标对照：两侧取值、绝对/相对偏差与容差判定。"""

    name:    str
    sim:     float
    ref:     float
    delta:   float
    rel_pct: float
    tol:     str
    ok:      bool

    def line(self) -> str:
        sim = "nan" if not math.isfinite(self.sim) else f"{self.sim:.4f}"
        ref = "nan" if not math.isfinite(self.ref) else f"{self.ref:.4f}"
        rel = "—" if not math.isfinite(self.rel_pct) else f"{self.rel_pct:+.2f}%"
        mark = "✓" if self.ok else "✗"
        return f"  {mark} {self.name:<4} 仿真={sim:>9}  参考={ref:>9}  " \
               f"相对={rel:>8}  容差={self.tol}"


@dataclass(frozen=True)
class ComparisonReport:
    """一次对照的完整结果；``lines()`` 给 CLI/日志用的文本。"""

    metrics:   Tuple[MetricDelta, ...]
    rmse_mA_cm2: float
    max_abs_mA_cm2: float
    n_points:   int
    v_window:   Tuple[float, float]
    rmse_tol:   float

    @property
    def passed(self) -> bool:
        return (all(m.ok for m in self.metrics)
                and self.rmse_mA_cm2 <= self.rmse_tol)

    def lines(self) -> List[str]:
        out = [f"对照区间 {self.v_window[0]:g} → {self.v_window[1]:g} V，"
               f"公共点 {self.n_points} 个"]
        out.extend(m.line() for m in self.metrics)
        mark = "✓" if self.rmse_mA_cm2 <= self.rmse_tol else "✗"
        out.append(f"  {mark} RMSE 仿真-参考 = {self.rmse_mA_cm2:.4f} mA/cm² "
                   f"（容差 {self.rmse_tol:g}），最大偏差 {self.max_abs_mA_cm2:.4f}")
        out.append(f"结论：{'通过' if self.passed else '未通过'}")
        return out


def _metric(name: str, sim: float, ref: float, *, rel_tol_pct: Optional[float] = None,
            abs_tol: Optional[float] = None, unit: str = "") -> MetricDelta:
    delta = sim - ref
    finite = math.isfinite(delta)
    rel = (delta / abs(ref) * 100.0) if finite and ref != 0.0 else float("nan")
    if abs_tol is not None:
        ok = math.isfinite(delta) and abs(delta) <= abs_tol
        tol = f"±{abs_tol:g}{unit}"
    else:
        ok = math.isfinite(rel) and abs(rel) <= float(rel_tol_pct)
        tol = f"±{rel_tol_pct:g}%"
    return MetricDelta(name=name, sim=sim, ref=ref, delta=delta,
                       rel_pct=rel, tol=tol, ok=ok)


def compare(sim_jv: JVData, ref_jv: JVData, *,
            jsc_tol_pct: float = 5.0,
            voc_tol_mV: float = 30.0,
            ff_tol_abs: float = 0.03,
            pce_tol_pct: float = 8.0,
            rmse_tol: float = 1.0) -> ComparisonReport:
    """按同一套指标比较两条 J-V 曲线。

    默认容差是"同结构、同材料参数"下可接受的上限；收紧/放宽由调用方给。
    曲线误差只在两条曲线电压重叠区里算，并截到两者较小的 Voc——正向
    指数段（V > Voc）的电流量级会放大无关差异，不参与 RMS 统计。
    """
    Vs = np.asarray(sim_jv.V, dtype=float)
    Js = np.asarray(sim_jv.J, dtype=float)           # 与 ref_jv 同为内核惯例
    Vr = np.asarray(ref_jv.V, dtype=float)
    Jr = np.asarray(ref_jv.J, dtype=float)

    lo = max(float(Vs.min()), float(Vr.min()))
    hi = min(float(Vs.max()), float(Vr.max()))
    if hi <= lo:
        raise ValueError(
            f"两条曲线没有重叠的电压区间：仿真 [{Vs.min():g}, {Vs.max():g}]，"
            f"参考 [{Vr.min():g}, {Vr.max():g}]")
    vocs = [v for v in (sim_jv.Voc, ref_jv.Voc) if math.isfinite(v) and v > 0]
    if vocs:
        hi = min(hi, min(vocs))                      # 只比功率相关区

    mask = (Vs >= lo) & (Vs <= hi)
    if int(mask.sum()) < 2:
        raise ValueError(f"重叠区内仿真点不足 2 个（{int(mask.sum())}）")
    ref_on_sim = np.interp(Vs[mask], Vr, Jr)
    diff = Js[mask] - ref_on_sim

    metrics = (
        _metric("Jsc", sim_jv.Jsc, ref_jv.Jsc, rel_tol_pct=jsc_tol_pct),
        _metric("Voc", sim_jv.Voc, ref_jv.Voc, abs_tol=voc_tol_mV / 1000.0, unit="V"),
        _metric("FF", sim_jv.FF, ref_jv.FF, abs_tol=ff_tol_abs),
        _metric("PCE", sim_jv.PCE, ref_jv.PCE, rel_tol_pct=pce_tol_pct),
    )
    return ComparisonReport(
        metrics=metrics,
        rmse_mA_cm2=float(np.sqrt(np.mean(diff ** 2))),
        max_abs_mA_cm2=float(np.max(np.abs(diff))),
        n_points=int(mask.sum()),
        v_window=(lo, hi),
        rmse_tol=rmse_tol,
    )
