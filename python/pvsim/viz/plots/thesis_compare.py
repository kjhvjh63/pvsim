"""论文数据对照：把文献/实测曲线叠加到本次仿真结果上。

数据格式
--------
* ``.dat/.txt/.csv``：空白、逗号或分号分列，``#`` 之后为注释，非数值行
  （表头）自动跳过；只取前两列 ``(V, J)``。
* ``.xlsx/.xlsm``：用 openpyxl 读，默认第一个工作表、前两列。

``J`` 一律是**交付电流** (mA/cm²)，与仿真同号：光照下 ``J > 0``。

    ts = ThesisSet(result, [load_curve("refs/paper_a.dat", label="Paper A")])
    REGISTRY.savefig(ts, ["compare_jv", "compare_pce_bar"], "out/compare.png")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from ..axes_style import COLORS
from ..registry import REGISTRY, MissingData, PlotConfig

if TYPE_CHECKING:
    from ..solver.simulator import SimResult

GROUP = "论文对照"

_XLSX_SUFFIXES = (".xlsx", ".xlsm")
_SPLIT_RE = re.compile(r"[,\s;]+")


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def load_dat(path: Any) -> Tuple[np.ndarray, np.ndarray]:
    """读文本两列 (V, J)；跳过注释与表头。"""
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        parts = [s for s in _SPLIT_RE.split(body) if s]
        try:
            vals = [float(s) for s in parts]
        except ValueError:
            continue                      # 表头等非数值行
        if len(vals) >= 2:
            rows.append(vals[:2])

    if len(rows) < 2:
        raise ValueError(f"{p}: 至少需要 2 行数值 (V, J)，实际 {len(rows)} 行")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def load_xlsx(path: Any, sheet: Any = 0,
              col_v: int = 0, col_j: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """读 Excel 两列 (V, J)；``sheet`` 可为索引或表名。"""
    try:
        import openpyxl
    except ImportError as exc:            # pragma: no cover - 依赖缺失分支
        raise RuntimeError("读取 .xlsx 需要 openpyxl（pip install openpyxl）") from exc

    p = Path(path)
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if isinstance(sheet, str) else wb.worksheets[int(sheet)]
        need = max(col_v, col_j)
        rows = []
        for row in ws.iter_rows(values_only=True):
            if row is None or len(row) <= need:
                continue
            a, b = row[col_v], row[col_j]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                rows.append((float(a), float(b)))
    finally:
        wb.close()

    if len(rows) < 2:
        raise ValueError(f"{p}: 至少需要 2 行数值 (V, J)，实际 {len(rows)} 行")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def load_curve(path: Any, label: Optional[str] = None, *,
               sheet: Any = 0) -> "LiteratureEntry":
    """按后缀自动选择解析器，装成一条文献曲线。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"对照数据不存在: {p}")
    if p.suffix.lower() in _XLSX_SUFFIXES:
        v, j = load_xlsx(p, sheet=sheet)
    else:
        v, j = load_dat(p)
    return LiteratureEntry(label=label or p.stem, jv=(v, j), source=str(p))


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiteratureEntry:
    """一条文献数据：至少要有 (V, J) 曲线或 PCE 指标之一。"""

    label:   str
    jv:      Optional[Tuple[np.ndarray, np.ndarray]] = None
    Jsc:     Optional[float] = None
    Voc:     Optional[float] = None
    FF:      Optional[float] = None
    PCE:     Optional[float] = None
    source:  str = ""

    def __post_init__(self):
        if not self.label:
            raise ValueError("LiteratureEntry.label 不能为空")
        if self.jv is not None:
            v, j = (np.asarray(a, dtype=float).ravel() for a in self.jv)
            if v.size != j.size:
                raise ValueError(
                    f"{self.label}: V({v.size}) 与 J({j.size}) 长度不一致"
                )
            if v.size < 2:
                raise ValueError(f"{self.label}: 曲线至少需要 2 个点")
            object.__setattr__(self, "jv", (v, j))
        if self.jv is None and self.PCE is None:
            raise ValueError(
                f"{self.label}: jv 与 PCE 至少要有一个，否则对照图无从画起"
            )

    @property
    def legend(self) -> str:
        """图例文字；来源只显示文件名，避免长路径撑爆图例。"""
        if not self.source:
            return self.label
        return f"{self.label} ({Path(self.source).name})"


@dataclass(frozen=True)
class ThesisSet:
    """仿真结果 + 若干文献条目。"""

    result:  "SimResult"
    entries: Tuple[LiteratureEntry, ...]
    title:   str = ""

    def __post_init__(self):
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("ThesisSet.entries 不能为空")

    def __len__(self) -> int:
        return len(self.entries)


def _require(result: Any) -> ThesisSet:
    if not isinstance(result, ThesisSet):
        raise MissingData(
            "<compare>", "compare",
            f"论文对照图需要 ThesisSet（仿真结果 + 文献条目），"
            f"当前传入 {type(result).__name__}",
        )
    return result


# ---------------------------------------------------------------------------
# 对照图
# ---------------------------------------------------------------------------

@REGISTRY.register(id="compare_jv", group=GROUP, needs=("compare",),
                   title="J-V 论文对照",
                   description="本文仿真 J-V 与文献/实测数据叠画")
def plot_compare_jv(result: Any, cfg: PlotConfig, ax: Any) -> None:
    ts = _require(result)
    jv = ts.result.jv
    ax.plot(jv.V, -jv.J, "-", lw=1.8, color=COLORS["main"],
            label=f"本文仿真（{ts.result.run_id}）")

    for i, entry in enumerate(ts.entries):
        if entry.jv is None:
            continue
        v, j = entry.jv
        color = f"C{(i + 1) % 10}"
        ax.plot(v, j, "-o", lw=1.0, ms=3.2, color=color,
                mfc="white", mew=0.9, label=entry.legend)

    ax.axhline(0.0, color=COLORS["muted"], lw=0.8, zorder=0)
    ax.set_xlabel("电压 V (V)")
    ax.set_ylabel("电流密度 J (mA/cm²)")
    if cfg.annotate:
        ax.legend(loc="best", frameon=False, fontsize=8)


@REGISTRY.register(id="compare_pce_bar", group=GROUP, needs=("compare",),
                   title="PCE 对照柱状图",
                   description="本文仿真效率与文献报道效率并排比较")
def plot_compare_pce_bar(result: Any, cfg: PlotConfig, ax: Any) -> None:
    ts = _require(result)
    names = ["本文仿真"] + [e.label for e in ts.entries]
    pces = [ts.result.jv.PCE] + [
        e.PCE if e.PCE is not None else float("nan") for e in ts.entries
    ]

    heights = [p if np.isfinite(p) else 0.0 for p in pces]
    colors = [COLORS["main"]] + [COLORS["muted"]] * len(ts.entries)
    positions = np.arange(len(names))
    ax.bar(positions, heights, color=colors, width=0.62,
           edgecolor="0.3", lw=0.7, zorder=2)

    ymax = max(heights + [1.0]) * 1.18
    for x, value, height in zip(positions, pces, heights):
        label = f"{value:.2f}" if np.isfinite(value) else "—"
        # 0 高度（缺指标）也要能看见标注，统一抬高一点
        ax.text(x, height + 0.015 * ymax, label, ha="center", va="bottom",
                fontsize=8, color="0.2", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("效率 PCE (%)")
    ax.set_ylim(0, ymax)
