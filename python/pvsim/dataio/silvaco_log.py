"""SSF/.log 读写（Silvaco Atlas 兼容）。

SSF（Silvaco Standard Format）的 .log 文件由若干 ``<Hn>`` 头记录和 ``<L>``
数据行组成，典型结构::

    <H1> PVK Simulator
    <H2> SSF 2.0
    <H3> run_id=mapbi3_ref_20261111_155333
    <H4> Anode Voltage (V)  Anode Current (mA/cm2)  Power (mW/cm2)
    <L>  0.000000e+00  -2.200000e+01  0.000000e+00

读侧刻意做宽容：

* 列名从 ``<H4>`` 推断，优先按"两个以上空格/制表符"切分（能保住
  ``Anode Voltage`` 这类多词列名），切不动再退回单空格；
* 单位从列名末尾的 ``(...)`` 提取，读回时据此把 A 换算成 mA/cm²；
  **单位缺失时按 Silvaco 约定视为 A**（Atlas 的 Anode Current 就是 A），
  换算需要 ``area_cm2``；
* 兼容 Fortran 的 ``D`` 指数（``1.0D+01``）、逗号分隔、无 ``<L>`` 前缀的纯数值行；
* 多个 ``<H4>`` 数据块全部保留；:meth:`SilvacoLogReader.read` 默认取最后一块
  （一次 J-V 扫描通常只有一块，多块一般来自重复 ``log`` 语句）。

写侧输出三列：电压、电流、功率，单位写在列名里，保证 round-trip 无损。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..solver.simulator import JVData

_HEADER_RE = re.compile(r"^<\s*([A-Za-z]\d?)\s*>\s*(.*)$")
_UNIT_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")
_NUM_SPLIT_RE = re.compile(r"[,\s]+")

DEFAULT_COLUMNS: Tuple[str, ...] = (
    "Anode Voltage (V)",
    "Anode Current (mA/cm2)",
    "Power (mW/cm2)",
)


# ---------------------------------------------------------------------------
# 表格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogTable:
    """一个 SSF 数据块：列名、单位、数值矩阵（列 → 1-D 数组）。"""

    columns: Tuple[str, ...]
    data:    Mapping[str, np.ndarray]
    units:   Mapping[str, str] = field(default_factory=dict)
    meta:    Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.columns, tuple):
            object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise ValueError("LogTable.columns 不能为空")
        lengths = {name: np.asarray(arr).size for name, arr in self.data.items()}
        if not lengths:
            raise ValueError("LogTable.data 不能为空")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"LogTable 各列长度不一致：{lengths}")

    @property
    def n_rows(self) -> int:
        return int(next(iter(self.data.values())).size)

    def __getitem__(self, name: str) -> np.ndarray:
        key = self.find(name)
        if key is None:
            raise KeyError(f"没有列 {name!r}；可用：{self.columns}")
        return np.asarray(self.data[key], dtype=float)

    def find(self, *keywords: str) -> Optional[str]:
        """自动列识别：先精确匹配（不区分大小写），再按子串包含匹配。"""
        exact = {c.lower(): c for c in self.columns}
        for kw in keywords:
            if kw.lower() in exact:
                return exact[kw.lower()]
        for kw in keywords:
            low = kw.lower()
            for col in self.columns:
                if low in col.lower():
                    return col
        return None

    def unit_of(self, column: str) -> str:
        return self.units.get(column, "")


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------

class SilvacoLogWriter:
    """把 :class:`JVData` 输出成 SSF 2.0 文本（可直接被 DeckBuild 打开）。"""

    def __init__(self, columns: Sequence[str] = DEFAULT_COLUMNS,
                 fmt: str = "{:.6e}"):
        self._columns = tuple(columns)
        if len(self._columns) != 3:
            raise ValueError("当前只支持 V/J/P 三列输出")
        self._fmt = fmt

    def dumps(self, jv: JVData, *, title: str = "PVK Simulator",
              run_id: str = "", meta: Optional[Mapping[str, str]] = None) -> str:
        lines = [
            f"<H1> {title}",
            "<H2> SSF 2.0",
            f"<H3> run_id={run_id or '-'}",
        ]
        for key, value in (meta or {}).items():
            lines.append(f"<H3> {key}={value}")
        lines.append("<H4> " + "  ".join(self._columns))

        V = np.asarray(jv.V, dtype=float)
        J = np.asarray(jv.J, dtype=float)
        P = -J * V
        for v, j, p in zip(V, J, P):
            row = "  ".join(self._fmt.format(x) for x in (v, j, p))
            lines.append(f"<L>  {row}")
        return "\n".join(lines) + "\n"

    def write(self, jv: JVData, path: Path, *, title: str = "PVK Simulator",
              run_id: str = "", meta: Optional[Mapping[str, str]] = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.dumps(jv, title=title, run_id=run_id, meta=meta),
                       encoding="utf-8", newline="\n")
        return out


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------

class SilvacoLogReader:
    """解析 SSF .log；列名自动识别，单位参与电流换算。"""

    # ----- 文件级 -------------------------------------------------------
    def read(self, path: Path) -> LogTable:
        """返回最后一个非空数据块（一次扫描的正常情形）。"""
        blocks = self.read_blocks(path)
        if not blocks:
            raise ValueError(f"{path}: 没有找到 SSF 数据块")
        return blocks[-1]

    def read_blocks(self, path: Path) -> Tuple[LogTable, ...]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.parse(text)

    # ----- 文本级 -------------------------------------------------------
    def parse(self, text: str) -> Tuple[LogTable, ...]:
        meta: Dict[str, str] = {}
        raw_blocks: List[Dict[str, object]] = []
        current: Optional[Dict[str, object]] = None

        for raw_line in text.splitlines():
            line = raw_line.lstrip("\ufeff").strip()
            if not line:
                continue

            header = _HEADER_RE.match(line)
            if header is not None:
                tag = header.group(1).upper()
                body = header.group(2).strip()
                if tag == "H4":
                    current = {"legend": body, "rows": [], "meta": dict(meta)}
                    raw_blocks.append(current)
                elif tag == "L":
                    values = _parse_numbers(body)
                    if values is not None:
                        if current is None:
                            current = {"legend": "", "rows": [], "meta": dict(meta)}
                            raw_blocks.append(current)
                        current["rows"].append(values)   # type: ignore[union-attr]
                elif tag.startswith("H"):
                    meta[tag] = body
                continue

            values = _parse_numbers(line)
            if values is not None and current is not None:
                current["rows"].append(values)           # type: ignore[union-attr]

        tables = []
        for block in raw_blocks:
            table = _build_table(block["legend"], block["rows"], block["meta"])
            if table is not None:
                tables.append(table)
        return tuple(tables)

    # ----- JV 转换 ------------------------------------------------------
    def to_jv(self, table: LogTable, *, area_cm2: float = 1.0,
              pin_mW_cm2: float = 100.0, sign: float = 1.0) -> JVData:
        """识别电压/电流列并转成 :class:`JVData`。

        ``area_cm2`` 在电流单位是 A / A/cm²，或单位缺失（按 Silvaco 约定
        视为 A）时参与换算；``sign=-1`` 用于电流方向与内核约定相反的导出文件。
        """
        v_col = table.find("voltage", "bias") or _exact_column(table, "v")
        j_col = (table.find("current", "density")
                 or _exact_column(table, "j", "i"))
        if v_col is None or j_col is None:
            raise ValueError(
                f"无法自动识别电压/电流列；现有列：{list(table.columns)}。"
                f"可在 deck 的 log 语句里改用标准列名"
            )

        scale = _current_scale(table.unit_of(j_col), area_cm2)
        V = np.asarray(table.data[v_col], dtype=float)
        J = sign * np.asarray(table.data[j_col], dtype=float) * scale
        return JVData.from_points(V, J, pin_mW_cm2=pin_mW_cm2)

    def read_jv(self, path: Path, **kwargs) -> JVData:
        return self.to_jv(self.read(path), **kwargs)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _exact_column(table: LogTable, *names: str) -> Optional[str]:
    """按列名精确匹配（用于 ``V`` / ``I`` / ``J`` 这类单字母列）。"""
    lowered = {c.lower(): c for c in table.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _parse_numbers(text: str) -> Optional[List[float]]:
    tokens = [t for t in _NUM_SPLIT_RE.split(text.strip()) if t]
    if not tokens:
        return None
    values = []
    for tok in tokens:
        try:
            values.append(float(tok.replace("D", "E").replace("d", "e")))
        except ValueError:
            return None
    return values


def _resolve_columns(legend: str, n_cols: int) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    if not legend:
        return tuple(f"col{i + 1}" for i in range(n_cols)), {}

    parts = [p for p in re.split(r"\s{2,}|\t+", legend.strip()) if p]
    if len(parts) != n_cols:
        parts = legend.split()

    if len(parts) != n_cols:
        return tuple(f"col{i + 1}" for i in range(n_cols)), {}

    names: List[str] = []
    units: Dict[str, str] = {}
    for part in parts:
        match = _UNIT_RE.match(part.strip())
        if match is not None:
            name, unit = match.group(1).strip(), match.group(2).strip()
            names.append(name)
            units[name] = unit
        else:
            names.append(part.strip())

    # 去重（Silvaco 偶尔出现同名列）
    unique: List[str] = []
    for name in names:
        candidate, k = name, 2
        while candidate in unique:
            candidate = f"{name}_{k}"
            k += 1
        unique.append(candidate)
        if name in units and candidate != name:
            units[candidate] = units[name]
    return tuple(unique), units


def _build_table(legend: object, rows: object, meta: object) -> Optional[LogTable]:
    row_list = [r for r in rows if r]            # type: ignore[union-attr]
    if not row_list:
        return None
    n_cols = len(row_list[0])
    consistent = [r for r in row_list if len(r) == n_cols]
    columns, units = _resolve_columns(str(legend), n_cols)
    data = {
        name: np.asarray([r[i] for r in consistent], dtype=float)
        for i, name in enumerate(columns)
    }
    return LogTable(columns=columns, data=data, units=units,
                    meta=dict(meta or {}))       # type: ignore[arg-type]


def _current_scale(unit: str, area_cm2: float) -> float:
    """把电流列换算成 mA/cm²。

    单位缺失时按 Silvaco 约定视为 **A**（Atlas 的 Anode Current 就是 A）；
    自己写出的 .log 一定带 ``(mA/cm2)``，不受这条约定影响。
    """
    norm = unit.lower().replace(" ", "").replace("^", "")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2={area_cm2} 必须 > 0")
    if norm in ("ma/cm2", "ma/cm²", "macm-2"):
        return 1.0
    if norm in ("a/cm2", "a/cm²", "acm-2"):
        return 1e3
    if norm in ("ma", "milliamp", "milliampere"):
        return 1.0
    # a / ampere / 空单位 / 未知 → 按 A 处理
    return 1e3 / area_cm2
