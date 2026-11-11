"""Silvaco ``.mat`` 材料库解析。

文件语法（Atlas 风格子集）::

    # 注释到行尾
    material name=MAPBI3
      permittivity=24.0 eg300=1.6 chi3=3.93 \\
      nc300=2.2e18 nv300=1.0e19 mun=1.6 mup=1.7 taun0=8e-9 taup0=8e-9

约定
----
* 每个 ``material`` 块以一个 ``name=``（或旧写法 ``material=``）开头，
  块内的 ``key=value`` 按空白/逗号分隔，行尾 ``\\`` 续行；
* 关键字大小写不敏感，数值支持 ``1.0D-03`` 这种 Fortran 指数；
* 只取 300 K 参数：``eg300`` / ``nc300`` / ``nv300`` 优先于 ``eg`` / ``nc`` / ``nv``
  （内核当前不做温度标定，拿到别的温度值只会误导）；
* 未识别的键（``auger``、``user.default`` 等）原样保留在
  :attr:`MaterialSpec.extra`，不丢信息也不参与计算；
* 一个材料至少要有一个可识别参数，否则视为写错并报错——
  否则它会退化成"未知材料"，错误信息离现场太远。

:class:`MaterialDb` 提供 ``lookup(name) -> dict | None``，正好是
``DeckCompiler(material_db=...)`` 需要的协议。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*([^\s,;]+)")

# .mat 键 → LayerSpec 字段（同义键按 _PREFERENCE 的先后取第一个出现的）
_FIELD_KEYS: Dict[str, str] = {
    "eg300": "Eg_eV", "eg": "Eg_eV", "eg0": "Eg_eV",
    "chi3": "chi_eV", "chi": "chi_eV", "affinity": "chi_eV",
    "permittivity": "eps_r", "epsilon": "eps_r", "eps": "eps_r",
    "mun": "mun_cm2Vs", "mup": "mup_cm2Vs",
    "nc300": "Nc_cm3", "nc": "Nc_cm3",
    "nv300": "Nv_cm3", "nv": "Nv_cm3",
    "taun0": "taun_s", "taun": "taun_s",
    "taup0": "taup_s", "taup": "taup_s",
}

# 同义键的优先级：先 300 K 值，再通用值
_PREFERENCE: Dict[str, Tuple[str, ...]] = {
    "Eg_eV":   ("eg300", "eg", "eg0"),
    "chi_eV":  ("chi3", "chi", "affinity"),
    "eps_r":   ("permittivity", "epsilon", "eps"),
    "Nc_cm3":  ("nc300", "nc"),
    "Nv_cm3":  ("nv300", "nv"),
    "taun_s":  ("taun0", "taun"),
    "taup_s":  ("taup0", "taup"),
    "mun_cm2Vs": ("mun",),
    "mup_cm2Vs": ("mup",),
}

_NAME_KEYS = ("name", "material")


class MatParseError(ValueError):
    """``.mat`` 文件内容不合法。"""


# ---------------------------------------------------------------------------
# 单个材料
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSpec:
    """一条材料记录：已映射成 ``LayerSpec`` 字段的参数 + 未识别键。"""

    name:   str
    fields: Tuple[Tuple[str, float], ...] = ()
    extra:  Tuple[Tuple[str, str], ...] = ()
    source: str = ""

    def as_dict(self) -> Dict[str, float]:
        """给 ``LayerSpec(**params)`` 用的参数字典。"""
        return dict(self.fields)

    def __len__(self) -> int:
        return len(self.fields)


# ---------------------------------------------------------------------------
# 材料库
# ---------------------------------------------------------------------------

class MaterialDb:
    """按材料名索引的只读材料库（键统一大写，查询大小写不敏感）。"""

    def __init__(self, specs: Sequence[MaterialSpec] = ()):
        self._items: Dict[str, MaterialSpec] = {}
        for spec in specs:
            self.add(spec)

    # ----- 构建 ---------------------------------------------------------
    def add(self, spec: MaterialSpec) -> None:
        key = spec.name.upper()
        if not key:
            raise ValueError("MaterialSpec.name 不能为空")
        if key in self._items:
            raise ValueError(
                f"材料 '{spec.name}' 重复定义"
                f"（已有来自 {self._items[key].source or '<内存>'}）"
            )
        self._items[key] = spec

    def load_file(self, path: Path) -> int:
        """解析一个 .mat 文件并加入库，返回新增条数。"""
        specs = parse_mat(Path(path).read_text(encoding="utf-8", errors="replace"),
                          source=str(path))
        for spec in specs:
            self.add(spec)
        return len(specs)

    def load_directory(self, directory: Path, pattern: str = "*.mat") -> int:
        """按文件名顺序加载目录下所有 .mat。"""
        total = 0
        for path in sorted(Path(directory).glob(pattern)):
            total += self.load_file(path)
        return total

    @classmethod
    def from_files(cls, *paths: Iterable[Path]) -> "MaterialDb":
        db = cls()
        for path in paths:
            db.load_file(Path(path))
        return db

    @classmethod
    def from_directory(cls, directory: Path, pattern: str = "*.mat") -> "MaterialDb":
        db = cls()
        db.load_directory(Path(directory), pattern)
        return db

    # ----- 查询（DeckCompiler 协议） -------------------------------------
    def lookup(self, name: str) -> Optional[Dict[str, float]]:
        spec = self._items.get(name.upper())
        return spec.as_dict() if spec is not None else None

    def get(self, name: str) -> MaterialSpec:
        try:
            return self._items[name.upper()]
        except KeyError:
            raise KeyError(
                f"材料库中没有 '{name}'；可用：{', '.join(self.keys()) or '(空)'}"
            ) from None

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._items)

    def names(self) -> Tuple[str, ...]:
        """原始大小写的材料名（给 UI 显示用）。"""
        return tuple(spec.name for spec in self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.upper() in self._items

    def __iter__(self) -> Iterator[MaterialSpec]:
        return iter(self._items.values())

    def __repr__(self) -> str:
        return f"MaterialDb({len(self._items)} materials: {', '.join(self.keys())})"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def parse_mat(text: str, source: str = "") -> Tuple[MaterialSpec, ...]:
    """解析 .mat 文本，返回文件里定义的所有材料。"""
    specs: List[MaterialSpec] = []
    raw: Optional[Dict[str, str]] = None
    name: Optional[str] = None

    def flush() -> None:
        nonlocal raw, name
        if raw is None:
            return
        specs.append(_build_spec(name or "", raw, source))
        raw, name = None, None

    for line in _logical_lines(text):
        pairs = _PAIR_RE.findall(line)
        for key, value in pairs:
            key = key.lower()
            if key in _NAME_KEYS:
                flush()
                name = value.strip()
                raw = {}
            elif raw is None:
                raise MatParseError(
                    f"{_where(source)}: 参数 {key}={value} 出现在任何 material 定义之前"
                )
            else:
                if key in raw:
                    raise MatParseError(
                        f"{_where(source)}: 材料 '{name}' 的参数 {key} 重复"
                    )
                raw[key] = value
    flush()
    return tuple(specs)


def _logical_lines(text: str) -> Iterator[str]:
    """去掉注释、合并行尾 ``\\`` 续行。"""
    buffer = ""
    for raw_line in text.splitlines():
        body = raw_line.split("#", 1)[0].rstrip()
        if not body.strip():
            continue
        if body.endswith("\\"):
            buffer += body[:-1] + " "
            continue
        buffer += body
        yield buffer
        buffer = ""
    if buffer.strip():
        yield buffer


def _build_spec(name: str, raw: Mapping[str, str],
                source: str) -> MaterialSpec:
    if not name:
        raise MatParseError(f"{_where(source)}: material 块缺少 name=")

    fields: Dict[str, float] = {}
    used: set = set()
    for field, keys in _PREFERENCE.items():
        for key in keys:
            if key in raw:
                fields[field] = _to_float(raw[key], key, name, source)
                used.add(key)
                break

    if not fields:
        raise MatParseError(
            f"{_where(source)}: 材料 '{name}' 没有任何可识别参数；"
            f"支持的键：{', '.join(sorted(set(_FIELD_KEYS)))}"
        )

    extra = tuple(sorted((k, v) for k, v in raw.items() if k not in used))
    return MaterialSpec(name=name,
                        fields=tuple(sorted(fields.items())),
                        extra=extra, source=source)


def _to_float(text: str, key: str, name: str, source: str) -> float:
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise MatParseError(
            f"{_where(source)}: 材料 '{name}' 的 {key}={text!r} 不是数值"
        ) from None


def _where(source: str) -> str:
    return source or "<text>"
