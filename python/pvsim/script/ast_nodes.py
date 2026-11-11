"""deck AST：每条语句一个 frozen dataclass，全部带 ``line`` 便于定位。

解析（parser）只负责"语法对不对"，语义（材料是否存在、region 编号是否
合法）留给编译器。因此节点里尽量保留可选值，缺省用 ``None`` 表示"未指定"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class MeshStmt:
    nodes:      Optional[int] = None
    refine:     Optional[float] = None
    space_mult: Optional[float] = None
    temperature: Optional[float] = None
    line:       int = 0


@dataclass(frozen=True)
class RegionStmt:
    material:     str = ""
    thickness_nm: Optional[float] = None
    number:       Optional[int] = None
    name:         Optional[str] = None
    line:         int = 0


@dataclass(frozen=True)
class DopingStmt:
    kind:   str = "n"          # n / p
    conc:   float = 0.0        # cm⁻³
    region: Optional[int] = None
    line:   int = 0


@dataclass(frozen=True)
class ContactStmt:
    name:         str = ""     # anode / cathode
    region:       Optional[int] = None
    voltage:      Optional[float] = None
    workfunction: Optional[float] = None
    line:         int = 0


@dataclass(frozen=True)
class MaterialStmt:
    name:   str = ""
    params: Tuple[Tuple[str, float], ...] = ()   # 已映射成 LayerSpec 字段名
    line:   int = 0

    def as_dict(self) -> Dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True)
class ModelStmt:
    models: Tuple[str, ...] = ()
    line:   int = 0


@dataclass(frozen=True)
class OpticsStmt:
    intensity:    Optional[float] = None
    bounces:      Optional[int] = None
    bandgap_mask: Optional[bool] = None
    spectrum:     Optional[str] = None
    nk_dir:       Optional[str] = None
    line:         int = 0


@dataclass(frozen=True)
class MethodStmt:
    method:   Optional[str] = None
    max_iter: Optional[int] = None
    tol:      Optional[float] = None
    damping:  Optional[float] = None
    line:     int = 0


@dataclass(frozen=True)
class SolveStmt:
    v_start:  Optional[float] = None
    v_stop:   Optional[float] = None
    v_step:   Optional[float] = None
    n_points: Optional[int] = None
    log:      bool = False
    line:     int = 0


@dataclass(frozen=True)
class OutputStmt:
    fields:   Tuple[str, ...] = ()
    run_name: Optional[str] = None
    file:     Optional[str] = None
    format:   Optional[str] = None       # npy / ssf
    line:     int = 0


@dataclass(frozen=True)
class LogStmt:
    path: str = ""
    line: int = 0


@dataclass(frozen=True)
class UnknownStmt:
    keyword: str = ""
    raw:     str = ""
    line:    int = 0


Statement = Any


@dataclass(frozen=True)
class Deck:
    """解析产物：语句序列 + 非致命警告 + 标题。"""

    statements: Tuple[Statement, ...] = ()
    warnings:   Tuple[str, ...] = ()
    title:      str = ""

    def __len__(self) -> int:
        return len(self.statements)

    def __iter__(self) -> Iterator[Statement]:
        return iter(self.statements)

    def of(self, cls: Type[T]) -> Tuple[T, ...]:
        return tuple(s for s in self.statements if isinstance(s, cls))

    def first(self, cls: Type[T]) -> Optional[T]:
        for s in self.statements:
            if isinstance(s, cls):
                return s
        return None

    def last(self, cls: Type[T]) -> Optional[T]:
        found = self.of(cls)
        return found[-1] if found else None

    def has(self, cls: Type[T]) -> bool:
        return any(isinstance(s, cls) for s in self.statements)
