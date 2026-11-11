"""ProblemSpec — the single Intermediate Representation (IR).

This module defines the frozen dataclass tree that sits on the boundary
between the Python business layer and the C numerical kernel.  Every
downstream module (solver, dataio, viz, gui) only understands these
types.  The C types.h PODs in ``include/pvk/types.h`` map 1:1 onto the
simpler sub-structures (DeviceSpec → pvk_device, LayerSpec → pvk_layer,
OpticsSpec → pvk_optics_in, the bias fields inside SweepSpec → pvk_bias,
SolverOpts → pvk_solver_opts).

Design rules
------------
* All classes are ``@dataclass(frozen=True)`` — safe to hold long-term
  references and to use as dictionary keys.  Mutable containers (list,
  dict, numpy arrays) are normalised to immutable forms in
  ``__post_init__`` (tuple, tuple-of-dicts, numpy arrays kept as-is —
  they *are* a mutable type but are always replaced, never mutated).
* ``validate()`` returns ``None`` on success or raises ``SpecError``
  with a human-readable message; it also records which checks passed
  into ``_validation_notes`` for debugging.
* ``to_dict()`` / ``from_dict()`` provide round-trip JSON-serialisable
  output — numpy arrays become lists on the way out, numpy arrays on
  the way in.
* Zero kernel-specific logic.  No imports from ``pvk.kernel`` or
  anything below.  This file must be importable in isolation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class SpecError(ValueError):
    """Raised by ``ProblemSpec.validate`` when fields violate physics or
    the cross-language contract.  ``message`` is a short one-liner;
    ``details`` (optional) holds a longer bullet list for logging."""

    def __init__(self, message: str, details: Optional[List[str]] = None):
        super().__init__(message)
        self.details = details or []


# 子结构校验返回的都是自由文本，这里用关键字把"致命项"挑出来。
# 约定：致命项必须包含下列任一短语（与各 validate 的措辞保持一致）。
_FATAL_MARKERS: Tuple[str, ...] = ("must be", "out of", "empty", "unknown")


# ---------------------------------------------------------------------------
# Field whitelist — C kernel field_id constants must match these names
# ---------------------------------------------------------------------------

VALID_FIELDS: Tuple[str, ...] = (
    "X",         # 网格坐标 x (cm) — 始终输出
    "PSI",       # 静电势 ψ (V)
    "N",         # 电子浓度 n (cm⁻³)
    "P",         # 空穴浓度 p (cm⁻³)
    "EFIELD",    # 电场 E = −∂ψ/∂x (V/cm)
    "JN",        # 电子电流密度 J_n (mA/cm²)
    "JP",        # 空穴电流密度 J_p (mA/cm²)
    "JTOTAL",    # 总电流 J = J_n + J_p (mA/cm²)
    "R_NET",     # 净复合率 SRH(x) (cm⁻³ s⁻¹)
    "G_OPT",     # 光学产生率 G(x) (cm⁻³ s⁻¹) — 上一次 solve 所用
    "CB",        # 导带边 E_c(x) (eV) — 参考费米势为 0
    "VB",        # 价带边 E_v(x) (eV)
    "PHIN",      # 电子准费米势 φ_n (V)
    "PHIP",      # 空穴准费米势 φ_p (V)
)


# ---------------------------------------------------------------------------
# DefectSpec — single-level SRH trap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefectSpec:
    """单能级 SRH 陷阱（体缺陷）。

    能级约定
    --------
    ``Et_eV`` 是陷阱能级相对**价带顶 Ev** 的能量（eV），与
    ``LayerSpec.Eg_eV`` 共用同一参考点，因此合法范围是
    ``0 <= Et_eV <= Eg_eV``。需要带隙正中的陷阱时用
    ``DefectSpec.midgap(layer.Eg_eV)`` 构造。

    与 SRH 寿命的关系
    -----------------
    微观参数 → 寿命：``tau = 1 / (sigma * v_th * Nt)``。
    ``LayerSpec`` 同时有显式 ``taun_s/taup_s`` 和 ``defects`` 时，按下式
    合并（见 ``LayerSpec.effective_tau_n_s``）：

        1/tau_eff = 1/tau_显式 + Σ_i 1/tau_缺陷_i

    **ABI 说明**：``pvk_layer`` 结构本版不暴露缺陷数组。DefectSpec 只在
    Python 侧折算成 ``effective_tau_*`` 后填进 ``taun_s/taup_s`` 交给内核，
    因此不改变 C ABI、也不需要升 ``PVK_ABI_VERSION``。
    """

    name:          str   = "trap"
    Et_eV:         float = 0.8         # 相对 Ev 的陷阱能级 (eV)
    sigma_n_cm2:   float = 1e-15       # 电子俘获截面 (cm²)
    sigma_p_cm2:   float = 1e-15       # 空穴俘获截面 (cm²)
    Nt_cm3:        float = 1e15        # 陷阱体密度 (cm⁻³)
    kind:          str   = "ACCEPTOR"  # ACCEPTOR / DONOR / NEUTRAL

    VTH_CM_S = 1.0e7                   # 热速度 cm/s（300 K 近似）
    VALID_KINDS = ("ACCEPTOR", "DONOR", "NEUTRAL")

    # ----- derived ------------------------------------------------------
    @property
    def tau_n_s(self) -> float:
        """电子 SRH 寿命 (s)；参数非法时返回 inf（即"无复合"）。"""
        denom = self.sigma_n_cm2 * self.VTH_CM_S * self.Nt_cm3
        return 1.0 / denom if denom > 0 else float("inf")

    @property
    def tau_p_s(self) -> float:
        """空穴 SRH 寿命 (s)；参数非法时返回 inf。"""
        denom = self.sigma_p_cm2 * self.VTH_CM_S * self.Nt_cm3
        return 1.0 / denom if denom > 0 else float("inf")

    @classmethod
    def midgap(cls, Eg_eV: float, **kwargs) -> "DefectSpec":
        """构造带隙正中的陷阱。"""
        return cls(Et_eV=0.5 * float(Eg_eV), **kwargs)

    # ----- validation ---------------------------------------------------
    def validate(self, idx: int = 0, Eg_eV: Optional[float] = None) -> List[str]:
        """返回问题清单；空列表表示合法。

        ``Eg_eV`` 传入宿主层带隙时，额外检查 ``Et_eV`` 是否落在带隙内。
        """
        notes: List[str] = []
        prefix = f"defect[{idx}] '{self.name}'"
        if self.name == "":
            notes.append(f"{prefix}: name is empty")
        if self.Et_eV < 0:
            notes.append(f"{prefix}: Et_eV={self.Et_eV} must be >= 0 (相对 Ev)")
        if Eg_eV is not None and self.Et_eV > Eg_eV:
            notes.append(
                f"{prefix}: Et_eV={self.Et_eV} out of gap [0, {Eg_eV}] eV"
            )
        if self.sigma_n_cm2 <= 0:
            notes.append(f"{prefix}: sigma_n_cm2 must be > 0")
        if self.sigma_p_cm2 <= 0:
            notes.append(f"{prefix}: sigma_p_cm2 must be > 0")
        if self.Nt_cm3 <= 0:
            notes.append(f"{prefix}: Nt_cm3 must be > 0")
        if self.kind not in self.VALID_KINDS:
            notes.append(f"{prefix}: kind='{self.kind}' not in {self.VALID_KINDS}")
        return notes

    # ----- serialisation ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DefectSpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# LayerSpec  — one membrane of the device stack
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerSpec:
    """Python side of ``pvk_layer``.

    ``material`` is the Silvaco .mat library key name (or ``None`` if
    this layer is defined entirely by its explicit numeric fields).
    When non-``None``, the library is the primary source and the
    numeric fields on this struct act as overrides for whichever the
    caller wants to pin down — e.g.  a perovskite with ``material=
    "MAPBI3"`` but a bespoke ``Eg_eV=1.58``.

    All carrier-concentration fields (Nc, Nv, Nd, Na) are in the
    standard 3D cm⁻³ units.  Thickness is in nanometres, as that is
    how the process designers specify it.
    """

    name:                    str         # 友好标签，如 "ETL", "Perovskite", "HTL"
    material:                Optional[str] = None   # .mat 库键名；None 则完全用显式参数
    thick_nm:                float     = 0.0
    eps_r:                   float     = 1.0   # 相对介电常数
    Eg_eV:                   float     = 1.0   # 带隙
    chi_eV:                  float     = 0.0   # 电子亲和能
    Nc_cm3:                  float     = 1e19
    Nv_cm3:                  float     = 1e19
    mun_cm2Vs:               float     = 1.0   # 电子迁移率
    mup_cm2Vs:               float     = 0.1   # 空穴迁移率
    taun_s:                  float     = 1e-9  # 复合时间常数（SRH）
    taup_s:                  float     = 1e-9
    Nd_cm3:                  float     = 0.0   # n 型掺杂
    Na_cm3:                  float     = 0.0   # p 型掺杂
    is_contact:              bool      = False # 是否是电极层（边界条件挂载点）
    is_gen:                  bool      = True  # 是否参与光吸收计算
    defects:                 Tuple[DefectSpec, ...] = ()  # bulk SRH 陷阱（可空）

    def __post_init__(self):
        # 接受 list / 任意 Iterable → tuple，保证 frozen 语义与可哈希
        if not isinstance(self.defects, tuple):
            object.__setattr__(self, "defects", tuple(self.defects))

    # ----- derived helpers (no dataclass field) -------------------------
    @property
    def thick_cm(self) -> float:
        return self.thick_nm * 1e-7

    def _combine_tau(self, explicit_s: float, carrier: str) -> float:
        """1/tau_eff = 1/tau_显式 + Σ 1/tau_缺陷；两者都为空时返回 inf。"""
        rate = 0.0
        if explicit_s > 0:
            rate += 1.0 / explicit_s
        for d in self.defects:
            tau = d.tau_n_s if carrier == "n" else d.tau_p_s
            if 0.0 < tau < float("inf"):
                rate += 1.0 / tau
        return (1.0 / rate) if rate > 0 else float("inf")

    @property
    def effective_tau_n_s(self) -> float:
        """显式 taun_s 与所有缺陷折算寿命的合并结果（送内核用这个）。"""
        return self._combine_tau(self.taun_s, "n")

    @property
    def effective_tau_p_s(self) -> float:
        """显式 taup_s 与所有缺陷折算寿命的合并结果（送内核用这个）。"""
        return self._combine_tau(self.taup_s, "p")

    def validate(self, idx: int) -> List[str]:
        """Return a list of warning strings.  Empty → layer is OK."""
        notes: List[str] = []
        prefix = f"layer[{idx}] '{self.name}'"
        if self.name == "":
            notes.append(f"{prefix}: name is empty")
        if self.thick_nm <= 0:
            notes.append(f"{prefix}: thick_nm={self.thick_nm} must be > 0")
        if self.eps_r <= 0:
            notes.append(f"{prefix}: eps_r={self.eps_r} must be > 0")
        if self.Eg_eV <= 0:
            notes.append(f"{prefix}: Eg_eV={self.Eg_eV} must be > 0")
        if self.chi_eV < 0:
            # 电子亲和能理论上可以为负（如某些绝缘体），但钙钛矿体系中基本为正
            notes.append(f"{prefix}: chi_eV={self.chi_eV} is unusual (<0)")
        if self.Nc_cm3 <= 0 or self.Nv_cm3 <= 0:
            notes.append(f"{prefix}: Nc/Nv must be > 0")
        if self.mun_cm2Vs < 0 or self.mup_cm2Vs < 0:
            notes.append(f"{prefix}: mobility must be >= 0")
        if self.taun_s < 0 or self.taup_s < 0:
            notes.append(f"{prefix}: SRH tau must be >= 0")
        if self.Nd_cm3 < 0 or self.Na_cm3 < 0:
            notes.append(f"{prefix}: doping must be >= 0")
        for j, d in enumerate(self.defects):
            notes.extend(d.validate(j, Eg_eV=self.Eg_eV))
        return notes

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["defects"] = [x.to_dict() for x in self.defects]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerSpec":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw["defects"] = tuple(DefectSpec.from_dict(x) for x in d.get("defects", []))
        return cls(**kw)


# ---------------------------------------------------------------------------
# DeviceSpec  — layer stack + T + mesh hint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceSpec:
    """Python side of ``pvk_device``."""

    layers:                 Tuple[LayerSpec, ...]   # 层栈，顺序 = 从 anode 侧起
    T_K:                    float         = 300.0
    nodes:                  int           = 101      # 目标网格点数（实际可能因加密而略多）
    interface_refine:       float         = 0.0      # 界面加密比例，0=关闭

    def __post_init__(self):
        # 接受 list / 任意 Iterable → 转为 tuple 保证 frozen 安全
        if not isinstance(self.layers, tuple):
            object.__setattr__(self, "layers", tuple(self.layers))

    # ----- derived ------------------------------------------------------
    @property
    def total_thick_nm(self) -> float:
        return sum(l.thick_nm for l in self.layers)

    @property
    def total_thick_cm(self) -> float:
        return self.total_thick_nm * 1e-7

    @property
    def contact_layers(self) -> Tuple[LayerSpec, ...]:
        return tuple(l for l in self.layers if l.is_contact)

    @property
    def absorber_layers(self) -> Tuple[LayerSpec, ...]:
        """is_gen=True、非电极、厚>0 且带隙 0.8–4 eV 的光吸收层。

        显式排除 ``is_contact=True`` 的电极层（FTO/ITO/Au/Ag 等）——
        它们是边界条件挂载点，虽有 is_gen 默认 True，但不参与真正的
        产生率剖分。"""
        return tuple(
            l for l in self.layers
            if l.is_gen
            and not l.is_contact
            and l.thick_nm > 0
            and 0.8 <= l.Eg_eV <= 4.0
        )

    def validate(self) -> List[str]:
        notes: List[str] = []
        if len(self.layers) == 0:
            notes.append("DeviceSpec.layers is empty")
            return notes
        if self.T_K <= 0 or self.T_K > 1000:
            notes.append(f"DeviceSpec.T_K={self.T_K} K is out of sensible range")
        if self.nodes < 3:
            notes.append(f"DeviceSpec.nodes={self.nodes} must be >= 3")
        if not (0.0 <= self.interface_refine <= 2.0):
            notes.append(f"DeviceSpec.interface_refine={self.interface_refine} out of [0, 2]")
        # 每层逐验
        for i, l in enumerate(self.layers):
            notes.extend(l.validate(i))
        # 首尾必须是 contact
        if not self.layers[0].is_contact:
            notes.append("DeviceSpec: first layer must be a contact")
        if not self.layers[-1].is_contact:
            notes.append("DeviceSpec: last layer must be a contact")
        # 至少要有一个 absorber
        if len(self.absorber_layers) == 0:
            notes.append("DeviceSpec: no absorber layer detected (is_gen=True + Eg 0.8–4 eV)")
        return notes

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = [l.to_dict() for l in self.layers]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeviceSpec":
        layers = tuple(LayerSpec.from_dict(x) for x in d.get("layers", []))
        return cls(
            layers=layers,
            T_K=float(d.get("T_K", 300.0)),
            nodes=int(d.get("nodes", 101)),
            interface_refine=float(d.get("interface_refine", 0.0)),
        )


# ---------------------------------------------------------------------------
# OpticsSpec — incident spectrum + optical model knobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpticsSpec:
    """Python side of ``pvk_optics_in``.

    ``wl_nm`` and ``flux`` must be 1-D arrays of equal length.
    ``flux`` is spectral photon flux (cm⁻² s⁻¹ nm⁻¹) — the C kernel
    will internally apply ``phi_tot = flux.sum() * Δwl * intensity_mW_cm2 / 100``
    to scale everything.  Default flux is AM1.5G standard spectrum.
    """

    wl_nm:              np.ndarray
    flux:               np.ndarray
    intensity_mW_cm2:   float       = 100.0    # AM1.5G 1 sun = 100 mW/cm²
    bounces:            int         = 100      # 多程反射迭代次数（1/100/1000）
    bandgap_mask:       bool        = True     # λ > 1239.84/Eg 的波长不参与
    nk_dir:             Optional[str] = None   # 逐层 n,k 数据目录；None 则用统一默认表

    def __post_init__(self):
        # 确保 numpy 数组
        if not isinstance(self.wl_nm, np.ndarray):
            object.__setattr__(self, "wl_nm", np.asarray(self.wl_nm, dtype=float))
        if not isinstance(self.flux, np.ndarray):
            object.__setattr__(self, "flux", np.asarray(self.flux, dtype=float))

    # ----- derived ------------------------------------------------------
    @property
    def n_wl(self) -> int:
        return int(self.wl_nm.size)

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.wl_nm.ndim != 1 or self.flux.ndim != 1:
            notes.append("OpticsSpec.wl_nm/flux must be 1-D arrays")
            return notes
        if self.wl_nm.size != self.flux.size:
            notes.append(f"OpticsSpec: wl_nm size={self.wl_nm.size} ≠ flux size={self.flux.size}")
            return notes
        if self.wl_nm.size == 0:
            notes.append("OpticsSpec: empty wl_nm/flux")
            return notes
        if self.wl_nm.min() <= 0:
            notes.append(f"OpticsSpec: wl_nm must be positive; got min={self.wl_nm.min()}")
        if self.flux.min() < 0:
            notes.append(f"OpticsSpec: flux must be >= 0; got min={self.flux.min()}")
        if self.intensity_mW_cm2 < 0:
            notes.append(f"OpticsSpec: intensity_mW_cm2 must be >= 0")
        if self.bounces not in (1, 100, 1000):
            notes.append(f"OpticsSpec: bounces={self.bounces} unusual; expected 1/100/1000")
        # nk_dir=None 是完全合法的默认态（内核内置简化 n,k 表），不做 note
        return notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wl_nm": self.wl_nm.tolist(),
            "flux": self.flux.tolist(),
            "intensity_mW_cm2": self.intensity_mW_cm2,
            "bounces": self.bounces,
            "bandgap_mask": self.bandgap_mask,
            "nk_dir": self.nk_dir,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpticsSpec":
        return cls(
            wl_nm=np.asarray(d["wl_nm"], dtype=float),
            flux=np.asarray(d["flux"], dtype=float),
            intensity_mW_cm2=float(d.get("intensity_mW_cm2", 100.0)),
            bounces=int(d.get("bounces", 100)),
            bandgap_mask=bool(d.get("bandgap_mask", True)),
            nk_dir=d.get("nk_dir"),
        )

    @classmethod
    def default_am15g(cls, n_pts: int = 200) -> "OpticsSpec":
        """Construct a crudely linear AM1.5G spectrum from 300–1100 nm.

        Only used when no deck / script provides a real flux.  The C
        kernel is the authority on spectral integration; this just
        gives ProblemSpec a never-empty default."""
        wl = np.linspace(300, 1100, n_pts)
        # 一个简化的 AM1.5G 形状：峰值 ~550 nm、尾部 ~1100 nm
        flux = np.exp(-((wl - 550) ** 2) / (2 * 180 ** 2))
        flux *= 1e15  # 把量级拉到合理的光子数
        return cls(wl_nm=wl, flux=flux)


# ---------------------------------------------------------------------------
# SolverOpts  — 求解器数值控制
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverOpts:
    """Python side of ``pvk_solver_opts``."""

    res_tol:            float   = 1e-8
    max_iter:           int     = 60
    method:             str     = "NEWTON_WITH_WARMSTART"  # NEWTON / GUMMEL / NEWTON_WITH_WARMSTART
    damping:            float   = 1.0   # 阻尼 Newton 系数（0.5~1.0）
    v_init_step:        float   = 0.05  # continuation 初始电压步长 (V)
    v_min_step:         float   = 0.002 # knee 段自动缩到 (V)
    v_max_step:         float   = 0.10  # 平直段允许的最大步长 (V)
    clamp_residual:     bool    = True

    VALID_METHODS = ("NEWTON", "GUMMEL", "NEWTON_WITH_WARMSTART")

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.res_tol <= 0:
            notes.append(f"SolverOpts.res_tol={self.res_tol} must be > 0")
        if self.max_iter < 5:
            notes.append(f"SolverOpts.max_iter={self.max_iter} too low; >= 5")
        if self.method not in self.VALID_METHODS:
            notes.append(f"SolverOpts.method='{self.method}' not in {self.VALID_METHODS}")
        if not (0.0 < self.damping <= 1.0):
            notes.append(f"SolverOpts.damping={self.damping} out of (0, 1]")
        if self.v_init_step <= 0 or self.v_min_step <= 0 or self.v_max_step <= 0:
            notes.append("SolverOpts voltage steps must be > 0")
        if not (self.v_min_step <= self.v_init_step <= self.v_max_step):
            notes.append(
                f"SolverOpts: v_min({self.v_min_step}) ≤ v_init({self.v_init_step}) ≤ v_max({self.v_max_step}) violated"
            )
        return notes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverOpts":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# SweepSpec  — 扫描计划 + 边界条件
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepSpec:
    """描述一次偏压扫描，同时承载 pvk_bias 所需的全部字段。

    单点求解（``pvk_solve_dc``）就是 ``n_points == 1``、
    ``v_start == v_stop`` 的特例；``pvk_sweep_dc`` 则使用
    内部的 continuation。
    """

    # 扫描范围
    v_start:      float        = 0.0   # 起始偏压 (V)
    v_stop:       float        = 1.8   # 结束偏压 (V)
    n_points:     int          = 37    # 扫描点数；1 = 单点 solve_dc

    # 固定的边界电压（对极保持不变）
    anode_V_fixed:   float = 0.0       # anode 侧固定电压 (V)
    cathode_V_fixed: float = 0.0       # cathode 侧固定电压 (V)
    sweep_side:      str   = "ANODE"   # 扫描哪个电极："ANODE" / "CATHODE"

    # 边界类型
    bc_type:              str  = "OHMIC"       # OHMIC / SCHOTTKY
    work_fn_anode:        float = 4.7          # eV（仅 SCHOTTKY 生效）
    work_fn_cathode:      float = 4.7

    # 数值求解器选项
    solver_opts:          SolverOpts = field(default_factory=SolverOpts)

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.n_points < 1:
            notes.append(f"SweepSpec.n_points={self.n_points} must be >= 1")
        if self.n_points > 1 and not (self.v_start <= self.v_stop):
            notes.append(
                f"SweepSpec: multi-point sweep requires v_start({self.v_start}) ≤ v_stop({self.v_stop})"
            )
        if self.sweep_side not in ("ANODE", "CATHODE"):
            notes.append(f"SweepSpec.sweep_side='{self.sweep_side}' must be ANODE or CATHODE")
        if self.bc_type not in ("OHMIC", "SCHOTTKY"):
            notes.append(f"SweepSpec.bc_type='{self.bc_type}' must be OHMIC or SCHOTTKY")
        if self.bc_type == "SCHOTTKY":
            if self.work_fn_anode < 3.0 or self.work_fn_anode > 6.0:
                notes.append(f"SweepSpec.work_fn_anode={self.work_fn_anode} out of typical [3,6] eV")
            if self.work_fn_cathode < 3.0 or self.work_fn_cathode > 6.0:
                notes.append(f"SweepSpec.work_fn_cathode={self.work_fn_cathode} out of typical [3,6] eV")
        notes.extend(self.solver_opts.validate())
        return notes

    # ----- convenience factories ----------------------------------------
    @classmethod
    def single_point(cls, v: float, **kwargs) -> "SweepSpec":
        return cls(v_start=v, v_stop=v, n_points=1, **kwargs)

    @classmethod
    def jv_sweep(cls, v_stop: float = 1.8, n_points: int = 37,
                 v_start: float = 0.0, **kwargs) -> "SweepSpec":
        """标准 J-V 扫描：0 → v_stop V，均匀分点。"""
        return cls(v_start=v_start, v_stop=v_stop, n_points=n_points, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["solver_opts"] = self.solver_opts.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepSpec":
        return cls(
            v_start=float(d.get("v_start", 0.0)),
            v_stop=float(d.get("v_stop", 1.8)),
            n_points=int(d.get("n_points", 37)),
            anode_V_fixed=float(d.get("anode_V_fixed", 0.0)),
            cathode_V_fixed=float(d.get("cathode_V_fixed", 0.0)),
            sweep_side=str(d.get("sweep_side", "ANODE")),
            bc_type=str(d.get("bc_type", "OHMIC")),
            work_fn_anode=float(d.get("work_fn_anode", 4.7)),
            work_fn_cathode=float(d.get("work_fn_cathode", 4.7)),
            solver_opts=SolverOpts.from_dict(d.get("solver_opts", {})),
        )


# ---------------------------------------------------------------------------
# MaterialSet  — 材料库引用 + 覆盖参数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSet:
    """指向 Silvaco 通用 .mat 库的路径，加上按材料名的字段级覆盖。

    ``overrides`` 形如 ``{"MAPBI3": {"Eg_eV": 1.58, "eps_r": 24.0}}``。
    覆盖在 ``DeckCompiler`` / ``DeviceSpec.resolve_materials`` 里
    应用，Kernel 只收到已经解析好的数值。
    """

    mat_path:    Optional[str] = None   # .mat 文件路径；None = 只用显式参数
    overrides:   Tuple[Dict[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self):
        # overrides 传入 dict → 转为 tuple-of-sorted-kv 保证可哈希
        if isinstance(self.overrides, dict):
            norm = tuple(sorted(self.overrides.items()))
            object.__setattr__(self, "overrides", norm)

    # ----- derived ------------------------------------------------------
    def get(self, material_name: str, field_name: str) -> Optional[float]:
        """返回覆盖值或 None（没覆盖过）。"""
        if isinstance(self.overrides, dict):
            d = self.overrides
        else:
            d = dict(self.overrides)
        mat_overrides = d.get(material_name)
        if mat_overrides is None:
            return None
        return mat_overrides.get(field_name)

    def validate(self) -> List[str]:
        notes: List[str] = []
        # mat_path 不存在性的检查留给运行期（C kernel 会报错），这里只做语法性检查
        if self.mat_path is not None and not isinstance(self.mat_path, str):
            notes.append(f"MaterialSet.mat_path must be str or None")
        if isinstance(self.overrides, dict):
            d = self.overrides
        else:
            d = dict(self.overrides)
        for mat, fields in d.items():
            if not isinstance(mat, str) or mat == "":
                notes.append(f"MaterialSet.overrides has invalid material key: {mat!r}")
            for k, v in fields.items():
                if not isinstance(k, str):
                    notes.append(f"MaterialSet.overrides['{mat}'] has non-str field key {k!r}")
                if not isinstance(v, (int, float)):
                    notes.append(f"MaterialSet.overrides['{mat}']['{k}'] must be numeric")
        return notes

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.overrides, dict):
            ov = self.overrides
        else:
            ov = dict(self.overrides)
        return {"mat_path": self.mat_path, "overrides": ov}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaterialSet":
        ov_raw = d.get("overrides", {})
        return cls(mat_path=d.get("mat_path"), overrides=ov_raw)


# ---------------------------------------------------------------------------
# OutputSpec  — 场量清单 + run 元数据
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputSpec:
    fields:              Tuple[str, ...] = ("X", "PSI", "N", "P", "EFIELD", "JTOTAL", "G_OPT")
    run_name:            str             = "device"     # run_id 前缀："device_20261111_143015"
    format:              str             = "npy"        # npy / ssf（Silvaco 兼容 .log）

    def __post_init__(self):
        if not isinstance(self.fields, tuple):
            object.__setattr__(self, "fields", tuple(self.fields))

    def validate(self) -> List[str]:
        notes: List[str] = []
        known = set(VALID_FIELDS)
        unknown = [f for f in self.fields if f not in known]
        if unknown:
            notes.append(f"OutputSpec.fields has unknown entries: {unknown}")
        if self.run_name == "":
            notes.append("OutputSpec.run_name is empty")
        if self.format not in ("npy", "ssf"):
            notes.append(f"OutputSpec.format='{self.format}' must be 'npy' or 'ssf'")
        if "X" not in self.fields:
            notes.append("OutputSpec: 'X' should almost always be included (mesh coords)")
        return notes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputSpec":
        return cls(
            fields=tuple(d.get("fields", ["X", "PSI", "N", "P", "JTOTAL"])),
            run_name=str(d.get("run_name", "device")),
            format=str(d.get("format", "npy")),
        )


# ---------------------------------------------------------------------------
# ProblemSpec  — the IR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemSpec:
    """唯一中间表示。所有下游模块（solver、dataio、viz、gui）只认这一个类型。"""

    device:     DeviceSpec
    optics:     OpticsSpec
    sweep:      SweepSpec
    material:   MaterialSet = field(default_factory=MaterialSet)
    output:     OutputSpec  = field(default_factory=OutputSpec)

    # ----- 顶层验证 ----------------------------------------------------
    def validate(self) -> None:
        """跑遍所有子结构的 ``validate``，汇总报错后抛 SpecError。

        返回 None 表示全部通过；抛异常时 ``details`` 里带完整子项清单。
        只想知道"哪里不对"而不想处理异常时，用 ``validate_notes()``。"""
        fatal, warnings = self.validate_notes()
        if fatal or warnings:
            summary = ("ProblemSpec validation failed" if fatal
                       else "ProblemSpec has warnings")
            raise SpecError(summary, details=fatal + warnings)

    def validate_notes(self) -> Tuple[List[str], List[str]]:
        """返回 ``(fatal, warnings)`` 两个清单，不抛异常。

        fatal 表示会直接导致内核错误；warnings 只是可疑配置（例如
        带了 material 名却没有 .mat 路径）。供 Simulator / GUI 决定
        是中止还是仅记日志。"""
        notes: List[str] = self._collect_notes()
        fatal = [n for n in notes if any(k in n for k in _FATAL_MARKERS)]
        warnings = [n for n in notes if n not in fatal]
        return fatal, warnings

    def _collect_notes(self) -> List[str]:
        notes: List[str] = []

        notes.extend(self.device.validate())
        notes.extend(self.optics.validate())
        notes.extend(self.sweep.validate())
        notes.extend(self.material.validate())
        notes.extend(self.output.validate())

        # 交叉校验：device 层里引用的 material 是否能在 MaterialSet 里找到（不致命）
        # 注意 overrides 的默认值是空 tuple（不是 None），所以用真值判断
        if self.material.mat_path is None and not self.material.overrides:
            for l in self.device.layers:
                if l.material is not None:
                    notes.append(
                        f"ProblemSpec: layer material='{l.material}' but MaterialSet.mat_path is None — "
                        f"will fall back to layer's explicit numeric fields only"
                    )

        # 交叉校验：absorber Eg 是否能被光学产生（带隙掩模下至少一个波长能穿透）
        if self.optics.bandgap_mask and len(self.device.absorber_layers) > 0:
            longest_absorb = self.device.absorber_layers[0].Eg_eV
            # 用最窄带隙 absorber 做保守估计
            for l in self.device.absorber_layers:
                if l.Eg_eV < longest_absorb:
                    longest_absorb = l.Eg_eV
            cutoff_nm = 1239.84 / longest_absorb
            if self.optics.wl_nm.max() < cutoff_nm * 0.9:
                notes.append(
                    f"ProblemSpec: optics spectrum stops at {self.optics.wl_nm.max():.1f} nm "
                    f"but absorber cutoff is {cutoff_nm:.1f} nm — bandgap_mask 会遮掉全部吸收"
                )
        return notes

    # ----- 序列化 ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device.to_dict(),
            "optics": self.optics.to_dict(),
            "sweep": self.sweep.to_dict(),
            "material": self.material.to_dict(),
            "output": self.output.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProblemSpec":
        return cls(
            device=DeviceSpec.from_dict(d["device"]),
            optics=OpticsSpec.from_dict(d["optics"]),
            sweep=SweepSpec.from_dict(d["sweep"]),
            material=MaterialSet.from_dict(d.get("material", {})),
            output=OutputSpec.from_dict(d.get("output", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "ProblemSpec":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# SweepPlan — 扫描计划（一个计划展开成多个 ProblemSpec 变体）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanStep:
    """扫描计划展开后的单步：可读标签 + 该步完整的 ProblemSpec。"""

    label: str
    spec:  ProblemSpec


@dataclass(frozen=True)
class SweepPlan(ABC):
    """扫描计划基类。

    区分两个层次：

    * ``SweepSpec`` 是**一次求解**的数值包线（偏压范围、点数、求解器选项）；
    * ``SweepPlan`` 是**一次研究**：``expand(base_spec)`` 展开成若干
      ``PlanStep``，每个 step 携带一份完整、可直接交给 Simulator 的
      ``ProblemSpec``。

    这样 solver / dataio / viz 都不必理解"扫描"概念，只处理单个 spec；
    扫描的语义集中在 plan 里，便于 GUI 列出"这个计划会跑几个点"。

    实现约定：

    * ``expand`` 必须是纯函数，不得修改传入的 ``base_spec``（frozen + ``replace``）；
    * 子类用 ``validate()`` 报告自身参数问题，``expand`` 展开前会自动检查，
      有问题抛 ``SpecError``。
    """

    label: str = ""

    @abstractmethod
    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        """展开成若干具体 ProblemSpec。"""

    def validate(self) -> List[str]:
        """校验计划自身参数（不含与 base_spec 相关的检查）。"""
        return []

    # ----- shared helpers ------------------------------------------------
    def _ensure_valid(self) -> None:
        notes = self.validate()
        if notes:
            raise SpecError(f"{type(self).__name__} invalid", details=notes)

    def __call__(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        """让 plan 直接可调用：``plan(base_spec)`` ≡ ``plan.expand(base_spec)``。"""
        return self.expand(base_spec)


def _layer_index(layers: Tuple[LayerSpec, ...], name: str) -> int:
    for i, l in enumerate(layers):
        if l.name == name:
            return i
    raise SpecError(
        f"layer '{name}' not found",
        details=[f"available layers: {[l.name for l in layers]}"],
    )


def _replace_layer(layers: Tuple[LayerSpec, ...], name: str,
                   **changes) -> Tuple[LayerSpec, ...]:
    """按名字替换一层，返回新的层栈元组（不修改原对象）。"""
    i = _layer_index(layers, name)
    return layers[:i] + (replace(layers[i], **changes),) + layers[i + 1:]


@dataclass(frozen=True)
class BiasSweep(SweepPlan):
    """偏压扫描：单个 spec，内部由 kernel 的 continuation 完成。

    产出一个 ``SimResult``（其中含整条 J-V）。若想要"每个厚度一条 J-V"，
    用 ``ThicknessScan`` 把它包起来即可。三个字段为 ``None`` 时沿用
    ``base_spec.sweep`` 的对应值。
    """

    v_start:  Optional[float] = None
    v_stop:   Optional[float] = None
    n_points: Optional[int]   = None

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.n_points is not None and self.n_points < 1:
            notes.append("BiasSweep.n_points must be >= 1")
        if (self.n_points is not None and self.n_points > 1
                and self.v_start is not None and self.v_stop is not None
                and self.v_start > self.v_stop):
            notes.append("BiasSweep: multi-point requires v_start <= v_stop")
        return notes

    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        self._ensure_valid()
        sw = base_spec.sweep
        sw2 = replace(
            sw,
            v_start=self.v_start if self.v_start is not None else sw.v_start,
            v_stop=self.v_stop if self.v_stop is not None else sw.v_stop,
            n_points=self.n_points if self.n_points is not None else sw.n_points,
        )
        prefix = f"{self.label} " if self.label else ""
        label = f"{prefix}J-V {sw2.v_start:g}→{sw2.v_stop:g} V ({sw2.n_points} pts)"
        return (PlanStep(label=label, spec=replace(base_spec, sweep=sw2)),)


@dataclass(frozen=True)
class IntensitySweep(SweepPlan):
    """光强扫描（单位 sun，1 sun = 100 mW/cm²）。

    每个 sun 值产出一个 PlanStep，只改 ``optics.intensity_mW_cm2``。
    """

    sun_values: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)

    def __post_init__(self):
        if not isinstance(self.sun_values, tuple):
            object.__setattr__(self, "sun_values", tuple(self.sun_values))

    def validate(self) -> List[str]:
        notes: List[str] = []
        if len(self.sun_values) == 0:
            notes.append("IntensitySweep.sun_values is empty")
        for s in self.sun_values:
            if s < 0:
                notes.append(f"IntensitySweep: sun value {s} must be >= 0")
        return notes

    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        self._ensure_valid()
        prefix = f"{self.label} " if self.label else ""
        return tuple(
            PlanStep(
                label=f"{prefix}{s:g} sun",
                spec=replace(
                    base_spec,
                    optics=replace(base_spec.optics,
                                   intensity_mW_cm2=100.0 * float(s)),
                ),
            )
            for s in self.sun_values
        )

    @classmethod
    def two_stage(cls, n_low: int = 5, n_high: int = 5, **kwargs) -> "IntensitySweep":
        """两联扫描：1→0.2 sun（弱光）+ 1.0→1.8 sun（聚光）。"""
        if n_low < 1 or n_high < 1:
            raise ValueError("n_low / n_high must be >= 1")
        low = tuple(float(x) for x in np.linspace(1.0, 0.2, n_low))
        high = tuple(float(x) for x in np.linspace(1.0, 1.8, n_high))
        return cls(sun_values=low + high, **kwargs)


@dataclass(frozen=True)
class ThicknessScan(SweepPlan):
    """厚度扫描：改指定层的 ``thick_nm``。"""

    layer_name:      str                    = ""
    thick_nm_values: Tuple[float, ...]      = ()

    def __post_init__(self):
        if not isinstance(self.thick_nm_values, tuple):
            object.__setattr__(self, "thick_nm_values", tuple(self.thick_nm_values))

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.layer_name == "":
            notes.append("ThicknessScan.layer_name is empty")
        if len(self.thick_nm_values) == 0:
            notes.append("ThicknessScan.thick_nm_values is empty")
        for t in self.thick_nm_values:
            if t <= 0:
                notes.append(f"ThicknessScan: thick_nm={t} must be > 0")
        return notes

    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        self._ensure_valid()
        layers = base_spec.device.layers
        _layer_index(layers, self.layer_name)  # 提前报"层不存在"
        prefix = f"{self.label} " if self.label else ""
        return tuple(
            PlanStep(
                label=f"{prefix}{self.layer_name}={float(t):g} nm",
                spec=replace(
                    base_spec,
                    device=replace(
                        base_spec.device,
                        layers=_replace_layer(layers, self.layer_name,
                                              thick_nm=float(t)),
                    ),
                ),
            )
            for t in self.thick_nm_values
        )


@dataclass(frozen=True)
class DopingScan(SweepPlan):
    """掺杂浓度扫描：改指定层的 ``Nd_cm3`` 或 ``Na_cm3``。"""

    layer_name: str                = ""
    values_cm3: Tuple[float, ...]  = ()
    kind:       str                = "Nd"   # Nd | Na

    def __post_init__(self):
        if not isinstance(self.values_cm3, tuple):
            object.__setattr__(self, "values_cm3", tuple(self.values_cm3))

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.layer_name == "":
            notes.append("DopingScan.layer_name is empty")
        if len(self.values_cm3) == 0:
            notes.append("DopingScan.values_cm3 is empty")
        for v in self.values_cm3:
            if v < 0:
                notes.append(f"DopingScan: doping {v} must be >= 0")
        if self.kind not in ("Nd", "Na"):
            notes.append(f"DopingScan.kind='{self.kind}' must be 'Nd' or 'Na'")
        return notes

    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        self._ensure_valid()
        layers = base_spec.device.layers
        _layer_index(layers, self.layer_name)
        prefix = f"{self.label} " if self.label else ""
        return tuple(
            PlanStep(
                label=f"{prefix}{self.layer_name} {self.kind}={float(v):.3g} cm⁻³",
                spec=replace(
                    base_spec,
                    device=replace(
                        base_spec.device,
                        layers=_replace_layer(layers, self.layer_name,
                                              **{f"{self.kind}_cm3": float(v)}),
                    ),
                ),
            )
            for v in self.values_cm3
        )


@dataclass(frozen=True)
class DefectScan(SweepPlan):
    """缺陷扫描：改指定层第 ``defect_index`` 个陷阱的 ``Nt_cm3``。

    要求该层已有 ``DefectSpec``（否则抛 SpecError，提示先加缺陷）。
    """

    layer_name:     str                = ""
    Nt_values_cm3:  Tuple[float, ...]  = ()
    defect_index:   int                = 0

    def __post_init__(self):
        if not isinstance(self.Nt_values_cm3, tuple):
            object.__setattr__(self, "Nt_values_cm3", tuple(self.Nt_values_cm3))

    def validate(self) -> List[str]:
        notes: List[str] = []
        if self.layer_name == "":
            notes.append("DefectScan.layer_name is empty")
        if len(self.Nt_values_cm3) == 0:
            notes.append("DefectScan.Nt_values_cm3 is empty")
        for nt in self.Nt_values_cm3:
            if nt <= 0:
                notes.append(f"DefectScan: Nt={nt} must be > 0")
        if self.defect_index < 0:
            notes.append("DefectScan.defect_index must be >= 0")
        return notes

    def expand(self, base_spec: ProblemSpec) -> Tuple[PlanStep, ...]:
        self._ensure_valid()
        layers = base_spec.device.layers
        i = _layer_index(layers, self.layer_name)
        layer = layers[i]
        if not layer.defects:
            raise SpecError(
                f"DefectScan: layer '{self.layer_name}' has no DefectSpec",
                details=["add defects=(DefectSpec(...),) to this layer before scanning"],
            )
        if not (0 <= self.defect_index < len(layer.defects)):
            raise SpecError(
                f"DefectScan: defect_index={self.defect_index} out of range",
                details=[f"layer '{self.layer_name}' has {len(layer.defects)} defect(s)"],
            )

        prefix = f"{self.label} " if self.label else ""
        steps: List[PlanStep] = []
        for nt in self.Nt_values_cm3:
            defects = list(layer.defects)
            defects[self.defect_index] = replace(defects[self.defect_index],
                                                 Nt_cm3=float(nt))
            new_layers = (layers[:i]
                          + (replace(layer, defects=tuple(defects)),)
                          + layers[i + 1:])
            steps.append(
                PlanStep(
                    label=f"{prefix}{self.layer_name} Nt={float(nt):.3g} cm⁻³",
                    spec=replace(
                        base_spec,
                        device=replace(base_spec.device, layers=new_layers),
                    ),
                )
            )
        return tuple(steps)


# ---------------------------------------------------------------------------
# Ready-made constructors — 省得每次手敲
# ---------------------------------------------------------------------------

def mapbi3_stack() -> DeviceSpec:
    """一个合理的 5 层钙钛矿 NIP 异质结：FTO / TiO2 / MAPbI3 / Spiro-OMeTAD / Au。

    数字是常见值的近似，后续 .mat 库加载后覆盖。"""
    L = LayerSpec
    return DeviceSpec(
        layers=tuple([
            L(name="FTO",      thick_nm=200.0, eps_r=9.0,  Eg_eV=3.7,  chi_eV=4.2,
               Nc_cm3=1e21,    Nv_cm3=1e21, mun_cm2Vs=20.0,  mup_cm2Vs=10.0,
               Nd_cm3=1e20,    is_contact=True),
            L(name="TiO2",    thick_nm=40.0,  eps_r=10.0, Eg_eV=3.2,  chi_eV=4.0,
               Nc_cm3=5e20,    Nv_cm3=5e20, mun_cm2Vs=0.1,   mup_cm2Vs=0.01,
               Nd_cm3=0.0,     is_gen=False),
            L(name="MAPbI3",  thick_nm=500.0, eps_r=24.0, Eg_eV=1.6,  chi_eV=3.93,
               Nc_cm3=2.2e18,  Nv_cm3=1.0e19, mun_cm2Vs=1.6,   mup_cm2Vs=1.7,
               taun_s=8e-9,    taup_s=8e-9,  Nd_cm3=0, Na_cm3=0, is_gen=True),
            L(name="Spiro",   thick_nm=200.0, eps_r=3.0,  Eg_eV=3.0,  chi_eV=2.5,
               Nc_cm3=5e19,    Nv_cm3=5e19, mun_cm2Vs=0.01,  mup_cm2Vs=0.02,
               Na_cm3=0,       is_gen=False),
            L(name="Au",      thick_nm=80.0,  eps_r=1.0,  Eg_eV=1.0,  chi_eV=4.7,
               Nc_cm3=1e22,    Nv_cm3=1e22, mun_cm2Vs=100.0, mup_cm2Vs=80.0,
               Na_cm3=1e20,    is_contact=True),
        ]),
        T_K=300.0,
        nodes=101,
        interface_refine=1.0,
    )


def default_problem_spec() -> ProblemSpec:
    """完整的 5 层 MAPbI3 NIP + AM1.5G + 0→1.8 V J-V 扫描。"""
    return ProblemSpec(
        device=mapbi3_stack(),
        optics=OpticsSpec.default_am15g(),
        sweep=SweepSpec.jv_sweep(v_stop=1.8, n_points=37),
        material=MaterialSet(),
        output=OutputSpec(fields=("X", "PSI", "N", "P", "EFIELD", "JTOTAL", "G_OPT")),
    )
