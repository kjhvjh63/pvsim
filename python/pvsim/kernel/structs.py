"""ctypes.Structure 映射 — 必须与 csrc/include/pvk/types.h 字段顺序完全一致。

任何一侧增删改字段，另一侧必须同步改，否则会造成 silently wrong 内存布局。
"""

from __future__ import annotations

import ctypes as ct
import math
from ctypes import c_int, c_double, c_char_p, POINTER, byref

# ── ctype 快捷别名 ──
CCharP   = c_char_p       # const char*
CDouble  = c_double
CInt     = c_int
PPDouble = POINTER(CDouble)
PCtx     = ct.c_void_p    # pvk_ctx* opaque handle

# ── LayerSpec ────────────────────────────────────────────────────────────
# types.h pvk_layer 的字段顺序
class Layer(ct.Structure):
    _fields_ = [
        ("name",           CCharP),
        ("material",       CCharP),
        ("thick_nm",       CDouble),
        ("eps_r",          CDouble),
        ("Eg_eV",          CDouble),
        ("chi_eV",         CDouble),
        ("Nc_cm3",         CDouble),
        ("Nv_cm3",         CDouble),
        ("mun_cm2Vs",      CDouble),
        ("mup_cm2Vs",      CDouble),
        ("taun_s",         CDouble),
        ("taup_s",         CDouble),
        ("Nd_cm3",         CDouble),
        ("Na_cm3",         CDouble),
        ("is_contact",     CInt),
        ("is_gen",         CInt),
    ]

    def __repr__(self):
        return (f"Layer(name={self.name!r}, thick_nm={self.thick_nm}, "
                f"Eg_eV={self.Eg_eV}, eps_r={self.eps_r})")


# ── DeviceSpec ───────────────────────────────────────────────────────────
class Device(ct.Structure):
    _fields_ = [
        ("layers",            POINTER(Layer)),
        ("n_layers",          CInt),
        ("T_K",               CDouble),
        ("nodes",             CInt),
        ("interface_refine",  CDouble),
    ]


# ── OpticsSpec ───────────────────────────────────────────────────────────
class OpticsIn(ct.Structure):
    _fields_ = [
        ("wl_nm",              PPDouble),
        ("n_wl",               CInt),
        ("flux",               PPDouble),
        ("intensity_mW_cm2",  CDouble),
        ("bounces",            CInt),
        ("bandgap_mask",       CInt),
        ("nk_dir",             CCharP),
    ]


# ── Bias ─────────────────────────────────────────────────────────────────
class Bias(ct.Structure):
    _fields_ = [
        ("anode_V",           CDouble),
        ("cathode_V",         CDouble),
        ("bc_type",           CInt),
        ("work_fn_anode_eV",  CDouble),
        ("work_fn_cathode_eV", CDouble),
    ]

BC_OHMIC   = 0
BC_SCHOTTKY = 1


# ── SolverOpts ────────────────────────────────────────────────────────────
class SolverOpts(ct.Structure):
    _fields_ = [
        ("res_tol",          CDouble),
        ("max_iter",         CInt),
        ("method",           CInt),
        ("damping",          CDouble),
        ("v_init_step",      CDouble),
        ("v_min_step",       CDouble),
        ("v_max_step",       CDouble),
        ("clamp_residual",   CInt),
    ]

METHOD_NEWTON                 = 0
METHOD_GUMMEL                 = 1
METHOD_NEWTON_WITH_WARMSTART  = 2


# ── SolveInfo ────────────────────────────────────────────────────────────
class SolveInfo(ct.Structure):
    _fields_ = [
        ("n",                 CInt),
        ("J_total_mA_cm2",    CDouble),
        ("converged",         CInt),
        ("residual",          CDouble),
        ("iterations",        CInt),
        ("v_applied",         CDouble),
    ]


# ── 枚举镜像（与 types.h 一致） ──────────────────────────────────────────
FIELD_ID = {
    "X":       0,
    "PSI":     1,
    "N":       2,
    "P":       3,
    "EFIELD":  4,
    "JN":      5,
    "JP":      6,
    "JTOTAL":  7,
    "R_NET":   8,
    "G_OPT":   9,
    "CB":     10,
    "VB":     11,
    "PHIN":   12,
    "PHIP":   13,
    "COUNT":  14,
}


# ── 把 Python dataclass 转成 ctypes ───────────────────────────────────────
# 用于让 kernel/bridge.py 直接从 ProblemSpec 构建 Device/OpticsIn/Bias
# 等 C 结构，避免 structs.py 和 schema.py 脱节。

def layer_from_spec(spec: "LayerSpec") -> Layer:  # type: ignore[name-defined]
    """schema.LayerSpec → ctypes Layer

    ``defects`` 不进入 C 结构（pvk_layer 不暴露缺陷数组），而是在这里
    折算成 ``effective_tau_*`` 写入 taun_s/taup_s —— 不改变 ABI。
    """
    tau_n = spec.effective_tau_n_s
    tau_p = spec.effective_tau_p_s
    if not math.isfinite(tau_n):
        tau_n = spec.taun_s
    if not math.isfinite(tau_p):
        tau_p = spec.taup_s
    # 注意：ctypes 里 char* 指向的字符串不能在 Layer 对象构造后释放，
    # 所以这里把 .name / .material 保持为 bytes 生命周期和 Layer 对象绑定
    return Layer(
        name=spec.name.encode("utf-8") if spec.name else None,
        material=(spec.material.encode("utf-8") if spec.material else None),
        thick_nm=spec.thick_nm,
        eps_r=spec.eps_r,
        Eg_eV=spec.Eg_eV,
        chi_eV=spec.chi_eV,
        Nc_cm3=spec.Nc_cm3,
        Nv_cm3=spec.Nv_cm3,
        mun_cm2Vs=spec.mun_cm2Vs,
        mup_cm2Vs=spec.mup_cm2Vs,
        taun_s=tau_n,
        taup_s=tau_p,
        Nd_cm3=spec.Nd_cm3,
        Na_cm3=spec.Na_cm3,
        is_contact=1 if spec.is_contact else 0,
        is_gen=1 if spec.is_gen else 0,
    )


def device_from_spec(spec: "DeviceSpec") -> tuple[Device, list[Layer]]:  # type: ignore[name-defined]
    """schema.DeviceSpec → (ctypes Device, [Layer...])

    返回 Device 和 Layer 列表两个对象，必须共同持有（Layer 的字符串
    不能在 Device 被使用时释放）。"""
    layers = [layer_from_spec(l) for l in spec.layers]
    arr = (Layer * len(layers))(*layers)
    dev = Device(
        layers=arr,
        n_layers=len(layers),
        T_K=spec.T_K,
        nodes=spec.nodes,
        interface_refine=spec.interface_refine,
    )
    return dev, layers


def optics_from_spec(spec: "OpticsSpec") -> OpticsIn:  # type: ignore[name-defined]
    """schema.OpticsSpec → ctypes OpticsIn

    把 numpy 数组转成 ctypes double*，调用方要保证 numpy 数组
    在 OpticsIn 被 C 使用期间不释放（正常 ProblemSpec 会一直持有）。"""
    import numpy as np
    wl_ptr = spec.wl_nm.ctypes.data_as(POINTER(CDouble))
    flux_ptr = spec.flux.ctypes.data_as(POINTER(CDouble))
    return OpticsIn(
        wl_nm=wl_ptr,
        n_wl=int(spec.n_wl),
        flux=flux_ptr,
        intensity_mW_cm2=spec.intensity_mW_cm2,
        bounces=int(spec.bounces),
        bandgap_mask=1 if spec.bandgap_mask else 0,
        nk_dir=(spec.nk_dir.encode("utf-8") if spec.nk_dir else None),
    )


def bias_from_spec(sweep: "SweepSpec", v: float) -> Bias:  # type: ignore[name-defined]
    """在某个偏压 v 下构造 ctypes Bias"""
    bc = BC_OHMIC if sweep.bc_type == "OHMIC" else BC_SCHOTTKY
    return Bias(
        anode_V=v if sweep.sweep_side == "ANODE" else sweep.anode_V_fixed,
        cathode_V=v if sweep.sweep_side == "CATHODE" else sweep.cathode_V_fixed,
        bc_type=bc,
        work_fn_anode_eV=sweep.work_fn_anode,
        work_fn_cathode_eV=sweep.work_fn_cathode,
    )


def solver_opts_from_spec(sopts: "SolverOpts") -> SolverOpts:  # type: ignore[name-defined]
    """schema.SolverOpts → ctypes SolverOpts"""
    m = {
        "NEWTON":                  METHOD_NEWTON,
        "GUMMEL":                  METHOD_GUMMEL,
        "NEWTON_WITH_WARMSTART":   METHOD_NEWTON_WITH_WARMSTART,
    }
    return SolverOpts(
        res_tol=sopts.res_tol,
        max_iter=sopts.max_iter,
        method=m.get(sopts.method, METHOD_NEWTON_WITH_WARMSTART),
        damping=sopts.damping,
        v_init_step=sopts.v_init_step,
        v_min_step=sopts.v_min_step,
        v_max_step=sopts.v_max_step,
        clamp_residual=1 if sopts.clamp_residual else 0,
    )
