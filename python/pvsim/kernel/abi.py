"""abi.py — libpvk 加载 + ABI 版本校验。

ctypes 封装层只做三件事：
  1. locate_and_load() — 根据常见路径找 DLL/SO/DYLIB
  2. bind_signatures() — 声明所有函数签名（argtypes/restype）
  3. PvkLib — 持有已加载库的轻量对象，提供 get_field 等便捷方法
"""

from __future__ import annotations

import ctypes as ct
import os
import sys
from pathlib import Path
from typing import Optional, Union

from . import structs as st


PVK_ABI_VERSION = 1  # 必须等于 csrc/include/pvk/pvk.h 里的 PVK_ABI_VERSION
ERROR_CODES = {
    0: "PVK_OK",
    1: "PVK_EINVAL",
    2: "PVK_ECONV",
    3: "PVK_EMEM",
    4: "PVK_EIO",
    5: "PVK_EMAT",
    6: "PVK_EAPI",
    7: "PVK_EMATH",
}


class PvkDLError(Exception):
    pass


class ABIError(PvkDLError):
    pass


def _candidate_paths() -> list[Path]:
    """返回按优先级排序的可能库路径。

    约定的库位置（从优先到兜底）：
      1. PVK_LIB 环境变量显式指定
      2. 源码树 csrc/build 下（用户构建后的默认位置）
      3. csrc 下（某些直接把 .so/.dll 放源码根的构建方式）
      4. 当前 Python 可执行文件同目录（pip install 后 wheel 方式）
    """
    out: list[Path] = []

    if os.environ.get("PVK_LIB"):
        out.append(Path(os.environ["PVK_LIB"]))

    here = Path(__file__).resolve().parents[3]   # python/pvsim/kernel → 项目根
    sysroot = Path(sys.executable).parent

    is_win  = sys.platform.startswith("win")
    is_mac  = sys.platform == "darwin"

    ext = ".dll" if is_win else (".dylib" if is_mac else ".so")

    for base in (here / "csrc" / "build", here / "csrc", sysroot):
        for name in (f"pvk{ext}", f"libpvk{ext}", f"pvk.0{ext}", f"pvk-0{ext}"):
            out.append(base / name)
            out.append(base / "Release" / name)
            out.append(base / "Debug"   / name)

    return out


def locate_and_load() -> ct.CDLL:
    """搜索并加载 libpvk。 找不到或加载失败抛 PvkDLError。"""
    tried = []
    for p in _candidate_paths():
        tried.append(str(p))
        if not p.exists():
            continue
        try:
            lib = ct.CDLL(str(p))
            return lib
        except OSError as e:
            raise PvkDLError(
                f"Found candidate {p} but failed to load: {e}"
            ) from e
    raise PvkDLError(
        f"libpvk not found.  Tried:\n  " + "\n  ".join(tried)
        + "\nSet PVK_LIB to explicit path or build the library first."
    )


def bind_signatures(lib: ct.CDLL) -> None:
    """给 lib 的所有导出函数声明 argtypes/restype。 必须和 pvk.h 签名一致。"""

    # ── pvk.h ──────────────────────────────────────────────────────────
    lib.pvk_abi_version.restype = ct.c_int

    lib.pvk_create.argtypes  = [ct.c_void_p,   # pvk_log_fn (can be NULL)
                                ct.c_void_p]   # user_data (can be NULL)
    lib.pvk_create.restype    = ct.c_void_p   # pvk_ctx*

    lib.pvk_destroy.argtypes  = [ct.c_void_p]
    lib.pvk_destroy.restype   = None

    lib.pvk_last_error.argtypes  = [ct.c_void_p, ct.c_char_p, ct.c_int]
    lib.pvk_last_error.restype   = ct.c_int

    # ── mesh.h ─────────────────────────────────────────────────────────
    lib.pvk_build_mesh.argtypes  = [ct.c_void_p, ct.POINTER(st.Device)]
    lib.pvk_build_mesh.restype   = ct.c_int

    lib.pvk_node_count.argtypes  = [ct.c_void_p]
    lib.pvk_node_count.restype   = ct.c_int

    lib.pvk_get_x.argtypes       = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_int]
    lib.pvk_get_x.restype        = ct.c_int

    # ── physics.h ──────────────────────────────────────────────────────
    lib.pvk_get_field.argtypes   = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_double), ct.c_int]
    lib.pvk_get_field.restype    = ct.c_int

    # ── optics.h ───────────────────────────────────────────────────────
    lib.pvk_optics_compute.argtypes = [
        ct.c_void_p,
        ct.POINTER(st.Device),
        ct.POINTER(st.OpticsIn),
        ct.POINTER(ct.c_double),   # out_absorb_spectrum (可 NULL)
        ct.POINTER(ct.c_double),   # out_gen_profile     (可 NULL)
    ]
    lib.pvk_optics_compute.restype  = ct.c_int

    # ── solver.h ────────────────────────────────────────────────────────
    lib.pvk_set_solver_opts.argtypes = [ct.c_void_p, ct.POINTER(st.SolverOpts)]
    lib.pvk_set_solver_opts.restype  = ct.c_int

    lib.pvk_solve_dc.argtypes = [
        ct.c_void_p,
        ct.POINTER(st.Device),
        ct.POINTER(st.OpticsIn),
        ct.POINTER(st.Bias),
        ct.POINTER(st.SolveInfo),
    ]
    lib.pvk_solve_dc.restype  = ct.c_int

    # sweep callback 类型
    CB_TYPE = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(st.SolveInfo))

    lib.pvk_sweep_dc.argtypes = [
        ct.c_void_p,
        ct.POINTER(st.Device),
        ct.POINTER(st.OpticsIn),
        ct.POINTER(st.Bias),
        ct.c_double, ct.c_double, ct.c_int,
        ct.c_void_p,     # pvk_point_cb —— 可 NULL
        ct.c_void_p,
        ct.POINTER(st.SolveInfo),
    ]
    lib.pvk_sweep_dc.restype  = ct.c_int


# ───────────────────────────────────────────────────────────────────────
# PvkLib — 顶层封装，隐藏 argtypes/restype 的底层细节
# ───────────────────────────────────────────────────────────────────────

class PvkLib:
    """持有已加载 libpvk 的句柄，提供高级 Python 友好 API。

    用法::

        lib = PvkLib()
        ctx = lib.create()
        lib.solve_dc(ctx, dev, opt, bias, out)
        lib.destroy(ctx)
    """

    def __init__(self, path: Optional[str] = None):
        if path:
            self._lib = ct.CDLL(path)
        else:
            self._lib = locate_and_load()
        bind_signatures(self._lib)
        self._verify_abi()

    # ── internal ──────────────────────────────────────────────────
    def _verify_abi(self) -> None:
        actual = self._lib.pvk_abi_version()
        if actual != PVK_ABI_VERSION:
            raise ABIError(
                f"C library reports ABI v{actual}, Python expects v{PVK_ABI_VERSION}. "
                f"Rebuild C library or update python/pvsim/kernel/structs.py."
            )

    def _last_err_str(self, ctx: int) -> str:
        buf = ct.create_string_buffer(512)
        rc = self._lib.pvk_last_error(ctx, buf, 512)
        return buf.value.decode("utf-8") if rc > 0 else "PVK_OK"

    def _check(self, ctx: int, rc: int, op: str) -> None:
        if rc != 0:
            msg = self._last_err_str(ctx)
            code_name = ERROR_CODES.get(rc, f"PVK_UNKNOWN({rc})")
            raise RuntimeError(f"pvk_{op} → {code_name}: {msg}")

    # ── lifecycle ────────────────────────────────────────────────
    def create(self) -> int:
        """返回 ctx handle（int 形式，ctypes 自动）。"""
        ctx = self._lib.pvk_create(None, None)   # log_cb=NULL, user=NULL
        if ctx is None or ctx == 0:
            raise RuntimeError("pvk_create returned NULL ctx")
        return int(ctx)

    def destroy(self, ctx: int) -> None:
        self._lib.pvk_destroy(ctx)

    # ── mesh ─────────────────────────────────────────────────────
    def build_mesh(self, ctx: int, dev: st.Device) -> None:
        rc = self._lib.pvk_build_mesh(ctx, ct.byref(dev))
        self._check(ctx, rc, "build_mesh")

    def node_count(self, ctx: int) -> int:
        return int(self._lib.pvk_node_count(ctx))

    def get_x(self, ctx: int, n: int) -> list[float]:
        out = (ct.c_double * n)()
        rc = self._lib.pvk_get_x(ctx, out, n)
        self._check(ctx, rc, "get_x")
        return list(out)

    # ── field getter ──────────────────────────────────────────────
    def get_field(self, ctx: int, field_id: int, n: int) -> list[float]:
        out = (ct.c_double * n)()
        rc = self._lib.pvk_get_field(ctx, field_id, out, n)
        self._check(ctx, rc, "get_field")
        return list(out)

    # ── solver opts ───────────────────────────────────────────────
    def set_solver_opts(self, ctx: int, opts: st.SolverOpts) -> None:
        rc = self._lib.pvk_set_solver_opts(ctx, ct.byref(opts))
        self._check(ctx, rc, "set_solver_opts")

    # ── optics ────────────────────────────────────────────────────
    def optics_compute(self, ctx: int, dev: st.Device, opt: st.OpticsIn) -> None:
        rc = self._lib.pvk_optics_compute(
            ctx, ct.byref(dev), ct.byref(opt),
            None, None
        )
        self._check(ctx, rc, "optics_compute")

    # ── solve ─────────────────────────────────────────────────────
    def solve_dc(self, ctx: int, dev: st.Device, opt: st.OpticsIn,
                 bias: st.Bias) -> st.SolveInfo:
        info = st.SolveInfo()
        rc = self._lib.pvk_solve_dc(ctx, ct.byref(dev), ct.byref(opt),
                                    ct.byref(bias), ct.byref(info))
        self._check(ctx, rc, "solve_dc")
        return info

    def sweep_dc(self, ctx: int, dev: st.Device, opt: st.OpticsIn,
                 bias: st.Bias, v_start: float, v_stop: float,
                 n_points: int) -> list[st.SolveInfo]:
        arr = (st.SolveInfo * n_points)()
        rc = self._lib.pvk_sweep_dc(
            ctx, ct.byref(dev), ct.byref(opt), ct.byref(bias),
            v_start, v_stop, n_points,
            None, None,      # cb / cb_user —— 首期 Python 端不做进度回调
            arr,
        )
        self._check(ctx, rc, "sweep_dc")
        # 手动拷贝避免 ctypes array → list 自动转换可能的内存问题
        return [st.SolveInfo.from_buffer_copy(arr[i]) for i in range(n_points)]
