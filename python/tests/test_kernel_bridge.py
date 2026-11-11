"""端到端冒烟：ProblemSpec → ctypes Device/OpticsIn/Bias → PvkLib.solve_dc → 读场

Run: python python/tests/test_kernel_bridge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pvsim.param import default_problem_spec
from pvsim.kernel.abi import PvkLib, PVK_ABI_VERSION
from pvsim.kernel import structs as st

passed = 0
failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  ✓ {name}")
    else:
        failed += 1; print(f"  ✗ {name} — {detail}")

# ── 1. 加载 DLL + ABI 校验 ─────────────────────────────────────────
print("=== 1. Load + ABI check ===")
lib = PvkLib()
check("ABI == 1", lib._lib.pvk_abi_version() == 1)

# ── 2. ProblemSpec → ctypes 结构 ────────────────────────────────────
print("\n=== 2. spec → ctypes ===")
spec = default_problem_spec()
spec.validate()   # 先让 Python 侧确认合理

dev, layers = st.device_from_spec(spec.device)
check("device.n_layers == 5", dev.n_layers == 5)

opt = st.optics_from_spec(spec.optics)
check("opt.n_wl > 0", opt.n_wl > 0)
check("opt.intensity == 100", abs(opt.intensity_mW_cm2 - 100.0) < 1e-9)

sopts = st.solver_opts_from_spec(spec.sweep.solver_opts)
check("solver_opts.method == NEWTON_WITH_WARMSTART", sopts.method == 2)

bias_v0 = st.bias_from_spec(spec.sweep, v=0.0)
bias_voc = st.bias_from_spec(spec.sweep, v=1.8)
check("v=0 anode_V == 0", abs(bias_v0.anode_V - 0.0) < 1e-9)
check("v=1.8 anode_V == 1.8", abs(bias_voc.anode_V - 1.8) < 1e-9)

# ── 3. build_mesh + 检查网格 ────────────────────────────────────────
print("\n=== 3. mesh build ===")
ctx = lib.create()
lib.set_solver_opts(ctx, sopts)
lib.build_mesh(ctx, dev)
n_node = lib.node_count(ctx)
check(f"node_count in reasonable range (expect ~101)", 80 <= n_node <= 200, f"got {n_node}")

xs = lib.get_x(ctx, n_node)
check("x[0] ≈ 0", abs(xs[0]) < 1e-20)
check("x[-1] ≈ 1020 nm in cm", abs(xs[-1] - 1020e-7) / 1020e-7 < 0.1, f"got {xs[-1]}")

# ── 4. solve_dc at V=0 (short-circuit) ──────────────────────────────
print("\n=== 4. solve_dc at V=0 ===")
info0 = lib.solve_dc(ctx, dev, opt, bias_v0)
check("converged", info0.converged == 1)
check("n matches mesh", info0.n == n_node)
# 光照短路：J 为 anode→cathode 方向，光电流使其为负
check("J_total < 0 at short circuit (photocurrent)",
      info0.J_total_mA_cm2 < 0.0 and info0.J_total_mA_cm2 > -200.0,
      f"got {info0.J_total_mA_cm2}")
check("iterations >= 1", info0.iterations >= 1, f"got {info0.iterations}")

# ── 5. solve_dc at V=1.8 ────────────────────────────────────────────
print("\n=== 5. solve_dc at V=1.8 ===")
info1 = lib.solve_dc(ctx, dev, opt, bias_voc)
check("converged at 1.8V", info1.converged == 1)

# ── 6. 读场量 ───────────────────────────────────────────────────────
print("\n=== 6. field readback ===")
psi = lib.get_field(ctx, st.FIELD_ID["PSI"], n_node)
check("PSI all finite", all(np.isfinite(psi)))
# 1.8 V 正偏：内置电势 + 外加偏压，阳极(末层)电位高于阴极(首层)，
# 但差值小于 1.8 V（一部分被内置电势抵消）
psi_bias_applied = psi[-1] - psi[0]
check(f"PSI 阳极高于阴极且差值 < 1.8V",
      0.05 < psi_bias_applied < 1.8,
      f"两端差 {psi_bias_applied:.3f}V")

n_conc = lib.get_field(ctx, st.FIELD_ID["N"], n_node)
p_conc = lib.get_field(ctx, st.FIELD_ID["P"], n_node)
check("N concentration positive & finite", all(x > 0 and np.isfinite(x) for x in n_conc))
check("P concentration positive & finite", all(x > 0 and np.isfinite(x) for x in p_conc))
# 强掺杂层里的少数载流子可以低到 1e-30 量级
check("N has reasonable magnitude", all(0.0 < x < 1e25 for x in n_conc),
      f"range [{min(n_conc):.2e}, {max(n_conc):.2e}]")

# G_OPT 应该非零（因为光强 100 mW/cm²）
g_opt = lib.get_field(ctx, st.FIELD_ID["G_OPT"], n_node)
check("G_OPT non-zero (light on)", sum(g_opt) > 0, f"sum={sum(g_opt):.2e}")

# ── 7. sweep_dc 扫描 ────────────────────────────────────────────────
print("\n=== 7. sweep_dc ===")
infos = lib.sweep_dc(ctx, dev, opt, bias_v0,
                     v_start=0.0, v_stop=1.8, n_points=10)
check("sweep returned 10 points", len(infos) == 10)
check("all converged", all(i.converged == 1 for i in infos))
check("v_applied monotonic",
      all(infos[i].v_applied <= infos[i+1].v_applied + 1e-9 for i in range(len(infos)-1)),
      [f"{i.v_applied:.3f}" for i in infos])

# ── 8. clean up ─────────────────────────────────────────────────────
print("\n=== 8. destroy ===")
lib.destroy(ctx)
check("destroy OK", True)

# ── summary ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
