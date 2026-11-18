"""Simulator 端到端测试：ProblemSpec → 内核 → SimResult → plan 串联。

需要先构建 libpvk（csrc/build）。运行环境需要 MSYS2 ucrt64 在 PATH 中。

Run: python python/tests/test_simulator.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pvsim.param import (
    DefectSpec,
    ThicknessScan,
    IntensitySweep,
    BiasSweep,
    SpecError,
    default_problem_spec,
)
from pvsim.solver import JVData, Simulator

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. JVData 数学（不需要内核）
# ---------------------------------------------------------------------------

section("1. JVData.from_points")

# 理想化 J-V：Jsc=20 mA/cm²，Voc=1.25 V
jv = JVData.from_points(V=[0.0, 0.5, 1.0, 1.5],
                        J=[-20.0, -15.0, -5.0, 5.0],
                        pin_mW_cm2=100.0)
check("Jsc = 20", abs(jv.Jsc - 20.0) < 1e-12, f"got {jv.Jsc}")
check("Voc 线性插值 = 1.25", abs(jv.Voc - 1.25) < 1e-12, f"got {jv.Voc}")
check("Pmax = 7.5 mW/cm²", abs(jv.Pmax_mW_cm2 - 7.5) < 1e-12, f"got {jv.Pmax_mW_cm2}")
check("FF = 0.3", abs(jv.FF - 0.3) < 1e-12, f"got {jv.FF}")
check("PCE = 7.5 %", abs(jv.PCE - 7.5) < 1e-12, f"got {jv.PCE}")

# 单调递减但不过零 → Voc 必须 NaN，不能拿终点冒充
jv2 = JVData.from_points(V=[0.0, 0.5, 1.0], J=[-20.0, -10.0, -1.0], pin_mW_cm2=100.0)
check("不过零 → Voc NaN", math.isnan(jv2.Voc))
check("不过零 → FF NaN", math.isnan(jv2.FF))
check("不过零 → PCE 仍可算", abs(jv2.PCE - 5.0) < 1e-12, f"got {jv2.PCE}")

# 全零（当前内核 stub 的输出）
jv3 = JVData.from_points(V=[0.0, 1.0, 1.8], J=[0.0, 0.0, 0.0], pin_mW_cm2=100.0)
check("全零 → Jsc = 0", jv3.Jsc == 0.0)
check("全零 → Voc NaN", math.isnan(jv3.Voc))
check("全零 → FF NaN", math.isnan(jv3.FF))
check("全零 → PCE NaN", math.isnan(jv3.PCE))

# 乱序输入也应按 V 排序处理
jv4 = JVData.from_points(V=[1.5, 0.0, 1.0, 0.5],
                         J=[5.0, -20.0, -5.0, -15.0], pin_mW_cm2=100.0)
check("乱序 V 仍得 Voc=1.25", abs(jv4.Voc - 1.25) < 1e-12, f"got {jv4.Voc}")
check("summary 可读", "Jsc=20" in jv.summary(), jv.summary())

# 断档保护：Voc 附近有未收敛点、两侧收敛点相隔 >2 个步长 → 不许插值
jv_gap = JVData.from_points(
    V=[0.0, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.44],
    J=[-20.0, -10.0, -5.3, -5.3, -5.3, -5.3, -5.3, 174.0],
    pin_mW_cm2=100.0,
    converged=[True, True, True, False, False, False, False, True])
check("过零断档（跨 6 步）→ Voc NaN", math.isnan(jv_gap.Voc), f"got {jv_gap.Voc}")
check("过零断档 → FF NaN", math.isnan(jv_gap.FF))
check("过零断档 → Jsc 仍按收敛点算", abs(jv_gap.Jsc - 20.0) < 1e-12, f"got {jv_gap.Jsc}")

# 相邻收敛点（间隔 1 个步长）照常插值
jv_ok = JVData.from_points(V=[0.0, 1.30, 1.32, 1.34],
                           J=[-20.0, -10.0, -5.0, 5.0],
                           pin_mW_cm2=100.0,
                           converged=[True, True, True, True])
check("收敛点相邻仍得 Voc=1.33", abs(jv_ok.Voc - 1.33) < 1e-12, f"got {jv_ok.Voc}")


# ---------------------------------------------------------------------------
# 2. Simulator.run 真实内核
# ---------------------------------------------------------------------------

section("2. Simulator.run (kernel)")

spec = default_problem_spec()
sim = Simulator()
res = sim.run(spec, run_id="smoke")

check("run_id 使用显式值", res.run_id == "smoke")
check("spec 溯源一致", res.spec is spec)
check("点数 = 37", len(res.points) == 37, f"got {len(res.points)}")
check("首点 V=0", abs(res.points[0].V) < 1e-12)
check("末点 V=1.8", abs(res.points[-1].V - 1.8) < 1e-12)
check("全部收敛", res.converged)
check("V/J 便捷属性长度一致", res.V.size == res.J.size == 37)

names = tuple(spec.output.fields)
check("场量名与 OutputSpec 一致", res.fields.names == names, str(res.fields.names))
check("n_node 合理", 80 <= res.fields.n_node <= 200, f"got {res.fields.n_node}")
check("每张场表形状 = (37, n_node)",
      all(res.fields[n].shape == (37, res.fields.n_node) for n in names),
      str({n: res.fields[n].shape for n in names}))
check("x 单调递增", bool(np.all(np.diff(res.fields.x_nm) > 0)))
check("x 末端 ≈ 1020 nm", abs(res.fields.x_nm[-1] - 1020.0) / 1020.0 < 0.1,
      f"got {res.fields.x_nm[-1]}")

check("Jsc > 0（光照下有光电流）", res.jv.Jsc > 0.0, f"got {res.jv.Jsc}")
check("Voc 落在 (0, 1.8) 内", math.isfinite(res.jv.Voc) and 0.0 < res.jv.Voc < 1.8,
      f"got {res.jv.Voc}")
check("FF 落在 (0, 1) 内", math.isfinite(res.jv.FF) and 0.0 < res.jv.FF < 1.0,
      f"got {res.jv.FF}")
check("PCE > 0", math.isfinite(res.jv.PCE) and res.jv.PCE > 0.0, f"got {res.jv.PCE}")

psi = res.fields["PSI"]
check("PSI 全有限", bool(np.all(np.isfinite(psi))))
d_psi = psi[-1, -1] - psi[-1, 0]
check("末点阳极电位高于阴极（内置电势 + 外加偏压）",
      0.05 < d_psi < 1.8, f"got {d_psi}")

slice0 = res.fields.at(0)
check("fields.at() 返回全部场量", set(slice0.keys()) == set(names))
check("fields.at() 剖面长度 = n_node",
      all(v.size == res.fields.n_node for v in slice0.values()))

check("不改动输入 spec", spec.sweep.n_points == 37
      and next(l for l in spec.device.layers if l.name == "MAPbI3").thick_nm == 500.0)

check("summary 含 run_id", res.run_id in res.summary(), res.summary())


# ---------------------------------------------------------------------------
# 3. 警告级配置：不中止，走 on_log
# ---------------------------------------------------------------------------

section("3. warnings via hook")


class RecordingHook:
    def __init__(self):
        self.points = []
        self.logs = []

    def on_point(self, index, total, point):
        self.points.append((index, total, point.V))

    def on_log(self, level, message):
        self.logs.append((level, message))


spec_warn = replace(spec, device=replace(spec.device, layers=tuple(
    replace(l, material="MAPBI3") if l.name == "MAPbI3" else l
    for l in spec.device.layers
)))
hook = RecordingHook()
res_warn = sim.run(spec_warn, hook=hook, run_id="warn_case")
check("warning 不中止求解", res_warn.converged)
check("on_point 被调用 37 次", len(hook.points) == 37, f"got {len(hook.points)}")
check("on_point index 递增",
      [p[0] for p in hook.points] == list(range(37)))
check("on_log 收到 warning", any(2 == lvl for lvl, _ in hook.logs),
      str(hook.logs[:3]))
check("on_log 消息含 material", any("material" in msg for _, msg in hook.logs))


# ---------------------------------------------------------------------------
# 4. 致命配置：碰内核前抛 SpecError
# ---------------------------------------------------------------------------

section("4. fatal spec")

bad = replace(spec, device=replace(spec.device, layers=()))
try:
    sim.run(bad)
    check("空层栈抛 SpecError", False, "should have raised")
except SpecError as e:
    check("空层栈抛 SpecError", True)
    check("detail 里指出 layers empty",
          any("empty" in d for d in e.details), str(e.details))


# ---------------------------------------------------------------------------
# 5. run_plan：把 SweepPlan 串起来
# ---------------------------------------------------------------------------

section("5. Simulator.run_plan")

plan = ThicknessScan(layer_name="MAPbI3", thick_nm_values=(300, 500, 700))
runs = sim.run_plan(plan, spec)
check("ThicknessScan → 3 个 PlanRun", len(runs) == 3, f"got {len(runs)}")
check("label 与 step 对应",
      [r.label for r in runs] == ["MAPbI3=300 nm", "MAPbI3=500 nm", "MAPbI3=700 nm"],
      str([r.label for r in runs]))
check("每个 PlanRun 都有完整结果",
      all(isinstance(r.result, type(res)) and r.result.converged for r in runs))
check("各步 n_node 各自合法",
      all(r.result.fields.n_node >= 3 for r in runs))
check("base spec 未被改动",
      next(l for l in spec.device.layers if l.name == "MAPbI3").thick_nm == 500.0)

runs_i = sim.run_plan(IntensitySweep(sun_values=(1.0, 0.5)), spec)
check("IntensitySweep → 2 个 PlanRun", len(runs_i) == 2)
check("0.5 sun 的 spec 强度 = 50",
      abs(runs_i[1].result.spec.optics.intensity_mW_cm2 - 50.0) < 1e-9)

runs_b = sim.run_plan(BiasSweep(v_start=0.2, v_stop=1.0, n_points=5), spec)
check("BiasSweep → 1 个 PlanRun", len(runs_b) == 1)
check("BiasSweep 生效", abs(runs_b[0].result.points[-1].V - 1.0) < 1e-12,
      f"got {runs_b[0].result.points[-1].V}")


# ---------------------------------------------------------------------------
# 6. 缺陷 → effective tau 进入内核路径
# ---------------------------------------------------------------------------

section("6. defects reach kernel")

spec_def = replace(spec, device=replace(spec.device, layers=tuple(
    replace(l, defects=(DefectSpec(name="mid", Et_eV=0.8, Nt_cm3=1e16),))
    if l.name == "MAPbI3" else l
    for l in spec.device.layers
)))
pvk_layer = next(l for l in spec_def.device.layers if l.name == "MAPbI3")
check("effective tau 被缺陷拉低", pvk_layer.effective_tau_n_s < pvk_layer.taun_s,
      f"eff={pvk_layer.effective_tau_n_s}, explicit={pvk_layer.taun_s}")

res_def = sim.run(spec_def, run_id="defect_case")
check("带缺陷 spec 可求解", res_def.converged)


# ---------------------------------------------------------------------------
# 7. 网格与接触层回归（A′）
# ---------------------------------------------------------------------------

section("7. 网格与接触层回归")


def _sweep_spec(nodes, refine):
    """0→1.8 V 全 37 点：既覆盖接触层隐式处理修复前的失效区（V>1.1），
    也守住整段 continuation 的收敛与网格无关性。"""
    return replace(spec,
                   device=replace(spec.device, nodes=nodes, interface_refine=refine),
                   sweep=replace(spec.sweep, v_start=0.0, v_stop=1.8, n_points=37))


spec_dark = replace(spec,
                    device=replace(spec.device, nodes=101, interface_refine=1.0),
                    optics=replace(spec.optics, intensity_mW_cm2=0.0),
                    sweep=replace(spec.sweep, v_start=0.0, v_stop=0.0, n_points=1))
res_dark = Simulator().run(spec_dark)
check("暗态 refine=1 收敛", res_dark.points[0].converged)
check("暗态 refine=1 J≈0", abs(res_dark.points[0].J_mA_cm2) < 1e-6,
      f"got {res_dark.points[0].J_mA_cm2}")

res_fine = Simulator().run(_sweep_spec(201, 2.0))
check("201/refine=2 全 37 点收敛", res_fine.converged,
      f"{sum(p.converged for p in res_fine.points)}/37")
check("201/refine=2 点数 = 37", len(res_fine.points) == 37,
      str(len(res_fine.points)))
check("201/refine=2 Jsc > 20", res_fine.jv.Jsc > 20.0, f"got {res_fine.jv.Jsc}")
check("201/refine=2 Voc 合理", 1.05 < res_fine.jv.Voc < 1.10,
      f"got {res_fine.jv.Voc}")
check("201/refine=2 FF 合理", 0.5 < res_fine.jv.FF < 0.6, f"got {res_fine.jv.FF}")
_j = [p.J_mA_cm2 for p in res_fine.points]
check("201/refine=2 J-V 单调递增",
      all(_j[i + 1] >= _j[i] - 1e-3 for i in range(len(_j) - 1)), str(_j))

res_coarse = Simulator().run(_sweep_spec(101, 1.0))
check("101/refine=1 全 37 点收敛", res_coarse.converged,
      f"{sum(p.converged for p in res_coarse.points)}/37")
d_jsc = abs(res_fine.jv.Jsc - res_coarse.jv.Jsc) / res_coarse.jv.Jsc
d_voc = abs(res_fine.jv.Voc - res_coarse.jv.Voc)
d_ff = abs(res_fine.jv.FF - res_coarse.jv.FF)
check("Jsc 网格差 < 0.5%", d_jsc < 5e-3, f"{d_jsc:.2e}")
check("Voc 网格差 < 1 mV", d_voc < 1e-3, f"{d_voc:.2e}")
check("FF 网格差 < 0.003", d_ff < 3e-3, f"{d_ff:.2e}")


# ---------------------------------------------------------------------------
# 9. nk 光学：α = 4πk/λ（k=0 应不产生光生载流子）
# ---------------------------------------------------------------------------

section("9. nk optics")

with tempfile.TemporaryDirectory() as tmp:
    nk_dir = Path(tmp)
    # MAPbI3 的 k 全为 0 → 吸收层不吸光；FTO 给个折射率供前表面反射
    (nk_dir / "MAPbI3.txt").write_text(
        "3\n0.3 2.5 0\n0.8 2.3 0\n1.2 2.2 0\n", encoding="utf-8")
    (nk_dir / "FTO.txt").write_text(
        "3\n0.3 1.9 0\n0.8 1.8 0\n1.2 1.7 0\n", encoding="utf-8")
    spec_nk = replace(spec, optics=replace(spec.optics, nk_dir=str(nk_dir)))
    res_nk = Simulator().run(spec_nk)
    check("k=0 时 Jsc 接近 0", abs(res_nk.jv.Jsc) < 0.5, f"Jsc={res_nk.jv.Jsc}")




print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
