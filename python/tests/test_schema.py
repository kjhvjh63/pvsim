"""Smoke tests for the ProblemSpec dataclass tree.

Run:  python -m pytest python/tests/test_schema.py -v
Or:   python python/tests/test_schema.py
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

# 让 python/pvsim 可被 import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.param.schema import (   # noqa: E402
    DeviceSpec,
    LayerSpec,
    DefectSpec,
    OpticsSpec,
    OutputSpec,
    ProblemSpec,
    SolverOpts,
    SweepSpec,
    MaterialSet,
    SpecError,
    VALID_FIELDS,
    PlanStep,
    SweepPlan,
    BiasSweep,
    IntensitySweep,
    ThicknessScan,
    DopingScan,
    DefectScan,
    default_problem_spec,
    mapbi3_stack,
)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")
        traceback.print_stack(limit=2)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. LayerSpec
# ---------------------------------------------------------------------------

section("1. LayerSpec basics")

good_layer = LayerSpec(
    name="MAPbI3", material=None, thick_nm=500.0,
    Eg_eV=1.6, chi_eV=3.93, eps_r=24.0,
)
check("default Nc/Nv present", good_layer.Nc_cm3 == 1e19)
check("thick_cm derived", abs(good_layer.thick_cm - 500e-7) < 1e-15)

notes = good_layer.validate(0)
check("good layer validate → 0 warnings", len(notes) == 0, str(notes))

bad = LayerSpec(name="", thick_nm=-1, Eg_eV=-0.5)
notes = bad.validate(0)
check("bad layer catches 3 issues", len(notes) >= 3, str(notes))


# ---------------------------------------------------------------------------
# 2. DeviceSpec
# ---------------------------------------------------------------------------

section("2. DeviceSpec")

stack = mapbi3_stack()
check("stack has 5 layers", len(stack.layers) == 5)
check("first layer is_contact", stack.layers[0].is_contact)
check("last layer is_contact", stack.layers[-1].is_contact)
check("absorber present", len(stack.absorber_layers) == 1)
check("absorber is MAPbI3", stack.absorber_layers[0].name == "MAPbI3")
check("total_thick_nm ≈ 1020", abs(stack.total_thick_nm - 1020.0) < 1e-9)

notes = stack.validate()
check("mapbi3_stack validates clean", len(notes) == 0, str(notes))

# Bad device: empty layers
empty = DeviceSpec(layers=(), T_K=300.0, nodes=101)
notes = empty.validate()
check("empty device catches issue", any("empty" in n for n in notes))

# Bad device: first layer not contact
nocontact_layers = (LayerSpec(name="foo", thick_nm=10, Eg_eV=1),)
nocontact = DeviceSpec(layers=nocontact_layers, nodes=5)
notes = nocontact.validate()
check("first non-contact caught", any("first layer" in n for n in notes))


# ---------------------------------------------------------------------------
# 3. OpticsSpec
# ---------------------------------------------------------------------------

section("3. OpticsSpec")

opt = OpticsSpec.default_am15g(n_pts=50)
check("wl/flux same length", opt.wl_nm.size == opt.flux.size == 50)
check("wl all positive", opt.wl_nm.min() > 0)
check("opt.validate clean", len(opt.validate()) == 0)

# 坏光谱：wl/flux 长度不符
bad_opt = OpticsSpec(
    wl_nm=np.array([400, 500, 600.0]),
    flux=np.array([1.0, 2.0]),
)
notes = bad_opt.validate()
check("wl/flux size mismatch caught", any("≠" in n for n in notes), str(notes))


# ---------------------------------------------------------------------------
# 4. SolverOpts
# ---------------------------------------------------------------------------

section("4. SolverOpts")

so = SolverOpts()
check("default NEWTON_WITH_WARMSTART", so.method == "NEWTON_WITH_WARMSTART")
check("damping ∈ (0,1]", so.damping == 1.0)
check("voltage step ordering OK",
      so.v_min_step <= so.v_init_step <= so.v_max_step)
check("validate clean", len(so.validate()) == 0)

# 坏：method 拼错
bad_so = SolverOpts(method="GUNNEL")
notes = bad_so.validate()
check("bad method caught", any("method" in n for n in notes))

# 坏：v_min > v_max
bad_so2 = SolverOpts(v_min_step=0.1, v_max_step=0.01)
notes = bad_so2.validate()
check("v_min>v_max caught", len(notes) >= 1)


# ---------------------------------------------------------------------------
# 5. SweepSpec
# ---------------------------------------------------------------------------

section("5. SweepSpec")

ss = SweepSpec.jv_sweep(v_stop=1.8, n_points=37)
check("jv_sweep defaults", ss.v_start == 0.0 and ss.v_stop == 1.8 and ss.n_points == 37)
check("validate clean", len(ss.validate()) == 0)

sp = SweepSpec.single_point(1.23)
check("single_point factory", sp.n_points == 1 and sp.v_start == sp.v_stop == 1.23)

# 坏：多点点却 v_start > v_stop
bad_ss = SweepSpec(v_start=1.0, v_stop=0.0, n_points=10)
notes = bad_ss.validate()
check("v_start>v_stop with multi-point caught", any("≤" in n for n in notes), str(notes))


# ---------------------------------------------------------------------------
# 6. MaterialSet & OutputSpec
# ---------------------------------------------------------------------------

section("6. MaterialSet + OutputSpec")

ms = MaterialSet()
check("empty mat validates", len(ms.validate()) == 0)
ms2 = MaterialSet(overrides={"MAPBI3": {"Eg_eV": 1.58, "eps_r": 24.0}})
check("override get works", ms2.get("MAPBI3", "Eg_eV") == 1.58)
check("override missing field → None", ms2.get("MAPBI3", "nonexistent") is None)

os_ = OutputSpec()
check("X in default fields", "X" in os_.fields)
check("validate clean", len(os_.validate()) == 0)

bad_os = OutputSpec(fields=("X", "PSI", "UNKNOWN_FIELD"))
notes = bad_os.validate()
check("unknown field caught", any("unknown" in n for n in notes))


# ---------------------------------------------------------------------------
# 7. ProblemSpec — whole spec
# ---------------------------------------------------------------------------

section("7. ProblemSpec: default_problem_spec")

spec = default_problem_spec()
check("type", isinstance(spec, ProblemSpec))
check("sweep n_points=37", spec.sweep.n_points == 37)
check("device has MAPbI3", any(l.name == "MAPbI3" for l in spec.device.layers))

# validate — 默认构造器应该全绿
try:
    spec.validate()
    check("default_problem_spec.validate() → OK", True)
except SpecError as e:
    # 收集 warnings 但不算 fatal
    fatal = [d for d in e.details if any(k in d for k in ("must be", "out of", "empty", "unknown"))]
    check("default_problem_spec.validate() → OK (only warnings)",
          len(fatal) == 0, f"fatal={fatal}, all details={e.details}")

# 制造一个坏 ProblemSpec
bad_spec = ProblemSpec(
    device=DeviceSpec(layers=(), nodes=5),
    optics=OpticsSpec.default_am15g(n_pts=10),
    sweep=SweepSpec.jv_sweep(v_stop=1.0, n_points=3),
)
try:
    bad_spec.validate()
    check("bad spec raises SpecError", False, "should have raised")
except SpecError as e:
    check("bad spec raises SpecError", True, str(e))


# ---------------------------------------------------------------------------
# 8. frozen + replace
# ---------------------------------------------------------------------------

section("8. frozen semantics")

try:
    # 尝试修改 frozen 字段 → FrozenInstanceError
    replace(spec.device, layers=())
    check("replace works", True)
except Exception as exc:
    check("replace works", False, f"unexpected: {exc}")

try:
    spec.device.layers = ()   # type: ignore[misc]
    check("assignment should have failed", False, "FrozenInstanceError expected")
except FrozenInstanceError:
    check("assignment blocked (frozen=True)", True)
except Exception as exc:
    check("assignment blocked", False, f"got {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 9. JSON round-trip
# ---------------------------------------------------------------------------

section("9. JSON round-trip")

spec = default_problem_spec()
d = spec.to_dict()
check("to_dict has 5 top keys", set(d.keys()) == {"device", "optics", "sweep", "material", "output"})

j = spec.to_json()
check("json.dumps succeeds", len(j) > 100)

spec2 = ProblemSpec.from_json(j)
check("from_json recovers layers", len(spec2.device.layers) == len(spec.device.layers))
check("from_json recovers absorber Eg",
      spec2.device.absorber_layers[0].Eg_eV == spec.device.absorber_layers[0].Eg_eV)
check("from_json recovers opt.wl_nm.size",
      spec2.optics.wl_nm.size == spec.optics.wl_nm.size)
check("from_json recovers sweep", spec2.sweep.n_points == spec.sweep.n_points)
check("from_json recovers output fields", spec2.output.fields == spec.output.fields)


# ---------------------------------------------------------------------------
# 10. VALID_FIELDS
# ---------------------------------------------------------------------------

section("10. VALID_FIELDS")

check("PSI, N, P present", all(k in VALID_FIELDS for k in ("PSI", "N", "P")))
check("JTOTAL present", "JTOTAL" in VALID_FIELDS)
check("G_OPT present", "G_OPT" in VALID_FIELDS)
# 所有 default OutputSpec 字段都应该在 VALID_FIELDS 里
default_fields = OutputSpec().fields
check("default OutputSpec fields all valid", all(f in VALID_FIELDS for f in default_fields))


# ---------------------------------------------------------------------------
# 11. DefectSpec
# ---------------------------------------------------------------------------

section("11. DefectSpec")

d0 = DefectSpec(name="mid", Et_eV=0.8)
# tau = 1/(1e-15 * 1e7 * 1e15) = 1e-7 s
check("default tau_n = 1e-7 s", abs(d0.tau_n_s - 1e-7) <= 1e-9 * 1e-7,
      f"got {d0.tau_n_s}")
check("default tau_p = 1e-7 s", abs(d0.tau_p_s - 1e-7) <= 1e-9 * 1e-7,
      f"got {d0.tau_p_s}")

mg = DefectSpec.midgap(1.6)
check("midgap Et = 0.8", abs(mg.Et_eV - 0.8) < 1e-12)
check("midgap validate clean", len(mg.validate(0, Eg_eV=1.6)) == 0, str(mg.validate(0, 1.6)))

bad_d = DefectSpec(Et_eV=2.5, sigma_n_cm2=0.0, Nt_cm3=0.0, kind="FOO")
notes = bad_d.validate(0, Eg_eV=1.6)
check("Et outside gap caught", any("out of" in n for n in notes), str(notes))
check("sigma_n <= 0 caught", any("sigma_n_cm2" in n for n in notes), str(notes))
check("Nt <= 0 caught", any("Nt_cm3" in n for n in notes), str(notes))
check("bad kind caught", any("kind" in n for n in notes), str(notes))
check("invalid defect tau = inf", bad_d.tau_n_s == float("inf"))

d_rt = DefectSpec.from_dict(mg.to_dict())
check("DefectSpec JSON round-trip", d_rt == mg)


# ---------------------------------------------------------------------------
# 12. LayerSpec + defects
# ---------------------------------------------------------------------------

section("12. LayerSpec defects")

layer_d = LayerSpec(
    name="PVK", thick_nm=500.0, Eg_eV=1.6, taun_s=1e-8, taup_s=1e-8,
    defects=(DefectSpec(Et_eV=0.8, Nt_cm3=1e15),),
)
# 1/tau = 1/1e-8 + 1/1e-7 = 1.1e8  → 9.0909e-9
expect = 1.0 / (1.0 / 1e-8 + 1.0 / 1e-7)
check("effective_tau_n combines explicit + defect",
      abs(layer_d.effective_tau_n_s - expect) <= 1e-9 * expect,
      f"got {layer_d.effective_tau_n_s}, expect {expect}")
check("effective_tau_p combines", abs(layer_d.effective_tau_p_s - expect) <= 1e-9 * expect)
check("layer validate clean", len(layer_d.validate(0)) == 0, str(layer_d.validate(0)))

layer_plain = LayerSpec(name="PVK", thick_nm=500.0, Eg_eV=1.6, taun_s=1e-8)
check("no defects → effective == explicit",
      layer_plain.effective_tau_n_s == 1e-8)

layer_bad_et = LayerSpec(
    name="PVK", thick_nm=500.0, Eg_eV=1.6,
    defects=(DefectSpec(Et_eV=2.0),),
)
check("layer validate propagates Et>Eg",
      any("out of" in n for n in layer_bad_et.validate(0)),
      str(layer_bad_et.validate(0)))

layer_d_rt = LayerSpec.from_dict(layer_d.to_dict())
check("layer defects survive round-trip",
      len(layer_d_rt.defects) == 1
      and abs(layer_d_rt.defects[0].Nt_cm3 - 1e15) < 1
      and layer_d_rt.defects[0].Et_eV == 0.8)

# defects 传 list 也归一化为 tuple
layer_list_in = LayerSpec(name="X", thick_nm=1, Eg_eV=1,
                          defects=[DefectSpec()])
check("defects list → tuple", isinstance(layer_list_in.defects, tuple))


# ---------------------------------------------------------------------------
# 13. SweepPlan 家族
# ---------------------------------------------------------------------------

section("13. SweepPlan")

base = default_problem_spec()


def _layer_of(spec_obj, name):
    return next(l for l in spec_obj.device.layers if l.name == name)


# 抽象基类不可直接实例化
try:
    SweepPlan()   # type: ignore[abstract]
    check("SweepPlan is abstract", False, "should have raised TypeError")
except TypeError:
    check("SweepPlan is abstract", True)

# BiasSweep
bs = BiasSweep(v_start=0.2, v_stop=1.2, n_points=11)
steps = bs.expand(base)
check("BiasSweep → 1 step", len(steps) == 1)
check("BiasSweep 应用 v_stop", steps[0].spec.sweep.v_stop == 1.2)
check("BiasSweep 应用 n_points", steps[0].spec.sweep.n_points == 11)
check("BiasSweep label 含 J-V", "J-V" in steps[0].label, steps[0].label)
check("BiasSweep 不改 base", base.sweep.v_stop == 1.8)
check("plan 可调用", len(bs(base)) == 1)

bs_default = BiasSweep()
steps = bs_default.expand(base)
check("BiasSweep(None) 沿用 base", steps[0].spec.sweep.v_stop == 1.8
      and steps[0].spec.sweep.n_points == 37)

try:
    BiasSweep(n_points=0).expand(base)
    check("BiasSweep n_points=0 raises", False)
except SpecError:
    check("BiasSweep n_points=0 raises", True)

# IntensitySweep
isw = IntensitySweep(sun_values=(1.0, 0.5))
steps = isw.expand(base)
check("IntensitySweep → 2 steps", len(steps) == 2)
check("1 sun = 100 mW/cm²",
      abs(steps[0].spec.optics.intensity_mW_cm2 - 100.0) < 1e-9)
check("0.5 sun = 50 mW/cm²",
      abs(steps[1].spec.optics.intensity_mW_cm2 - 50.0) < 1e-9)
check("IntensitySweep label 含 sun", all("sun" in s.label for s in steps))
check("IntensitySweep 不改 base",
      abs(base.optics.intensity_mW_cm2 - 100.0) < 1e-9)

two = IntensitySweep.two_stage(n_low=3, n_high=4)
check("two_stage → 7 steps", len(two.expand(base)) == 7)

try:
    IntensitySweep(sun_values=(1.0, -0.5)).expand(base)
    check("负 sun 值 raises", False)
except SpecError:
    check("负 sun 值 raises", True)

# ThicknessScan
ts = ThicknessScan(layer_name="MAPbI3", thick_nm_values=(300, 500, 700))
steps = ts.expand(base)
check("ThicknessScan → 3 steps", len(steps) == 3)
check("ThicknessScan 应用 700 nm",
      abs(_layer_of(steps[2].spec, "MAPbI3").thick_nm - 700.0) < 1e-9)
check("ThicknessScan label", steps[0].label == "MAPbI3=300 nm", steps[0].label)
check("ThicknessScan 不改 base",
      abs(_layer_of(base, "MAPbI3").thick_nm - 500.0) < 1e-9)
check("ThicknessScan 其它层不动",
      abs(_layer_of(steps[2].spec, "Spiro").thick_nm - 200.0) < 1e-9)

try:
    ThicknessScan(layer_name="NOPE", thick_nm_values=(1.0,)).expand(base)
    check("未知层 raises", False)
except SpecError:
    check("未知层 raises", True)

try:
    ThicknessScan(layer_name="MAPbI3", thick_nm_values=(0.0,)).expand(base)
    check("thick<=0 raises", False)
except SpecError:
    check("thick<=0 raises", True)

# DopingScan
ds = DopingScan(layer_name="MAPbI3", values_cm3=(1e15, 1e16), kind="Na")
steps = ds.expand(base)
check("DopingScan → 2 steps", len(steps) == 2)
check("DopingScan 应用 Na=1e16",
      abs(_layer_of(steps[1].spec, "MAPbI3").Na_cm3 - 1e16) < 1.0)
check("DopingScan 不动 Nd", _layer_of(steps[1].spec, "MAPbI3").Nd_cm3 == 0.0)
check("DopingScan label", "Na=1e+16" in steps[1].label, steps[1].label)

try:
    DopingScan(layer_name="MAPbI3", values_cm3=(1e15,), kind="Nx").expand(base)
    check("非法 kind raises", False)
except SpecError:
    check("非法 kind raises", True)

# DefectScan
try:
    DefectScan(layer_name="MAPbI3", Nt_values_cm3=(1e15,)).expand(base)
    check("无缺陷时 DefectScan raises", False)
except SpecError:
    check("无缺陷时 DefectScan raises", True)

base_defect = replace(base, device=replace(base.device, layers=tuple(
    replace(l, defects=(DefectSpec(name="mid", Et_eV=0.8, Nt_cm3=1e14),))
    if l.name == "MAPbI3" else l
    for l in base.device.layers
)))
dsc = DefectScan(layer_name="MAPbI3", Nt_values_cm3=(1e14, 1e15, 1e16))
steps = dsc.expand(base_defect)
check("DefectScan → 3 steps", len(steps) == 3)
check("DefectScan 应用 Nt=1e16",
      abs(_layer_of(steps[2].spec, "MAPbI3").defects[0].Nt_cm3 - 1e16) < 1.0)
check("DefectScan 改后 effective tau 变小",
      _layer_of(steps[2].spec, "MAPbI3").effective_tau_n_s
      < _layer_of(steps[0].spec, "MAPbI3").effective_tau_n_s)
check("DefectScan 不改 base_defect",
      abs(_layer_of(base_defect, "MAPbI3").defects[0].Nt_cm3 - 1e14) < 1.0)

try:
    DefectScan(layer_name="MAPbI3", Nt_values_cm3=(1e15,), defect_index=3).expand(base_defect)
    check("defect_index 越界 raises", False)
except SpecError:
    check("defect_index 越界 raises", True)

check("PlanStep 有 label 和 spec",
      isinstance(steps[0], PlanStep) and isinstance(steps[0].spec, ProblemSpec))


# ---------------------------------------------------------------------------
# 14. ProblemSpec 含缺陷时的 JSON round-trip
# ---------------------------------------------------------------------------

section("14. ProblemSpec + defects JSON")

spec_defect = base_defect
j2 = spec_defect.to_json()
spec_defect2 = ProblemSpec.from_json(j2)

check("JSON 保留 defects",
      any(len(l.defects) > 0 for l in spec_defect2.device.layers))
pvk2 = _layer_of(spec_defect2, "MAPbI3")
pvk1 = _layer_of(spec_defect, "MAPbI3")
check("JSON 保留 Nt", abs(pvk2.defects[0].Nt_cm3 - pvk1.defects[0].Nt_cm3) < 1.0)
check("JSON 后 effective tau 一致",
      abs(pvk2.effective_tau_n_s - pvk1.effective_tau_n_s) <= 1e-9 * pvk1.effective_tau_n_s)

try:
    spec_defect2.validate()
    check("带缺陷的 spec validate → OK", True)
except SpecError as e:
    fatal = [d for d in e.details
             if any(k in d for k in ("must be", "out of", "empty", "unknown"))]
    check("带缺陷的 spec validate → OK (only warnings)",
          len(fatal) == 0, f"fatal={fatal}, all={e.details}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
