"""ResultStore 测试：原子提交 / round-trip / NaN / 真实内核集成。

Run: python python/tests/test_dataio.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pvsim.dataio import LAYOUT_VERSION, ResultStore
from pvsim.param import default_problem_spec
from pvsim.solver import FieldTable, JVData, SimResult, Simulator, SolvePoint

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


def make_result(run_id="runA", n_pts=5, n_node=10):
    """构造一个不依赖内核的 SimResult。"""
    spec = default_problem_spec()
    V = np.linspace(0.0, 1.8, n_pts)
    J = np.linspace(-20.0, 5.0, n_pts)      # 负→正，存在真实 Voc
    pts = tuple(
        SolvePoint(index=i, V=float(V[i]), J_mA_cm2=float(J[i]),
                   converged=True, residual=1e-12, iterations=3)
        for i in range(n_pts)
    )
    x_nm = np.linspace(0.0, 1020.0, n_node)
    data = {
        name: np.arange(n_pts * n_node, dtype=float).reshape(n_pts, n_node) + k
        for k, name in enumerate(spec.output.fields)
    }
    jv = JVData.from_points(V, J, spec.optics.intensity_mW_cm2)
    return SimResult(run_id=run_id, spec=spec, points=pts,
                     fields=FieldTable(x_nm=x_nm, data=data), jv=jv)


tmp_root = Path(tempfile.mkdtemp(prefix="pvk_store_"))
store = ResultStore(tmp_root)
check("results/ 自动创建", store.results_dir.is_dir())
check("初始 list_runs 为空", store.list_runs() == [])


# ---------------------------------------------------------------------------
# 1. save → 目录结构
# ---------------------------------------------------------------------------

section("1. save layout")

res = make_result("runA")
path = store.save(res)

check("返回正式目录", path == store.results_dir / "runA")
check("spec.json 存在", (path / "spec.json").is_file())
check("meta.json 存在", (path / "meta.json").is_file())
check("points.json 存在", (path / "points.json").is_file())
check("jv.json 存在", (path / "jv.json").is_file())
check("fields/ 存在", (path / "fields").is_dir())
check("x_nm.npy 存在", (path / "fields" / "x_nm.npy").is_file())
check("每个场量一个 .npy",
      all((path / "fields" / f"{n}.npy").is_file() for n in res.fields.names),
      str(res.fields.names))
check("无残留临时目录",
      not any(p.name.startswith(".tmp_") for p in store.results_dir.iterdir()))
check("list_runs 含 runA", store.list_runs() == ["runA"])
check("exists(runA)", store.exists("runA"))
check("exists(nope) 为 False", not store.exists("nope"))

meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
check("meta.layout_version 正确", meta["layout_version"] == LAYOUT_VERSION)
check("meta 记录点数", meta["n_points"] == 5)
check("meta 记录节点数", meta["n_node"] == 10)
check("meta 记录收敛", meta["converged"] is True)
check("meta 记录场量清单", meta["fields"] == list(res.fields.names))


# ---------------------------------------------------------------------------
# 2. load round-trip
# ---------------------------------------------------------------------------

section("2. load round-trip")

loaded = store.load("runA")

check("run_id 一致", loaded.run_id == "runA")
check("spec sweep 点数一致", loaded.spec.sweep.n_points == res.spec.sweep.n_points)
check("spec 层数一致", len(loaded.spec.device.layers) == len(res.spec.device.layers))
check("点数值一致",
      [(p.V, p.J_mA_cm2, p.iterations) for p in loaded.points]
      == [(p.V, p.J_mA_cm2, p.iterations) for p in res.points])
check("x_nm 一致", np.allclose(loaded.fields.x_nm, res.fields.x_nm))
check("场量名一致", loaded.fields.names == res.fields.names)
check("每张场表数值一致",
      all(np.allclose(loaded.fields[n], res.fields[n]) for n in res.fields.names))
check("场表形状一致",
      all(loaded.fields[n].shape == res.fields[n].shape for n in res.fields.names))
check("Jsc 重算一致", abs(loaded.jv.Jsc - res.jv.Jsc) < 1e-12)
check("Voc 重算一致", abs(loaded.jv.Voc - res.jv.Voc) < 1e-12)
check("FF 重算一致", abs(loaded.jv.FF - res.jv.FF) < 1e-12)
check("PCE 重算一致", abs(loaded.jv.PCE - res.jv.PCE) < 1e-12)
check("converged 保持", loaded.converged is True)


# ---------------------------------------------------------------------------
# 3. begin → commit 的可见性
# ---------------------------------------------------------------------------

section("3. atomic visibility")

handle = store.begin(res.spec, run_id="manual")
check("begin 返回 run_id", handle.run_id == "manual")
check("begin 后对读者不可见", not store.exists("manual"))
check("begin 后 list_runs 不含", "manual" not in store.list_runs())
check("临时目录已建且是隐藏名",
      handle.tmp_dir.is_dir() and handle.tmp_dir.name.startswith(".tmp_"))
try:
    store.load("manual")
    check("commit 前 load 抛 FileNotFoundError", False, "should have raised")
except FileNotFoundError:
    check("commit 前 load 抛 FileNotFoundError", True)

store.commit(handle, make_result("manual"))
check("commit 后可见", store.exists("manual"))
check("commit 后临时目录消失", not handle.tmp_dir.exists())
check("commit 后 list_runs 含两条",
      store.list_runs() == ["manual", "runA"], str(store.list_runs()))


# ---------------------------------------------------------------------------
# 4. 防覆盖 + run_id 自动避让
# ---------------------------------------------------------------------------

section("4. no overwrite")

before = (store.results_dir / "manual" / "meta.json").read_text(encoding="utf-8")
try:
    store.commit(handle, make_result("manual"))
    check("重复 commit 抛 FileExistsError", False, "should have raised")
except FileExistsError:
    check("重复 commit 抛 FileExistsError", True)
after = (store.results_dir / "manual" / "meta.json").read_text(encoding="utf-8")
check("重复 commit 不改动原结果", before == after)

h2 = store.begin(res.spec, run_id="manual")
check("同名 begin 自动避让为 manual_2", h2.run_id == "manual_2", h2.run_id)
store.commit(h2, make_result(h2.run_id))
check("manual_2 已提交", store.exists("manual_2"))

h3 = store.begin(res.spec, run_id="manual")
check("再次避让为 manual_3", h3.run_id == "manual_3", h3.run_id)
store.discard(h3)
check("discard 后临时目录消失", not h3.tmp_dir.exists())
check("discard 不影响已提交结果", store.exists("manual") and store.exists("manual_2"))


# ---------------------------------------------------------------------------
# 5. 非法 run_id
# ---------------------------------------------------------------------------

section("5. invalid run_id")

for bad_id in ("", "  ", "..", "../evil", "a/b", "a\\b"):
    try:
        store.begin(res.spec, run_id=bad_id)
        check(f"拒绝非法 run_id {bad_id!r}", False, "should have raised")
    except ValueError:
        check(f"拒绝非法 run_id {bad_id!r}", True)

try:
    store.load("../evil")
    check("load 拒绝路径穿越", False, "should have raised")
except ValueError:
    check("load 拒绝路径穿越", True)


# ---------------------------------------------------------------------------
# 6. NaN 序列化
# ---------------------------------------------------------------------------

section("6. NaN handling")

res_nan = make_result("nan_case")
check("构造的 jv.Voc 有限", math.isfinite(res_nan.jv.Voc))
store.save(res_nan)

# 全零 J → 所有指标 NaN
spec = default_problem_spec()
n_pts, n_node = 3, 6
pts = tuple(
    SolvePoint(index=i, V=float(v), J_mA_cm2=0.0, converged=True,
               residual=0.0, iterations=1)
    for i, v in enumerate(np.linspace(0.0, 1.8, n_pts))
)
nan_res = SimResult(
    run_id="all_nan",
    spec=spec,
    points=pts,
    fields=FieldTable(x_nm=np.linspace(0, 1020, n_node),
                      data={n: np.zeros((n_pts, n_node)) for n in spec.output.fields}),
    jv=JVData.from_points([p.V for p in pts], [0.0] * n_pts,
                          spec.optics.intensity_mW_cm2),
)
nan_path = store.save(nan_res)
nan_jv_json = json.loads((nan_path / "jv.json").read_text(encoding="utf-8"))
check("jv.json 中 Voc 写成 null", nan_jv_json["Voc"] is None)
check("jv.json 中 FF 写成 null", nan_jv_json["FF"] is None)
check("jv.json 中 PCE 写成 null", nan_jv_json["PCE"] is None)

nan_loaded = store.load("all_nan")
check("加载回来 Voc 仍是 NaN", math.isnan(nan_loaded.jv.Voc))
check("加载回来 Jsc = 0", nan_loaded.jv.Jsc == 0.0)


# ---------------------------------------------------------------------------
# 7. 布局版本校验
# ---------------------------------------------------------------------------

section("7. layout version")

bad_dir = store.results_dir / "bad_layout"
bad_dir.mkdir()
(bad_dir / "meta.json").write_text(json.dumps({"layout_version": 999}),
                                   encoding="utf-8")
try:
    store.load("bad_layout")
    check("不支持的布局版本抛 ValueError", False, "should have raised")
except ValueError:
    check("不支持的布局版本抛 ValueError", True)


# ---------------------------------------------------------------------------
# 8. 真实内核集成（内核不可用时跳过）
# ---------------------------------------------------------------------------

section("8. kernel integration")

try:
    kernel_res = Simulator().run(default_problem_spec(), run_id="kernel_case")
except Exception as exc:            # DLL 未构建 / 运行时依赖缺失
    print(f"  ⏭ skip：内核不可用（{type(exc).__name__}: {exc}）")
else:
    kp = store.save(kernel_res)
    check("内核结果已提交", store.exists("kernel_case"))
    k_loaded = store.load("kernel_case")
    check("内核结果点数一致", len(k_loaded.points) == len(kernel_res.points))
    check("内核场表形状一致",
          all(k_loaded.fields[n].shape == kernel_res.fields[n].shape
              for n in kernel_res.fields.names))
    check("内核场表数值一致",
          all(np.allclose(k_loaded.fields[n], kernel_res.fields[n])
              for n in kernel_res.fields.names))
    check("内核 spec 可回溯",
          k_loaded.spec.device.total_thick_nm
          == kernel_res.spec.device.total_thick_nm)
    check("meta 记录 n_node", json.loads(
        (kp / "meta.json").read_text(encoding="utf-8"))["n_node"] == k_loaded.fields.n_node)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
