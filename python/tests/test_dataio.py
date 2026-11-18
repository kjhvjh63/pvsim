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


def _try(fn, exc):
    try:
        fn()
    except exc as e:
        return e
    return None


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
# 9. CSV 导出（GUI「导出结果」走的第二出口）
# ---------------------------------------------------------------------------

section("9. jv csv export")

from pvsim.dataio import CSV_HEADER, jv_to_csv, write_jv_csv

csv_jv = JVData.from_points([0.0, 0.5, 1.0], [-20.0, -10.0, 0.0], 100.0)
csv_text = jv_to_csv(csv_jv)
csv_lines = csv_text.strip().splitlines()
check("CSV 表头", csv_lines[0] == ",".join(CSV_HEADER), csv_lines[0])
check("CSV 行数 = 点数 + 1", len(csv_lines) == 4, str(len(csv_lines)))
check("CSV 功率列 = −J·V", csv_lines[2].split(",")[2] == "5",
      csv_lines[2])

with tempfile.TemporaryDirectory() as tmp:
    csv_path = write_jv_csv(csv_jv, Path(tmp) / "jv.csv")
    check("write_jv_csv 落盘", csv_path.exists())
    check("CSV 往返一致", csv_path.read_text(encoding="utf-8") == csv_text)


# ---------------------------------------------------------------------------
# 10. 外部对照（SCAPS / Silvaco 参考曲线）
# ---------------------------------------------------------------------------

section("10. external reference compare")

from pvsim.dataio import compare, load_reference
from pvsim.dataio.reference import read_text_curve
from pvsim.dataio import SilvacoLogWriter

REF_V = np.linspace(0.0, 1.0, 11)
REF_J = -22.0 + 22.0 * (np.exp(REF_V / 0.05) - 1.0) / (np.exp(20.0) - 1.0)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # 有表头的 CSV（SCAPS 常见导出）
    csv_ref = tmp / "scaps.iv"
    rows = ["# SCAPS-1D J-V export", "V (V)\tJ (mA/cm2)\tP"]
    for v, j in zip(REF_V, -REF_J):          # 参考惯例：短路电流为正
        rows.append(f"{v:.6f}\t{j:.6f}\t{-j * v:.6f}")
    csv_ref.write_text("\n".join(rows) + "\n", encoding="utf-8")

    V_csv, J_csv, header = read_text_curve(csv_ref)
    check("文本表头识别", header[0].startswith("V") and len(header) == 3,
          str(header))
    check("文本读数点数", V_csv.size == REF_V.size, str(V_csv.size))

    jv_ref = load_reference(csv_ref)
    check("参考曲线已归一为内核惯例（指标为正）",
          jv_ref.Jsc > 0 and abs(jv_ref.Jsc - 22.0) < 0.2, f"{jv_ref.Jsc:.4f}")

    # 无表头、短路为负（已是内核惯例）→ 保持不变
    plain = tmp / "atlas.dat"
    plain.write_text("\n".join(f"{v:.6f} {j:.6f}"
                               for v, j in zip(REF_V, REF_J)) + "\n",
                     encoding="utf-8")
    jv_plain = load_reference(plain)
    check("无表头文件自动识别惯例",
          abs(jv_plain.Jsc - 22.0) < 0.2, f"{jv_plain.Jsc:.4f}")

    # SSF 往返：写出去的 .log 能被 load_reference 读回同一曲线
    sim_like = JVData.from_points(REF_V, REF_J, 100.0)
    log_path = SilvacoLogWriter().write(sim_like, tmp / "sim.log")
    jv_log = load_reference(log_path)
    check("SSF 参考读回 Jsc", abs(jv_log.Jsc - 22.0) < 0.2,
          f"{jv_log.Jsc:.4f}")

    # 完全一致 → 全部通过
    rep = compare(sim_like, load_reference(csv_ref))
    check("完全一致时通过", rep.passed, "\n".join(rep.lines()))
    check("完全一致时 RMSE≈0", rep.rmse_mA_cm2 < 1e-6,
          f"{rep.rmse_mA_cm2:g}")
    check("公共区间覆盖 0→1 V",
          abs(rep.v_window[0]) < 1e-12 and abs(rep.v_window[1] - 1.0) < 1e-12,
          str(rep.v_window))

    # Jsc 偏 10% → Jsc 项应判超差
    worse = JVData.from_points(REF_V, REF_J * 1.10, 100.0)
    rep_bad = compare(worse, load_reference(csv_ref))
    jsc_metric = next(m for m in rep_bad.metrics if m.name == "Jsc")
    check("Jsc 偏 10% 判超差", not jsc_metric.ok and not rep_bad.passed,
          jsc_metric.line())

    # 区间不重叠 → 明确报错
    far = JVData.from_points(REF_V + 5.0, REF_J, 100.0)
    check("无重叠区间抛 ValueError",
          isinstance(_try(lambda: compare(far, load_reference(csv_ref)),
                          ValueError), ValueError))


# ---------------------------------------------------------------------------
# 11. SCAPS 对照材料导出
# ---------------------------------------------------------------------------

section("11. scaps export")

from pvsim.dataio import export_scaps, layers_of, parameter_table, scaps_script
from pvsim.script import DeckCompiler

SCAPS_DECK = (
    "region material=FTO thickness=200\n"
    "region material=MAPbI3 thickness=500\n"
    "region material=Au thickness=80\n"
    "doping region=1 n.type conc=1e20\n"
    "doping region=3 p.type conc=1e20\n"
    "solve vstart=0 vfinal=1.5 vstep=0.05\n"
)
scaps_spec = DeckCompiler().compile_text(SCAPS_DECK)
scaps_layers = layers_of(scaps_spec)

check("SCAPS 层数 = deck 层数", len(scaps_layers) == 3, str(len(scaps_layers)))
check("厚度换算成 µm", scaps_layers[1].thickness_um == 0.5,
      str(scaps_layers[1].thickness_um))
check("首末层标记为接触层",
      scaps_layers[0].is_contact and scaps_layers[-1].is_contact
      and not scaps_layers[1].is_contact)
check("吸收层参数带过来",
      scaps_layers[1].Eg_eV == 1.6 and scaps_layers[1].taun_s == 8e-9,
      f"{scaps_layers[1].Eg_eV} / {scaps_layers[1].taun_s}")

script_text = scaps_script(scaps_spec, iv_name="demo")
check("脚本含温度", "action workingpoint.temperature 300" in script_text)
check("脚本含光照", "action light" in script_text)
check("脚本 IV 区间来自 deck",
      "action iv.startV 0" in script_text and "action iv.stopV 1.5" in script_text,
      script_text)
check("脚本点数来自 deck", "action iv.points 31" in script_text)
check("脚本保存 .iv", "save results.iv demo" in script_text)
check("脚本只用已验证命令",
      all(line.startswith(("//", "action ", "save ", ""))
          for line in script_text.splitlines()),
      script_text)

table_text = parameter_table(scaps_spec, iv_name="demo")
check("清单含每层材料",
      all(l.material in table_text for l in scaps_layers))
check("清单含已验证的层命令",
      "set layer2.thickness 0.5" in table_text
      and "set layer2.Eg" in table_text and "set layer2.epsilon" in table_text,
      table_text[:200])
check("清单说明总层数与半导体层数", "层数：3" in table_text)

with tempfile.TemporaryDirectory() as tmp:
    sp, tp = export_scaps(scaps_spec, tmp, iv_name="demo")
    check("导出脚本文件", Path(sp).exists() and Path(sp).name == "demo.script")
    check("导出参数清单", Path(tp).exists() and Path(tp).name == "demo_scaps_params.md")
    check("导出内容与函数一致",
          Path(sp).read_text(encoding="utf-8") == script_text)


# ---------------------------------------------------------------------------
# 12. Silvaco 真实导出格式（EXPORTED XY-GRAPH / xlsx）
# ---------------------------------------------------------------------------

section("12. silvaco raw exports")

from pvsim.dataio.reference import read_silvaco_xy, read_xlsx_curve

xy_text = (
    "EXPORTED\tXY-GRAPH\tDATA\t\t10000000000\n"
    "4\t2\t2\t\t\n"
    "Anode\tCurrent\t\t\t\n"
    "Cathode\tVoltage\t\t\t\n"
    "15.04\t0.00E+00\t0\t\n"
    "12.00\t0.50E+00\t1\t\n"
    "6.00\t1.00E+00\t2\t\n"
    "0.00\t1.40E+00\t3\t\n"
)
with tempfile.TemporaryDirectory() as tmp:
    xy_path = Path(tmp) / "ref.dat"
    xy_path.write_text(xy_text, encoding="utf-8")
    V_xy, J_xy, head_xy = read_silvaco_xy(xy_path)
    check("XY-GRAPH 列名识别", head_xy == ("Anode Current", "Cathode Voltage"),
          str(head_xy))
    check("XY-GRAPH 自动纠正列序（电流在前）",
          np.allclose(V_xy, [0.0, 0.5, 1.0, 1.4]) and np.allclose(J_xy, [15.04, 12, 6, 0]),
          f"V={V_xy}, J={J_xy}")

    jv_xy = load_reference(xy_path)
    check("XY-GRAPH 读入后短路为负（内核惯例）",
          jv_xy.Jsc > 0 and abs(jv_xy.Jsc - 15.04) < 1e-6, f"{jv_xy.Jsc}")

    # xlsx：表头挑 Current Density 与 Voltage（openpyxl 缺失时跳过）
    try:
        import openpyxl
    except ImportError:                          # pragma: no cover
        print("  ⏭ skip xlsx：未安装 openpyxl")
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Anode Current (A)", "Cathode Voltage (V)",
                   "Current Density (mA/cm^2)", "Power (W)"])
        for v, j in ((0.0, 15.04), (0.5, 12.0), (1.0, 6.0), (1.4, 0.0)):
            ws.append([j * 1e-4, v, j, 0.0])
        xlsx_path = Path(tmp) / "ref.xlsx"
        wb.save(xlsx_path)
        wb.close()

        V_x, J_x, head_x = read_xlsx_curve(xlsx_path)
        check("xlsx 按表头挑列",
              np.allclose(V_x, [0.0, 0.5, 1.0, 1.4]) and np.allclose(J_x, [15.04, 12, 6, 0]),
              f"V={V_x}, J={J_x}")
        jv_x = load_reference(xlsx_path)
        check("xlsx 读入后短路为负（内核惯例）", jv_x.Jsc > 0 and abs(jv_x.Jsc - 15.04) < 1e-6)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
