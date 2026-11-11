"""viz 测试：注册表 / 出图 / 数据校验 / 数值正确性 / 落盘。

Run: python python/tests/test_viz.py

注意：必须先把 matplotlib 切到无界面后端再 import pvsim.viz。
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pvsim.param import default_problem_spec
from pvsim.param.schema import DopingScan, ThicknessScan
from pvsim.solver import FieldTable, JVData, PlanRun, SimResult, SolvePoint
from pvsim.viz import REGISTRY, MissingData, PlotConfig, PlotRegistry, ScanSet
from pvsim.viz import (LiteratureEntry, ThesisSet, axes_style, fallback,
                       load_curve, load_dat, load_xlsx, scan_summary_ids)

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
    """执行 fn，返回捕获到的异常实例；没抛就返回 None。"""
    try:
        fn()
    except exc as e:
        return e
    return None


def make_result(n_pts=5, n_node=12, fields=None, run_id="viz", V=None, J=None,
                spec=None):
    """造一个不依赖内核的 SimResult，场量数值可控便于断言。"""
    spec = default_problem_spec() if spec is None else spec
    V = np.linspace(0.0, 1.8, n_pts) if V is None else np.asarray(V, dtype=float)
    J = np.linspace(-22.0, 6.0, n_pts) if J is None else np.asarray(J, dtype=float)
    n_pts = V.size
    pts = tuple(
        SolvePoint(index=i, V=float(V[i]), J_mA_cm2=float(J[i]),
                   converged=True, residual=1e-12, iterations=3)
        for i in range(n_pts)
    )
    x_nm = np.linspace(0.0, spec.device.total_thick_nm, n_node)
    names = fields if fields is not None else (
        "X", "PSI", "N", "P", "EFIELD", "JTOTAL", "G_OPT", "R_NET", "CB", "VB",
    )
    data = {}
    for name in names:
        if name == "N":
            row = np.logspace(10, 18, n_node)
        elif name == "P":
            row = np.logspace(17, 9, n_node)
        elif name == "EFIELD":
            row = -np.linspace(1e3, 5e3, n_node)
        elif name == "R_NET":
            row = np.linspace(-1e18, 3e19, n_node)
        elif name == "G_OPT":
            row = np.linspace(1e21, 1e17, n_node)
        elif name == "CB":
            row = np.linspace(-3.9, -3.5, n_node)
        elif name == "VB":
            row = np.linspace(-5.5, -5.1, n_node)
        else:
            row = x_nm / max(float(x_nm[-1]), 1.0)
        data[name] = np.tile(row, (n_pts, 1))

    jv = JVData.from_points(V, J, spec.optics.intensity_mW_cm2)
    return SimResult(run_id=run_id, spec=spec, points=pts,
                     fields=FieldTable(x_nm=x_nm, data=data), jv=jv)


# ---------------------------------------------------------------------------
# 1. 注册表
# ---------------------------------------------------------------------------
section("注册表")

EXPECTED = {
    "cell_jv_curve", "cell_pv_curve", "cell_band_diagram",
    "cell_efield_profile", "cell_carrier_profile", "cell_recomb_profile",
    "cell_gen_profile",
    "scan_jsc_vs_x", "scan_voc_vs_x", "scan_ff_vs_x", "scan_pce_vs_x",
    "scan_jv_family",
    "structure_stack", "structure_band_align",
    "compare_jv", "compare_pce_bar",
}
check("已注册 16 张图", len(REGISTRY) == len(EXPECTED), f"实际 {len(REGISTRY)}")
check("id 集合一致", set(REGISTRY.ids()) == EXPECTED, str(REGISTRY.ids()))
check("分组顺序：电学 → 光学 → 扫描 → 结构 → 论文对照",
      REGISTRY.groups() == ("电学", "光学", "扫描", "结构", "论文对照"),
      str(REGISTRY.groups()))
check("电学组 6 张", len(REGISTRY.by_group()["电学"]) == 6)
check("光学组 1 张", len(REGISTRY.by_group()["光学"]) == 1)
check("扫描组 5 张", len(REGISTRY.by_group()["扫描"]) == 5)
check("结构组 2 张", len(REGISTRY.by_group()["结构"]) == 2)
check("论文对照组 2 张", len(REGISTRY.by_group()["论文对照"]) == 2)
check("J-V 只依赖 jv", REGISTRY.get("cell_jv_curve").needs == ("jv",))
check("产生率依赖 optics", REGISTRY.get("cell_gen_profile").needs == ("optics",))
check("剖面图依赖 fields",
      REGISTRY.get("cell_efield_profile").needs == ("fields",))
check("扫描图依赖 scan", REGISTRY.get("scan_jsc_vs_x").needs == ("scan",))
check("group 支持 in 查询", "cell_jv_curve" in REGISTRY)

try:
    REGISTRY.register(id="cell_jv_curve", group="重复", needs=("jv",))(lambda r, c, a: None)
    check("重复 id 抛 ValueError", False)
except ValueError:
    check("重复 id 抛 ValueError", True)

try:
    REGISTRY.get("no_such_plot")
    check("未知 id 抛 KeyError", False)
except KeyError as e:
    check("未知 id 抛 KeyError", "no_such_plot" in str(e))

try:
    PlotRegistry().register(id="x", group="g", needs=("magic",))(lambda r, c, a: None)
    check("非法 needs 抛 ValueError", False)
except ValueError:
    check("非法 needs 抛 ValueError", True)

# ---------------------------------------------------------------------------
# 2. 出图与外观
# ---------------------------------------------------------------------------
section("出图与外观")

r = make_result()
ax = REGISTRY.render("cell_jv_curve", r)
check("render 返回 Axes", hasattr(ax, "plot"))
check("默认标题来自注册项", ax.get_title() == "J-V 曲线", repr(ax.get_title()))
check("x 轴标签含 V", "V" in ax.get_xlabel(), ax.get_xlabel())
check("y 轴标签含 mA/cm²", "mA/cm²" in ax.get_ylabel(), ax.get_ylabel())
check("网格默认开启", ax.get_xgridlines()[0].get_visible())
check("指标框含 Jsc",
      any("Jsc" in t.get_text() for t in ax.texts))

ax2 = REGISTRY.render("cell_jv_curve", r, cfg=PlotConfig(title="自定标题",
                                                        xlim=(-0.2, 2.0),
                                                        grid=False))
check("cfg 覆盖标题", ax2.get_title() == "自定标题")
check("cfg 覆盖 xlim", tuple(ax2.get_xlim()) == (-0.2, 2.0), str(ax2.get_xlim()))
check("cfg 可关网格", not ax2.get_xgridlines()[0].get_visible())

ax3 = REGISTRY.render("cell_carrier_profile", r)
check("载流子图用对数纵轴", ax3.get_yscale() == "log")
check("载流子图有图例", ax3.get_legend() is not None)

ax4 = REGISTRY.render("cell_band_diagram", r)
check("能带图有图例", ax4.get_legend() is not None)
check("能带图 x 范围 = 器件厚度",
      tuple(ax4.get_xlim()) == (0.0, r.spec.device.total_thick_nm),
      str(ax4.get_xlim()))
check("层名标注 5 个",
      sum(1 for t in ax4.texts if t.get_text() in
          {l.name for l in r.spec.device.layers}) == 5)

# ---------------------------------------------------------------------------
# 3. 数值正确性
# ---------------------------------------------------------------------------
section("数值正确性")

jv_ax = REGISTRY.render("cell_jv_curve", r)
check("J-V 横轴 = 偏压", np.allclose(jv_ax.lines[0].get_xdata(), r.jv.V))
check("J-V 纵轴 = 交付电流 −J",
      np.allclose(jv_ax.lines[0].get_ydata(), -r.jv.J))

pv_ax = REGISTRY.render("cell_pv_curve", r)
check("P-V 纵轴 = −J·V",
      np.allclose(pv_ax.lines[0].get_ydata(), -r.jv.J * r.jv.V))
check("P-V 标出 Pmax", any("Pmax" in t.get_text() for t in pv_ax.texts))

e_ax = REGISTRY.render("cell_efield_profile", r)
check("电场剖面取最后一个偏压点",
      np.allclose(e_ax.lines[0].get_ydata(), r.fields["EFIELD"][-1, :]))
check("电场剖面标题带偏压", "V = " in e_ax.get_title(), e_ax.get_title())

e0_ax = REGISTRY.render("cell_efield_profile", r, cfg=PlotConfig(point=0))
check("cfg.point 可指定偏压点",
      np.allclose(e0_ax.lines[0].get_ydata(), r.fields["EFIELD"][0, :]))

try:
    REGISTRY.render("cell_efield_profile", r, cfg=PlotConfig(point=99))
    check("point 越界抛 IndexError", False)
except IndexError:
    check("point 越界抛 IndexError", True)

band_ax = REGISTRY.render("cell_band_diagram", r)
check("能带 Ec 曲线数值正确",
      np.allclose(band_ax.lines[0].get_ydata(), r.fields["CB"][-1, :]))
check("能带 Ev 曲线数值正确",
      np.allclose(band_ax.lines[1].get_ydata(), r.fields["VB"][-1, :]))

# ---------------------------------------------------------------------------
# 4. 数据缺失
# ---------------------------------------------------------------------------
section("数据缺失")

no_band = make_result(fields=("X", "PSI", "N", "P", "EFIELD", "JTOTAL", "G_OPT"))
try:
    REGISTRY.render("cell_band_diagram", no_band)
    check("缺 CB/VB 抛 MissingData", False)
except MissingData as e:
    check("缺 CB/VB 抛 MissingData", e.need.startswith("fields:"), e.need)
    check("MissingData 带 plot_id", e.plot_id == "cell_band_diagram", e.plot_id)

empty = make_result(fields=())
try:
    REGISTRY.render("cell_efield_profile", empty)
    check("空场表抛 MissingData", False)
except MissingData as e:
    check("空场表抛 MissingData", e.need == "fields", e.need)

no_optics = make_result(fields=("X", "PSI", "N", "P", "EFIELD", "JTOTAL"))
try:
    REGISTRY.render("cell_gen_profile", no_optics)
    check("缺 G_OPT 抛 MissingData", False)
except MissingData as e:
    check("缺 G_OPT 抛 MissingData", e.need == "optics", e.need)

# ---------------------------------------------------------------------------
# 5. 多子图与落盘
# ---------------------------------------------------------------------------
section("多子图与落盘")

fig = REGISTRY.figure(r, ["cell_jv_curve", "cell_pv_curve"])
check("figure 返回 Figure", hasattr(fig, "savefig"))
check("双图 = 2 个 Axes", len(fig.axes) == 2, str(len(fig.axes)))
plt.close(fig)

fig3 = REGISTRY.figure(r, ["cell_jv_curve", "cell_pv_curve",
                           "cell_efield_profile"], cols=2)
check("三图 cols=2 → 4 个 Axes 槽位", len(fig3.axes) == 4, str(len(fig3.axes)))
check("多余槽位隐藏", not fig3.axes[3].get_visible())
plt.close(fig3)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "nested" / "jv.png"
    path = REGISTRY.savefig(r, ["cell_jv_curve", "cell_pv_curve"], out)
    check("savefig 自动建目录", path.exists())
    check("PNG 文件非空", path.stat().st_size > 1000,
          f"{path.stat().st_size} B")
    check("PNG magic 正确", path.read_bytes()[:4] == b"\x89PNG")

    fig = REGISTRY.figure(r, ["cell_gen_profile"])
    saved = axes_style.save(fig, Path(tmp) / "gen.png")
    check("axes_style.save 可用", saved.exists() and saved.stat().st_size > 0)

# ---------------------------------------------------------------------------
# 5b. 扫描汇总
# ---------------------------------------------------------------------------
section("扫描汇总")

V7 = np.linspace(0.0, 1.8, 7)


def make_scan(n=3):
    """n 条 J-V：Jsc 随序号递增，形成可断言的趋势。"""
    return [
        PlanRun(label=f"MAPbI3={400 + 100 * i} nm",
                result=make_result(V=V7, J=np.linspace(-18.0 - 2.0 * i, 4.0 + i, 7),
                                   run_id=f"scan{i}"))
        for i in range(n)
    ]


runs = make_scan()
scan = ScanSet.from_runs(runs, x=(400, 500, 600), xlabel="厚度 (nm)")
check("ScanSet 长度 = run 数", len(scan) == 3)
check("ScanSet x 转成 float 数组",
      scan.x.dtype == float and np.allclose(scan.x, (400, 500, 600)))
check("ScanSet labels 取自 PlanRun", scan.labels[2] == "MAPbI3=600 nm")
check("Jsc 指标随扫描单调上升",
      np.all(np.diff(scan.metric("Jsc")) > 0), str(scan.metric("Jsc")))
check("未知指标抛 KeyError",
      isinstance(_try(lambda: scan.metric("nope"), KeyError), KeyError))

thick = ThicknessScan(layer_name="MAPbI3", thick_nm_values=(400.0, 500.0, 600.0))
s_thick = ScanSet.from_plan(thick, runs)
check("from_plan 取厚度轴", np.allclose(s_thick.x, (400, 500, 600)))
check("厚度轴标签含层名与单位",
      "MAPbI3" in s_thick.xlabel and "nm" in s_thick.xlabel, s_thick.xlabel)
check("厚度扫描非对数轴", s_thick.logx is False)

dop = DopingScan(layer_name="MAPbI3", values_cm3=(1e15, 1e16, 1e17), kind="Na")
s_dop = ScanSet.from_plan(dop, make_scan())
check("掺杂扫描自动对数轴", s_dop.logx is True)
check("掺杂轴标签含 kind", "Na" in s_dop.xlabel, s_dop.xlabel)

check("x 与 runs 数量不符抛 ValueError",
      isinstance(_try(lambda: ScanSet.from_runs(runs, x=(1, 2), xlabel="x"),
                      ValueError), ValueError))
check("空 ScanSet 抛 ValueError",
      isinstance(_try(lambda: ScanSet.from_runs((), x=(), xlabel="x"),
                      ValueError), ValueError))

ax_j = REGISTRY.render("scan_jsc_vs_x", scan)
check("Jsc 趋势数值正确", np.allclose(ax_j.lines[0].get_ydata(), scan.metric("Jsc")))
check("趋势图横轴 = 扫描参数", np.allclose(ax_j.lines[0].get_xdata(), scan.x))
check("趋势图 x 标签来自 ScanSet", ax_j.get_xlabel() == "厚度 (nm)")

ax_log = REGISTRY.render("scan_pce_vs_x", s_dop)
check("掺杂扫描出图用对数横轴", ax_log.get_xscale() == "log")

ax_fam = REGISTRY.render("scan_jv_family", scan)
check("家族曲线 = n 条 J-V + 零电流线", len(ax_fam.lines) == 4, str(len(ax_fam.lines)))
check("首条曲线 = −J", np.allclose(ax_fam.lines[0].get_ydata(), -runs[0].result.jv.J))
check("家族图有图例", ax_fam.get_legend() is not None)

# 全零 J（stub 内核的典型输出）→ Voc/FF/PCE 全 NaN，应画注记而不是空白
zero_runs = [PlanRun(label=f"z{i}",
                     result=make_result(V=V7, J=np.zeros(7), run_id=f"z{i}"))
             for i in range(2)]
zscan = ScanSet.from_runs(zero_runs, x=(1, 2), xlabel="扫描序号")
zax = REGISTRY.render("scan_voc_vs_x", zscan)
check("全 NaN 时画注记", any("NaN" in t.get_text() for t in zax.texts))
check("全 NaN 时指标数组仍保留 NaN",
      np.all(np.isnan(zscan.metric("Voc"))))

check("单结果不能画扫描图",
      isinstance(_try(lambda: REGISTRY.render("scan_jsc_vs_x", r), MissingData),
                 MissingData))
try:
    REGISTRY.render("scan_jsc_vs_x", r)
except MissingData as e:
    check("MissingData.need == 'scan'", e.need == "scan", e.need)

fig_scan = REGISTRY.figure(scan, scan_summary_ids(), cols=2)
check("扫描汇总 2×2 = 4 子图", len(fig_scan.axes) == 4, str(len(fig_scan.axes)))
check("汇总 id 共 4 个", scan_summary_ids() ==
      ("scan_jsc_vs_x", "scan_voc_vs_x", "scan_ff_vs_x", "scan_pce_vs_x"))
plt.close(fig_scan)

scan_png = Path(tempfile.mkdtemp()) / "scan.png"
saved_scan = REGISTRY.savefig(scan, scan_summary_ids(), scan_png, cols=2)
check("扫描汇总可落盘",
      saved_scan.exists() and saved_scan.stat().st_size > 1000,
      f"{saved_scan.stat().st_size if saved_scan.exists() else 0} B")

# ---------------------------------------------------------------------------
# 5c. 结构示意图
# ---------------------------------------------------------------------------
section("结构示意图")

check("结构图不依赖求解数据",
      REGISTRY.get("structure_stack").needs == ()
      and REGISTRY.get("structure_band_align").needs == ())

ax_s = REGISTRY.render("structure_stack", r)
rects = [p for p in ax_s.patches if isinstance(p, Rectangle)]
n_layer = len(r.spec.device.layers)
check("层矩形数 = 层数", len(rects) == n_layer, str(len(rects)))
check("首层宽度 = FTO 厚度 200 nm", np.isclose(rects[0].get_width(), 200.0))
check("宽度之和 = 器件总厚度",
      np.isclose(sum(p.get_width() for p in rects),
                 r.spec.device.total_thick_nm))
check("电极层带斜纹", rects[0].get_hatch() == "//" and rects[-1].get_hatch() == "//")
check("非电极层无斜纹", not rects[1].get_hatch())
check("吸收层单独配色", rects[2].get_facecolor() != rects[1].get_facecolor())
check("层名全部标注",
      all(any(t.get_text() == l.name for t in ax_s.texts)
          for l in r.spec.device.layers))
check("标出入射光", any("AM1.5G" in t.get_text() for t in ax_s.texts))
check("隐藏 y 刻度", len(ax_s.get_yticks()) == 0)
check("x 范围 = 器件厚度",
      np.isclose(ax_s.get_xlim()[1], r.spec.device.total_thick_nm))
check("场表为空也能画结构图",
      hasattr(REGISTRY.render("structure_stack", empty), "patches"))

check("生成参数表", len(ax_s.tables) == 1)
tbl = ax_s.tables[0] if ax_s.tables else None


def _cell(row, col):
    return tbl[row, col].get_text().get_text()


if tbl is not None:
    check("表格 = 表头 + 5 层",
          max(r for r, _ in tbl.get_celld()) == n_layer)
    check("表头含厚度列", _cell(0, 1) == "厚度 (nm)")
    check("厚度列：MAPbI3 = 500", _cell(3, 1) == "500")
    check("Eg 列：MAPbI3 = 1.6", _cell(3, 2) == "1.6")
    check("掺杂列：FTO 有 Nd", "N_D" in _cell(1, 3), _cell(1, 3))
    check("掺杂列：MAPbI3 本征", _cell(3, 3) == "本征", _cell(3, 3))
    check("角色列：FTO = 电极", _cell(1, 4) == "电极")
    check("角色列：MAPbI3 = 吸收层", _cell(3, 4) == "吸收层")
    check("角色列：TiO2 = 传输层", _cell(2, 4) == "传输层")

ax_b = REGISTRY.render("structure_band_align", r)
check("Ec 首值 = −χ(FTO)", np.isclose(ax_b.lines[0].get_ydata()[0], -4.2))
check("Ev 首值 = −(χ+Eg)(FTO)",
      np.isclose(ax_b.lines[1].get_ydata()[0], -(4.2 + 3.7)))
check("阶梯能带 ×2 段/层",
      len(ax_b.lines[0].get_xdata()) == 2 * n_layer)
check("吸收层标出 Eg", any("Eg =" in t.get_text() for t in ax_b.texts))
check("注明平带近似", any("平带近似" in t.get_text() for t in ax_b.texts))
check("能带图有图例", ax_b.get_legend() is not None)
check("能带图 x 范围 = 器件厚度",
      np.isclose(ax_b.get_xlim()[1], r.spec.device.total_thick_nm))

# ---------------------------------------------------------------------------
# 5d. 论文对照
# ---------------------------------------------------------------------------
section("论文对照")

check("对照图依赖 compare", REGISTRY.get("compare_jv").needs == ("compare",))

with tempfile.TemporaryDirectory() as tmp:
    dat = Path(tmp) / "paper_a.dat"
    dat.write_text("# V(mV?) J\n0.0 21.5\n0.4 21.0\n0.8 15.0\n1.1 0.0\n",
                   encoding="utf-8")
    v_dat, j_dat = load_dat(dat)
    check("load_dat 跳过注释行", v_dat.size == 4 and v_dat[0] == 0.0)
    check("load_dat 取第二列", np.isclose(j_dat[1], 21.0))

    entry_a = load_curve(dat, label="Paper A")
    check("load_curve 记录来源", entry_a.source.endswith("paper_a.dat"))
    check("图例含文件名", "paper_a.dat" in entry_a.legend)

    thin = Path(tmp) / "thin.dat"
    thin.write_text("V J\n", encoding="utf-8")
    check("少于 2 行抛 ValueError",
          isinstance(_try(lambda: load_dat(thin), ValueError), ValueError))
    check("文件不存在抛 FileNotFoundError",
          isinstance(_try(lambda: load_curve(Path(tmp) / "none.dat"),
                          FileNotFoundError), FileNotFoundError))

    import openpyxl

    xlsx = Path(tmp) / "paper_b.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["V", "J"])
    for vv, jj in ((0.0, 21.0), (0.5, 20.0), (1.1, 0.0)):
        ws.append([vv, jj])
    wb.save(xlsx)

    v_xl, j_xl = load_xlsx(xlsx)
    check("load_xlsx 跳过表头", v_xl.size == 3 and np.isclose(j_xl[0], 21.0))
    check("load_curve 支持 xlsx", load_curve(xlsx).jv is not None)

check("空 label 抛 ValueError",
      isinstance(_try(lambda: LiteratureEntry(label=""), ValueError), ValueError))
check("既无 jv 又无 PCE 抛 ValueError",
      isinstance(_try(lambda: LiteratureEntry(label="x"), ValueError), ValueError))
check("V/J 长度不符抛 ValueError",
      isinstance(_try(lambda: LiteratureEntry(label="x", jv=([0.0, 1.0],
                                                            [20.0])),
                      ValueError), ValueError))

lit = LiteratureEntry(label="Paper A",
                      jv=(np.array([0.0, 0.5, 1.1]), np.array([21.0, 18.0, 0.0])),
                      PCE=17.5)
ts = ThesisSet(r, (lit,), title="MAPbI3 对照")
check("ThesisSet 长度", len(ts) == 1)
check("空 entries 抛 ValueError",
      isinstance(_try(lambda: ThesisSet(r, []), ValueError), ValueError))

ax_c = REGISTRY.render("compare_jv", ts)
check("仿真 + 文献曲线 + 零电流线", len(ax_c.lines) == 3, str(len(ax_c.lines)))
check("文献曲线数值正确",
      np.allclose(ax_c.lines[1].get_ydata(), lit.jv[1]))
check("仿真曲线用交付电流",
      np.allclose(ax_c.lines[0].get_ydata(), -r.jv.J))
check("图例含文献标签",
      ax_c.get_legend() is not None
      and "Paper A" in [t.get_text() for t in ax_c.get_legend().get_texts()])

ax_bar = REGISTRY.render("compare_pce_bar", ts)
check("柱数 = 本文 + 1 条文献", len(ax_bar.patches) == 2, str(len(ax_bar.patches)))
check("本文柱高 = 仿真 PCE",
      np.isclose(ax_bar.patches[0].get_height(), r.jv.PCE))
check("文献柱高 = 文献 PCE", np.isclose(ax_bar.patches[1].get_height(), 17.5))
check("柱上标出数值", any("17.50" in t.get_text() for t in ax_bar.texts))

ts_nopce = ThesisSet(r, (LiteratureEntry(
    label="NoPCE", jv=(np.array([0.0, 1.0]), np.array([20.0, 0.0]))),))
ax_bar2 = REGISTRY.render("compare_pce_bar", ts_nopce)
check("缺 PCE 显示破折号", any("—" in t.get_text() for t in ax_bar2.texts))

check("单结果不能画对照图",
      isinstance(_try(lambda: REGISTRY.render("compare_jv", r), MissingData),
                 MissingData))

# ---------------------------------------------------------------------------
# 5e. 参考参数兜底
# ---------------------------------------------------------------------------
section("参考参数兜底")

zero = make_result(V=V7, J=np.zeros(7), run_id="stub")
check("全零 J（stub）判定需要兜底", fallback.needs_fallback(zero))
check("正常数据不需要兜底", not fallback.needs_fallback(r))

dark_spec = replace(
    default_problem_spec(),
    optics=replace(default_problem_spec().optics, intensity_mW_cm2=0.0),
)
dark_res = make_result(V=V7, J=np.zeros(7), spec=dark_spec)
check("暗态零电流不兜底", not fallback.needs_fallback(dark_res))

ref = fallback.reference_result(
    default_problem_spec(),
    fallback.ReferenceMetrics(Jsc=22.0, Voc=1.10, FF=0.78),
)
check("合成 Jsc 命中目标", np.isclose(ref.jv.Jsc, 22.0, rtol=1e-3), ref.jv.Jsc)
check("合成 Voc 命中目标", np.isclose(ref.jv.Voc, 1.10, rtol=1e-9), ref.jv.Voc)
check("合成 FF 命中目标", np.isclose(ref.jv.FF, 0.78, atol=2e-3), ref.jv.FF)
check("合成 PCE > 0", ref.jv.PCE > 0, ref.jv.PCE)
check("合成结果标记为参考", fallback.is_reference(ref))
check("合成结果未收敛", not ref.converged)
check("合成场表为空", ref.fields.n_points == 0)
check("参考结果本身不触发兜底", not fallback.needs_fallback(ref))

for ff in (0.60, 0.70, 0.85):
    jv_try = fallback.reference_jv(
        fallback.ReferenceMetrics(Jsc=20.0, Voc=1.0, FF=ff), 100.0)
    check(f"FF={ff} 可反解", abs(jv_try.FF - ff) < 3e-3, jv_try.FF)

check("FF 越界抛 ValueError",
      isinstance(_try(lambda: fallback.ReferenceMetrics(FF=0.95).validate(),
                      ValueError), ValueError))
check("Jsc<=0 抛 ValueError",
      isinstance(_try(lambda: fallback.ReferenceMetrics(Jsc=0.0).validate(),
                      ValueError), ValueError))

check("J-V 图可兜底",
      fallback.supports("cell_jv_curve") and fallback.supports("cell_pv_curve"))
check("剖面图不可兜底", not fallback.supports("cell_efield_profile"))
check("扫描图不可兜底", not fallback.supports("scan_jsc_vs_x"))

ax_ref = fallback.render_reference("cell_jv_curve", zero)
check("兜底图带水印",
      any("非仿真结果" in t.get_text() for t in ax_ref.texts))
check("兜底曲线 Jsc > 0", ax_ref.lines[0].get_ydata()[0] > 0)
check("不可兜底的图抛 MissingData",
      isinstance(_try(lambda: fallback.render_reference("cell_efield_profile",
                                                       zero), MissingData),
                 MissingData))
ax_ref_pv = fallback.render_reference("cell_pv_curve", zero,
                                      metrics=fallback.ReferenceMetrics())
check("P-V 也能兜底", ax_ref_pv.get_ylabel().startswith("功率"))

# ---------------------------------------------------------------------------
# 6. 样式
# ---------------------------------------------------------------------------
section("样式")

axes_style.ensure()
check("unicode_minus 已关闭", matplotlib.rcParams["axes.unicode_minus"] is False)
check("字体列表非空", len(matplotlib.rcParams["font.sans-serif"]) > 0)
check("ensure 幂等", (axes_style.ensure() or True))

plt.close("all")

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
