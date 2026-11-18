"""GUI 测试：纯逻辑（无 Tk）+ 窗口冒烟（有显示器时）。

Run: python python/tests/test_gui.py
"""
from __future__ import annotations

import queue
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.gui import (
    EVT_DONE,
    EVT_ERROR,
    EVT_LOG,
    EVT_POINT,
    LogBuffer,
    SolveWorker,
    classify,
    format_record,
    menu_label,
    plot_menu_tree,
    read_deck,
    write_deck,
)
from pvsim.gui import dialogs
from pvsim.gui.log_panel import LEVEL_INFO, LEVEL_WARN
from pvsim.viz import REGISTRY

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


class FakeSim:
    """只发事件、不做计算，用于验证线程/队列协议。"""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def run(self, spec, hook=None, run_id=None):
        if hook is not None:
            hook.on_log(LEVEL_INFO, "hello")
            hook.on_point(0, 2, "P0")
        if self.fail:
            raise RuntimeError("boom")
        return "RESULT"


def drain(worker):
    events = []
    while not worker.out.empty():
        events.append(worker.out.get_nowait())
    return events


MINIMAL_DECK = (
    "region material=FTO thickness=100\n"
    "region material=MAPbI3 thickness=400\n"
    "region material=Au thickness=80\n"
    "solve vstart=0 vfinal=1.8 vstep=0.1\n"
)


# ---------------------------------------------------------------------------
# 1. 菜单结构
# ---------------------------------------------------------------------------
section("菜单结构")

tree = plot_menu_tree(REGISTRY)
check("分组数与注册表一致", len(tree) == len(REGISTRY.groups()))
check("分组顺序一致", tuple(g for g, _ in tree) == REGISTRY.groups(),
      str(tuple(g for g, _ in tree)))
check("电学组 6 项", len(dict(tree)["电学"]) == 6)
check("菜单项 id 不重复",
      len({i.id for _, items in tree for i in items}) == len(REGISTRY))
check("菜单标签用 title", menu_label(REGISTRY.get("cell_jv_curve")) == "J-V 曲线")

# ---------------------------------------------------------------------------
# 2. 日志缓冲
# ---------------------------------------------------------------------------
section("日志缓冲")

buf = LogBuffer(maxlen=3)
buf.append(LEVEL_INFO, "第一条")
buf.append(LEVEL_WARN, "第二条")
check("缓冲长度", len(buf) == 2)
check("记录字段", buf.records[0].message == "第一条"
      and buf.records[0].level == LEVEL_INFO)
check("format 含级别",
      format_record(buf.records[1]).endswith("第二条")
      and "WARN" in format_record(buf.records[1]),
      format_record(buf.records[1]))
check("text 多行", buf.text().count("\n") == 1)

for i in range(5):
    buf.append(LEVEL_INFO, f"x{i}")
check("maxlen 生效", len(buf) == 3, str(len(buf)))
check("保留最新", buf.records[-1].message == "x4")

buf.clear()
check("clear 生效", len(buf) == 0 and buf.text() == "")

# ---------------------------------------------------------------------------
# 3. deck 文件读写
# ---------------------------------------------------------------------------
section("deck 文件读写")

with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / "sub" / "demo.tst"
    write_deck(target, MINIMAL_DECK)
    check("write_deck 建目录", target.exists())
    check("read_deck 往返", read_deck(target) == MINIMAL_DECK)
    check("统一使用 \\n 换行", "\r" not in target.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 4. DeckBuild 风格对话框（纯函数部分，无 Tk）
# ---------------------------------------------------------------------------
section("DeckBuild 对话框纯函数")

from pvsim.script import ast_nodes as ast
from pvsim.script.parser import parse_deck

mesh_stmt = dialogs.build_mesh_stmt(201, 1.5, 1.0, 300)[0]
check("mesh 语句回读节点数",
      parse_deck(mesh_stmt + "\n").first(ast.MeshStmt).nodes == 201, mesh_stmt)

phys = dialogs.build_physics_stmts("srh", 80.0, 100, True)
phys_deck = parse_deck("\n".join(phys["models"] + phys["optics"]) + "\n")
check("physics 回读模型", phys_deck.first(ast.ModelStmt).models == ("srh",))
check("physics 回读光强",
      phys_deck.first(ast.OpticsStmt).intensity == 80.0)
check("physics 带隙掩模", phys_deck.first(ast.OpticsStmt).bandgap_mask is True)

sol = dialogs.build_solve_stmts("NEWTON", 80, 1e-7, 0.8, 0.0, 1.2, 0.05)
sol_deck = parse_deck("\n".join(sol["method"] + sol["solve"]) + "\n")
check("solve 回读方法", sol_deck.first(ast.MethodStmt).method == "NEWTON")
check("solve 回读终点", sol_deck.first(ast.SolveStmt).v_stop == 1.2)
check("solve 回读步长", sol_deck.first(ast.SolveStmt).v_step == 0.05)

out_stmt = dialogs.build_output_stmt(["X", "PSI", "CB", "PSI"], "ssf", "demo")[0]
out_deck = parse_deck(out_stmt + "\n").first(ast.OutputStmt)
check("output 去重且保序", out_deck.fields == ("X", "PSI", "CB"), str(out_deck.fields))
check("output 格式/run", out_deck.format == "ssf" and out_deck.run_name == "demo")

BASE = ("mesh nodes=101\n"
        "region material=FTO thickness=100\n"
        "region material=MAPbI3 thickness=400\n"
        "region material=Au thickness=80\n"
        "solve vstart=0 vfinal=1.8 vstep=0.1\n")
replaced = dialogs.replace_statements(BASE, [(("solve",), ["solve vstart=0 vfinal=1.0 vstep=0.2"])])
check("替换已有语句", "vfinal=1.0" in replaced and "vfinal=1.8" not in replaced)
check("替换不动其它语句", "mesh nodes=101" in replaced and "thickness=400" in replaced)
appended = dialogs.replace_statements(BASE, [(("output",), ["output fields=X,PSI"])])
check("缺失语句追加到末尾",
      appended.strip().endswith("output fields=X,PSI"), appended)
cont_text = "mesh nodes=101 \\\n  interface.refine=1\nregion material=Au thickness=80\n"
cont_repl = dialogs.replace_statements(cont_text, [(("mesh",), ["mesh nodes=51"])])
check("替换吞掉续行块",
      "nodes=51" in cont_repl and "interface.refine" not in cont_repl, cont_repl)

cur = dialogs.current_values(BASE)
check("读取现有 nodes", cur["nodes"] == 101)
check("读取现有扫描", cur["v_stop"] == 1.8 and cur["v_step"] == 0.1)
check("坏 deck 也能给默认值", dialogs.current_values('title "未闭合\n')["nodes"] == 101)

spans = classify("mesh nodes=101  # 注释\nregion material=Au\n")
kinds = [s[0] for s in spans]
check("高亮识别关键字", "keyword" in kinds, str(kinds))
check("高亮识别参数", "param" in kinds, str(kinds))
check("高亮识别数值", "number" in kinds, str(kinds))
check("高亮识别注释", "comment" in kinds, str(kinds))
mesh_spans = [s for s in classify("mesh nodes=101\n") if s[0] == "keyword"]
check("关键字 span 正好覆盖 'mesh'",
      bool(mesh_spans) and mesh_spans[0][2] - mesh_spans[0][1] == 4,
      str(mesh_spans))

# ---------------------------------------------------------------------------
# 5. 求解线程
# ---------------------------------------------------------------------------
section("求解线程")

worker = SolveWorker(object(), FakeSim())
worker.start()
worker.join(3.0)
check("线程正常退出", not worker.is_alive())

events = drain(worker)
check("事件顺序 log → point → done",
      [e.kind for e in events] == [EVT_LOG, EVT_POINT, EVT_DONE],
      str([e.kind for e in events]))
check("日志事件带级别", events[0].level == LEVEL_INFO and events[0].payload == "hello")
check("进度事件带序号/总数",
      events[1].index == 0 and events[1].total == 2)
check("结果事件带 payload", events[2].payload == "RESULT")

bad = SolveWorker(object(), FakeSim(fail=True))
bad.start()
bad.join(3.0)
bad_events = drain(bad)
check("异常变成 error 事件",
      [e.kind for e in bad_events] == [EVT_LOG, EVT_POINT, EVT_ERROR],
      str([e.kind for e in bad_events]))
check("error 事件保留异常", isinstance(bad_events[-1].payload, RuntimeError))

# ---------------------------------------------------------------------------
# 6. 窗口冒烟（需要显示器）
# ---------------------------------------------------------------------------
section("窗口冒烟")

try:
    probe = tk.Tk()
    probe.destroy()
    has_display = True
except Exception as exc:          # noqa: BLE001 - 无显示环境直接跳过
    has_display = False
    print(f"  (跳过：{type(exc).__name__}: {exc})")

if has_display:
    from pvsim.gui import App

    with tempfile.TemporaryDirectory() as tmp:
        app = App(workspace=Path(tmp))
        try:
            app.withdraw()          # 测试时不弹窗
            check("标题", "pvsim" in app.title())
            check("绘图下拉项数量 = 注册表",
                  len(app._plot_box.cget("values")) == len(REGISTRY),
                  str(len(app._plot_box.cget("values"))))
            check("菜单栏 7 项（文件/网格/物理/求解/输出/绘图/帮助）",
                  app.nametowidget(app["menu"]).index("end") == 6,
                  str(app.nametowidget(app["menu"]).index("end")))
            check("右侧 4 个页签（绘图/结构/结果/日志）",
                  len(app.notebook.tabs()) == 4, str(len(app.notebook.tabs())))
            check("缺少示例 deck 时仍可启动", app.result is None)

            app.editor.set_text(MINIMAL_DECK)
            app.run_current()

            deadline = time.time() + 30.0
            while time.time() < deadline:
                app.update()
                if app.result is not None and not (
                        app._worker is not None and app._worker.is_alive()):
                    break
                time.sleep(0.02)

            check("运行得到 SimResult", app.result is not None)
            check("偏压点数量正确",
                  len(getattr(app.result, "points", ())) == 19,
                  str(len(getattr(app.result, "points", ()))))
            check("结构树填入 3 层",
                  len(app.structure_tree.get_children()) == 3,
                  str(len(app.structure_tree.get_children())))
            check("结果表填入 19 行",
                  len(app.result_tree.get_children()) == 19,
                  str(len(app.result_tree.get_children())))
            check("日志记录了编译", "编译通过" in app.log_panel.buffer.text())
            for _ in range(20):        # 让最后一次 poll 把按钮状态刷回来
                app.update()
                if "disabled" not in app._run_btn.state():
                    break
                time.sleep(0.02)
            check("运行结束后按钮恢复",
                  "disabled" not in app._run_btn.state())

            app.editor.set_text(MINIMAL_DECK)
            app.check_compile()
            check("编译检查刷新结构树",
                  len(app.structure_tree.get_children()) == 3,
                  str(len(app.structure_tree.get_children())))
            check("编译检查更新状态栏",
                  "编译通过" in app._status.get(), app._status.get())

            app.draw("cell_jv_curve")
            check("绘图产生 Axes", len(app.figure.axes) == 1)
            check("J-V 曲线已画", len(app.figure.axes[0].lines) >= 2,
                  str(len(app.figure.axes[0].lines)))
            check("状态栏更新", "J-V" in app._status.get(), app._status.get())

            app.draw("cell_band_diagram")
            check("缺场量时给出说明",
                  "缺少数据" in app._status.get(), app._status.get())
            check("缺场量时不画空曲线",
                  len(app.figure.axes[0].lines) == 0)

            app.result = None
            app.draw("cell_jv_curve")
            check("无结果时用参考参数兜底",
                  "参考参数" in app._status.get(), app._status.get())
            check("兜底图上带水印",
                  any("非仿真结果" in t.get_text()
                      for t in app.figure.axes[0].texts))

            from pvsim.gui.dialogs import FormDialog
            dlg = FormDialog(app, "测试表单", [("val", "数值", 1, "int")])
            dlg._vars["val"].set("7")
            dlg._ok()
            check("FormDialog 收集数值", dlg.result == {"val": 7}, str(dlg.result))

            app.editor.set_text(MINIMAL_DECK)
            app._apply_edits(
                [(("solve",), ["solve vstart=0 vfinal=1.0 vstep=0.5"])], "测试")
            check("对话框回写 deck 文本",
                  "vfinal=1.0" in app.editor.get_text()
                  and "vfinal=1.8" not in app.editor.get_text())
            check("编辑器高亮标签已注册",
                  "keyword" in app.editor.text.tag_names()
                  and app.editor.text.tag_ranges("keyword"))
        finally:
            app.close()

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
