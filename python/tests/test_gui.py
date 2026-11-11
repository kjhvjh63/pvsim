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
    format_record,
    menu_label,
    plot_menu_tree,
    read_deck,
    write_deck,
)
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
# 4. 求解线程
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
# 5. 窗口冒烟（需要显示器）
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
            check("菜单栏 3 项", app.nametowidget(app["menu"]).index("end") == 2)
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
            check("日志记录了编译", "编译通过" in app.log_panel.buffer.text())
            for _ in range(20):        # 让最后一次 poll 把按钮状态刷回来
                app.update()
                if "disabled" not in app._run_btn.state():
                    break
                time.sleep(0.02)
            check("运行结束后按钮恢复",
                  "disabled" not in app._run_btn.state())

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
        finally:
            app.close()

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
