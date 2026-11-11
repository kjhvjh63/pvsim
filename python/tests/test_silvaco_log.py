"""SSF/.log 读写测试：格式 / round-trip / 真实 Silvaco 风格 / 异常。

Run: python python/tests/test_silvaco_log.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pvsim.dataio import LogTable, ResultStore, SilvacoLogReader, SilvacoLogWriter
from pvsim.param import default_problem_spec
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


def _try(fn, exc):
    try:
        fn()
    except exc as e:
        return e
    return None


def make_jv(n=19):
    V = np.linspace(0.0, 1.8, n)
    J = np.linspace(-22.0, 6.0, n)          # 负 → 正：有 Jsc 和 Voc
    return JVData.from_points(V, J, pin_mW_cm2=100.0)


# ---------------------------------------------------------------------------
# 1. 写出格式
# ---------------------------------------------------------------------------
section("写出格式")

jv = make_jv()
writer = SilvacoLogWriter()
text = writer.dumps(jv, run_id="unit_test")

check("首行是 H1", text.splitlines()[0].startswith("<H1>"))
check("含 SSF 2.0 标识", "<H2> SSF 2.0" in text)
check("H3 带 run_id", "<H3> run_id=unit_test" in text)

lines = text.splitlines()
h4 = [l for l in lines if l.startswith("<H4>")]
check("恰好一条 H4 图例", len(h4) == 1, str(len(h4)))
check("图例含三列",
      h4[0].count("  ") == 2 and "Anode Voltage" in h4[0], h4[0])

rows = [l for l in lines if l.startswith("<L>")]
check("数据行数 = 点数", len(rows) == jv.V.size, str(len(rows)))
check("每行三个数值",
      len(rows[0].replace("<L>", "").split()) == 3, rows[0])

with tempfile.TemporaryDirectory() as tmp:
    path = writer.write(jv, Path(tmp) / "nested" / "jv.log", run_id="unit_test")
    check("write 自动建目录", path.exists())
    check("文件非空", path.stat().st_size > 100)
    check("文件以换行结尾", path.read_text(encoding="utf-8").endswith("\n"))

# ---------------------------------------------------------------------------
# 2. 读回与 round-trip
# ---------------------------------------------------------------------------
section("读回与 round-trip")

reader = SilvacoLogReader()
table = reader.parse(text)[0]

check("列名去掉单位",
      table.columns == ("Anode Voltage", "Anode Current", "Power"),
      str(table.columns))
check("单位解析", table.unit_of("Anode Current") == "mA/cm2",
      table.unit_of("Anode Current"))
check("行数一致", table.n_rows == jv.V.size)
check("meta 保留 H1", table.meta.get("H1") == "PVK Simulator", str(table.meta))
check("meta 保留 H2", table.meta.get("H2") == "SSF 2.0")
check("电压列数值", np.allclose(table["Anode Voltage"], jv.V))
check("电流列数值", np.allclose(table["Anode Current"], jv.J))
check("find 精确匹配", table.find("Anode Voltage") == "Anode Voltage")
check("find 子串匹配", table.find("voltage") == "Anode Voltage")
check("find 关键字电流", table.find("current") == "Anode Current")
check("find 找不到返回 None", table.find("humidity") is None)

back = reader.to_jv(table)
check("round-trip Jsc", np.isclose(back.Jsc, jv.Jsc, atol=1e-9),
      f"{back.Jsc} vs {jv.Jsc}")
check("round-trip Voc", np.isclose(back.Voc, jv.Voc, atol=1e-9),
      f"{back.Voc} vs {jv.Voc}")
check("round-trip FF", np.isclose(back.FF, jv.FF, atol=1e-9))
check("round-trip PCE", np.isclose(back.PCE, jv.PCE, atol=1e-9))
check("round-trip 电流数组", np.allclose(back.J, jv.J, rtol=1e-6),
      str(np.max(np.abs(back.J - jv.J))))
check("round-trip 电压数组", np.allclose(back.V, jv.V, rtol=1e-12))

check("sign=-1 翻转电流符号",
      np.isclose(reader.to_jv(table, sign=-1.0).Jsc, -jv.Jsc))
check("pin 可指定",
      np.isclose(reader.to_jv(table, pin_mW_cm2=50.0).PCE, jv.PCE * 2.0))

# ---------------------------------------------------------------------------
# 3. 真实 Silvaco 风格
# ---------------------------------------------------------------------------
section("真实 Silvaco 风格")

ATLAS_TEXT = """<H1> ATLAS
<H2> 5.28.0.R
<H3> 1D MAPbI3 cell
<H4> Anode Voltage     Anode Current
<L>  0.000000e+00      -8.000000e-04
<L>  5.000000e-01      -7.500000e-04
<L>  1.100000e+00       0.000000e+00
"""

atlas = reader.parse(ATLAS_TEXT)[0]
check("多空格分隔的多词列名",
      atlas.columns == ("Anode Voltage", "Anode Current"), str(atlas.columns))
check("无单位时 units 为空", atlas.units == {})

atlas_jv = reader.to_jv(atlas, area_cm2=0.04)
check("A → mA/cm² 按面积换算", np.isclose(atlas_jv.Jsc, 20.0),
      str(atlas_jv.Jsc))
check("Silvaco 风格 Voc", np.isclose(atlas_jv.Voc, 1.1), str(atlas_jv.Voc))
check("面积不同结果不同",
      np.isclose(reader.to_jv(atlas, area_cm2=1.0).Jsc, 0.8))

D_TEXT = """<H4> V (V)  I (A)
<L> 0.0D+00, -4.0D-04
<L> 5.0D-01, -3.0D-04
<L> 1.0D+00, 0.0D+00
"""
d_table = reader.parse(D_TEXT)[0]
check("兼容 D 指数与逗号", d_table.n_rows == 3, str(d_table.n_rows))
check("D 指数数值正确", np.isclose(d_table["I"][0], -4e-4))
check("D 指数 JV 换算", np.isclose(reader.to_jv(d_table, area_cm2=0.04).Jsc, 10.0))

PLAIN_TEXT = """<H4> Anode Voltage  Anode Current
0.0 1.0
0.5 0.5
1.0 0.0
"""
plain = reader.parse(PLAIN_TEXT)[0]
check("无 <L> 前缀的纯数值行", plain.n_rows == 3, str(plain.n_rows))

NO_LEGEND = """<L> 1.0 2.0
<L> 3.0 4.0
"""
fallback = reader.parse(NO_LEGEND)[0]
check("无图例时列名 col1/col2", fallback.columns == ("col1", "col2"),
      str(fallback.columns))
check("无图例仍能取数", np.allclose(fallback["col2"], (2.0, 4.0)))
check("识别不了列名时抛 ValueError",
      isinstance(_try(lambda: reader.to_jv(fallback), ValueError), ValueError))

MISMATCH = """<H4> Anode Current Anode Voltage
<L> 1.0 0.0
"""
mismatch = reader.parse(MISMATCH)[0]
check("列名数与数据列不符时回退 colN",
      mismatch.columns == ("col1", "col2"), str(mismatch.columns))

DUP = """<H4> V (V)  V (V)
<L> 0.0 1.0
"""
dup = reader.parse(DUP)[0]
check("同名列自动去重", dup.columns == ("V", "V_2"), str(dup.columns))

# ---------------------------------------------------------------------------
# 4. 多数据块
# ---------------------------------------------------------------------------
section("多数据块")

MULTI = """<H1> ATLAS
<H4> Anode Voltage  Anode Current
<L> 0.0 1.0
<L> 0.5 0.5
<H4> Anode Voltage  Anode Current  Time
<L> 0.0 2.0 0.1
<L> 0.5 1.0 0.2
"""
blocks = reader.parse(MULTI)
check("识别两个数据块", len(blocks) == 2, str(len(blocks)))
check("第一块 2 列", len(blocks[0].columns) == 2)
check("第二块 3 列", len(blocks[1].columns) == 3)
check("read 取最后一块语义", blocks[-1].n_rows == 2)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "multi.log"
    path.write_text(MULTI, encoding="utf-8")
    check("read_blocks 读文件", len(reader.read_blocks(path)) == 2)
    check("read 返回最后一块", len(reader.read(path).columns) == 3)

# ---------------------------------------------------------------------------
# 5. 异常与边界
# ---------------------------------------------------------------------------
section("异常与边界")

check("空文件无数据块", reader.parse("") == ())

with tempfile.TemporaryDirectory() as tmp:
    empty = Path(tmp) / "empty.log"
    empty.write_text("<H1> ATLAS\n", encoding="utf-8")
    check("只有头没有数据 → read 抛 ValueError",
          isinstance(_try(lambda: reader.read(empty), ValueError), ValueError))
    check("文件不存在抛 FileNotFoundError",
          isinstance(_try(lambda: reader.read(Path(tmp) / "none.log"),
                          FileNotFoundError), FileNotFoundError))

check("LogTable 空列抛 ValueError",
      isinstance(_try(lambda: LogTable(columns=(), data={}), ValueError),
                 ValueError))
check("LogTable 列长不一致抛 ValueError",
      isinstance(_try(lambda: LogTable(
          columns=("a", "b"),
          data={"a": np.zeros(2), "b": np.zeros(3)}), ValueError), ValueError))

check("非正面积抛 ValueError",
      isinstance(_try(lambda: reader.to_jv(atlas, area_cm2=0.0), ValueError),
                 ValueError))

# ---------------------------------------------------------------------------
# 6. 真实内核集成
# ---------------------------------------------------------------------------
section("真实内核集成")

result = Simulator().run(default_problem_spec(), run_id="ssf_case")
with tempfile.TemporaryDirectory() as tmp:
    path = writer.write(result.jv, Path(tmp) / "kernel.log",
                        run_id=result.run_id)
    loaded_jv = reader.read_jv(path)
    check("内核 J-V 可写可读", path.exists())
    check("电压数组一致", np.allclose(loaded_jv.V, result.jv.V, rtol=1e-12))
    check("电流数组一致", np.allclose(loaded_jv.J, result.jv.J, rtol=1e-6))
    check("Jsc 一致", np.isclose(loaded_jv.Jsc, result.jv.Jsc, atol=1e-12))

    # output format=ssf 时 ResultStore 顺手写出 jv.log
    ssf_result = replace(result, spec=replace(
        result.spec, output=replace(result.spec.output, format="ssf")))
    store = ResultStore(Path(tmp) / "workspace")
    run_dir = store.save(ssf_result)
    check("format=ssf 落盘 jv.log", (run_dir / "jv.log").exists(),
          str(sorted(p.name for p in run_dir.iterdir())))
    check("run 目录里的 .log 可读回",
          np.allclose(reader.read_jv(run_dir / "jv.log").J, result.jv.J,
                      rtol=1e-6))

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
