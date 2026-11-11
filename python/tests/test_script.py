"""deck 解析 / 编译测试：词法 → AST → ProblemSpec → 真实内核。

Run: python python/tests/test_script.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.param.schema import ProblemSpec, SpecError
from pvsim.script import DeckCompiler, DeckSyntaxError, parse_deck, tokenize
from pvsim.script import ast_nodes as ast
from pvsim.script.lexer import EOF, EQ, NEWLINE, NUMBER, STRING, WORD

DECK_PATH = ROOT.parent / "decks" / "device_example.tst"

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


MINIMAL = (
    "region material=FTO thickness=100\n"
    "region material=MAPbI3 thickness=400\n"
    "region material=Au thickness=80\n"
)


# ---------------------------------------------------------------------------
# 1. 词法
# ---------------------------------------------------------------------------
section("词法")

toks = tokenize("mesh nodes=101\n")
check("token 种类序列",
      [t.kind for t in toks] == [WORD, WORD, EQ, NUMBER, NEWLINE, EOF],
      str([t.kind for t in toks]))
check("数值 token", toks[3].value == "101")
check("首个 token 行号 = 1", toks[0].line == 1)

check("负数是一个 token",
      [t.value for t in tokenize("x=-1.5e-3\n") if t.kind == NUMBER] == ["-1.5e-3"])

check("注释被忽略",
      [t.value for t in tokenize("# comment\nmesh\n") if t.kind == WORD] == ["mesh"])

cont = tokenize("mesh \\\n  nodes=101\n")
check("续行合成一条语句",
      [t.kind for t in cont].count(NEWLINE) == 1,
      str([t.kind for t in cont]))

quoted = tokenize('title "MAPbI3 cell"\n')
check("双引号字符串", quoted[1].kind == STRING and quoted[1].value == "MAPbI3 cell")

check("单引号字符串", tokenize("title 'abc'\n")[1].value == "abc")

check("字符串未闭合抛 DeckSyntaxError",
      isinstance(_try(lambda: tokenize('title "abc\n'), DeckSyntaxError),
                 DeckSyntaxError))
check("非法字符抛 DeckSyntaxError",
      isinstance(_try(lambda: tokenize("mesh @ 1\n"), DeckSyntaxError),
                 DeckSyntaxError))
check("续行符不在行尾抛 DeckSyntaxError",
      isinstance(_try(lambda: tokenize("mesh \\ x\n"), DeckSyntaxError),
                 DeckSyntaxError))

# ---------------------------------------------------------------------------
# 2. 解析
# ---------------------------------------------------------------------------
section("解析")

DECK_TEXT = DECK_PATH.read_text(encoding="utf-8")
check("示例 deck 存在", DECK_PATH.exists(), str(DECK_PATH))
deck = parse_deck(DECK_TEXT)

check("示例 deck 无警告", not deck.warnings, str(deck.warnings))
check("title 解析", deck.title == "MAPbI3 NIP reference cell", deck.title)
check("语句总数 = 15", len(deck) == 15, str(len(deck)))

regions = deck.of(ast.RegionStmt)
check("5 个 region", len(regions) == 5, str(len(regions)))
check("region 顺序与材料",
      [r.material for r in regions] == ["FTO", "TiO2", "MAPbI3", "Spiro", "Au"],
      str([r.material for r in regions]))
check("region 厚度",
      [r.thickness_nm for r in regions] == [200.0, 40.0, 500.0, 200.0, 80.0],
      str([r.thickness_nm for r in regions]))
check("region 编号", [r.number for r in regions] == [1, 2, 3, 4, 5])

dopings = deck.of(ast.DopingStmt)
check("2 条 doping", len(dopings) == 2)
check("doping 类型 n/p", [d.kind for d in dopings] == ["n", "p"])
check("doping 浓度", [d.conc for d in dopings] == [1e20, 1e20])
check("doping 目标 region", [d.region for d in dopings] == [1, 5])

mat = deck.first(ast.MaterialStmt)
check("material 参数映射为 LayerSpec 字段",
      mat.as_dict().get("Eg_eV") == 1.6 and mat.as_dict().get("taun_s") == 8e-9,
      str(mat.as_dict()))
check("material 覆盖 9 个参数", len(mat.as_dict()) == 9, str(len(mat.as_dict())))

optics = deck.first(ast.OpticsStmt)
check("optics 光强", optics.intensity == 100.0)
check("optics 反射次数", optics.bounces == 1)
check("optics 带隙掩模标志", optics.bandgap_mask is True)

method = deck.first(ast.MethodStmt)
check("method newton", method.method == "NEWTON", str(method.method))
check("method maxiter", method.max_iter == 60)
check("method tol", method.tol == 1e-8)
check("method damping", method.damping == 1.0)

solve = deck.first(ast.SolveStmt)
check("solve vstart", solve.v_start == 0.0)
check("solve vfinal", solve.v_stop == 1.8)
check("solve vstep", solve.v_step == 0.05)

out = deck.first(ast.OutputStmt)
check("output 场量",
      out.fields == ("X", "PSI", "N", "P", "EFIELD", "JTOTAL", "G_OPT", "CB", "VB"),
      str(out.fields))
check("output run 名", out.run_name == "mapbi3_ref")
check("output format 解析",
      parse_deck("output fields=X format=SSF\n").first(ast.OutputStmt).format == "SSF")

check("log 路径", deck.first(ast.LogStmt).path == "out/jv.log")
check("models 解析", deck.first(ast.ModelStmt).models == ("srh",))
check("Deck.has/first/last 可用",
      deck.has(ast.MeshStmt)
      and deck.first(ast.MeshStmt).nodes == 101
      and deck.last(ast.RegionStmt).material == "Au")

unknown = parse_deck("wibble foo=1\nmesh nodes=51\n")
check("未知语句只警告", len(unknown.of(ast.UnknownStmt)) == 1)
check("未知语句产生 warning",
      any("wibble" in w for w in unknown.warnings), str(unknown.warnings))
check("未知语句不影响后续解析",
      unknown.first(ast.MeshStmt).nodes == 51)

unknown_key = parse_deck("mesh nodes=51 bogus=2\n")
check("未知参数只警告",
      any("bogus" in w for w in unknown_key.warnings), str(unknown_key.warnings))
check("未知参数不丢已知参数", unknown_key.first(ast.MeshStmt).nodes == 51)

ci = parse_deck("MESH NODES=51\nREGION MATERIAL=FTO THICKNESS=100\n")
check("关键字大小写不敏感",
      ci.first(ast.MeshStmt).nodes == 51
      and ci.first(ast.RegionStmt).thickness_nm == 100.0)

flag_bool = parse_deck("optics bandgap.mask=false\n")
check("bandgap.mask=false 解析", flag_bool.first(ast.OpticsStmt).bandgap_mask is False)

nd_form = parse_deck("doping region=1 nd=1e18\n")
check("doping nd= 等价 n 型",
      nd_form.first(ast.DopingStmt).kind == "n"
      and nd_form.first(ast.DopingStmt).conc == 1e18)

models = parse_deck("models srh auger\n")
check("models 多值", models.first(ast.ModelStmt).models == ("srh", "auger"))

contact = parse_deck("contact name=cathode region=5 workfunction=4.2\n")
c = contact.first(ast.ContactStmt)
check("contact 解析",
      c.name == "cathode" and c.region == 5 and c.workfunction == 4.2)

check("缺值抛 DeckSyntaxError",
      isinstance(_try(lambda: parse_deck("mesh nodes=\n"), DeckSyntaxError),
                 DeckSyntaxError))
check("重复参数抛 DeckSyntaxError",
      isinstance(_try(lambda: parse_deck("mesh nodes=1 nodes=2\n"),
                      DeckSyntaxError), DeckSyntaxError))
check("doping 同时 n/p 抛 DeckSyntaxError",
      isinstance(_try(lambda: parse_deck("doping n.type p.type conc=1e15\n"),
                      DeckSyntaxError), DeckSyntaxError))

# ---------------------------------------------------------------------------
# 3. 编译
# ---------------------------------------------------------------------------
section("编译")

spec = DeckCompiler().compile(deck)
layers = spec.device.layers

check("5 层", len(layers) == 5)
check("层名 = 材料名",
      [l.name for l in layers] == ["FTO", "TiO2", "MAPbI3", "Spiro", "Au"])
check("层厚", [l.thick_nm for l in layers] == [200.0, 40.0, 500.0, 200.0, 80.0])
check("首末层是接触层",
      [l.is_contact for l in layers] == [True, False, False, False, True],
      str([l.is_contact for l in layers]))
check("FTO 掺 n 1e20", layers[0].Nd_cm3 == 1e20 and layers[0].Na_cm3 == 0.0)
check("Au 掺 p 1e20", layers[4].Na_cm3 == 1e20 and layers[4].Nd_cm3 == 0.0)
check("MAPbI3 保持本征",
      layers[2].Nd_cm3 == 0.0 and layers[2].Na_cm3 == 0.0)
check("material 覆盖生效",
      layers[2].Eg_eV == 1.6 and layers[2].chi_eV == 3.93
      and layers[2].taun_s == 8e-9)
check("器件网格点 101", spec.device.nodes == 101)
check("界面加密 1.0", spec.device.interface_refine == 1.0)
check("温度 300 K", spec.device.T_K == 300.0)

check("光强 100 mW/cm²", spec.optics.intensity_mW_cm2 == 100.0)
check("bounces=1", spec.optics.bounces == 1)
check("带隙掩模开", spec.optics.bandgap_mask is True)
check("光谱非空", spec.optics.wl_nm.size > 0)

check("求解方法 NEWTON", spec.sweep.solver_opts.method == "NEWTON")
check("max_iter=60", spec.sweep.solver_opts.max_iter == 60)
check("res_tol=1e-8", spec.sweep.solver_opts.res_tol == 1e-8)
check("damping=1.0", spec.sweep.solver_opts.damping == 1.0)

check("扫描 0→1.8 V", spec.sweep.v_start == 0.0 and spec.sweep.v_stop == 1.8)
check("由 vstep 推出 37 点", spec.sweep.n_points == 37, str(spec.sweep.n_points))

check("输出场量含 CB/VB",
      "CB" in spec.output.fields and "VB" in spec.output.fields)
check("run 名", spec.output.run_name == "mapbi3_ref")

fatal, warnings = spec.validate_notes()
check("编译结果无致命问题", not fatal, str(fatal))
check("编译结果无警告", not warnings, str(warnings))
check("MaterialSet 记录材料参数",
      "MAPBI3" in dict(spec.material.overrides), str(spec.material.overrides))

round_trip = ProblemSpec.from_json(spec.to_json())
check("JSON round-trip", round_trip.to_dict() == spec.to_dict())

minimal = DeckCompiler().compile_text(MINIMAL)
check("compile_text 一步到位", len(minimal.device.layers) == 3)
check("缺省扫描 37 点", minimal.sweep.n_points == 37)
check("缺省光学 AM1.5G", minimal.optics.intensity_mW_cm2 == 100.0)
check("缺省输出格式 npy", minimal.output.format == "npy")

ssf_spec = DeckCompiler().compile_text(MINIMAL + "output fields=X format=ssf\n")
check("deck 可指定 ssf 输出", ssf_spec.output.format == "ssf")

custom_db = {"CUSTOM": dict(eps_r=5.0, Eg_eV=1.4, chi_eV=3.8,
                            Nc_cm3=1e19, Nv_cm3=1e19,
                            mun_cm2Vs=2.0, mup_cm2Vs=2.0)}
custom_spec = DeckCompiler(material_db=custom_db).compile_text(
    "region material=FTO thickness=100\n"
    "region material=CUSTOM thickness=300\n"
    "region material=Au thickness=80\n"
)
check("外部 material_db 优先",
      custom_spec.device.layers[1].Eg_eV == 1.4,
      str(custom_spec.device.layers[1].Eg_eV))

messages = []
DeckCompiler(log=messages.append).compile(parse_deck("wibble\n" + MINIMAL))
check("解析警告转发到 log", any("wibble" in m for m in messages),
      str(messages))

# ---------------------------------------------------------------------------
# 4. 错误路径
# ---------------------------------------------------------------------------
section("错误路径")


def compile_err(text):
    return _try(lambda: DeckCompiler().compile_text(text), SpecError)


check("缺 region 抛 SpecError",
      isinstance(compile_err("mesh nodes=51\n"), SpecError))
check("缺 thickness 抛 SpecError",
      isinstance(compile_err("region material=FTO\n" + MINIMAL), SpecError))
check("未知材料抛 SpecError",
      isinstance(compile_err("region material=UNOBTANIUM thickness=100\n"
                             + MINIMAL), SpecError))
check("region 编号重复抛 SpecError",
      isinstance(compile_err("region num=1 material=FTO thickness=100\n"
                             "region num=1 material=Au thickness=80\n"),
                 SpecError))
check("doping 指向不存在 region 抛 SpecError",
      isinstance(compile_err("doping region=9 n.type conc=1e15\n" + MINIMAL),
                 SpecError))
check("contact 名字非法抛 SpecError",
      isinstance(compile_err("contact name=gate\n" + MINIMAL), SpecError))
check("method 名非法抛 SpecError",
      isinstance(compile_err("method method=FOO\n" + MINIMAL), SpecError))
check("output 场量非法抛 SpecError",
      isinstance(compile_err("output fields=X,BOGUS\n" + MINIMAL), SpecError))
check("vstep<=0 抛 SpecError",
      isinstance(compile_err("solve vstep=-0.1\n" + MINIMAL), SpecError))
check("nodes<3 抛 SpecError",
      isinstance(compile_err("mesh nodes=2\n" + MINIMAL), SpecError))
check("spectrum 不支持抛 SpecError",
      isinstance(compile_err("optics spectrum=astm\n" + MINIMAL), SpecError))
check("output format 非法抛 SpecError",
      isinstance(compile_err("output format=json\n" + MINIMAL), SpecError))
check("未知场量信息里列出可用项", (lambda e: e is not None and "EFIELD" in str(e.details))(
    compile_err("output fields=X,BOGUS\n" + MINIMAL)))

# ---------------------------------------------------------------------------
# 5. 端到端：deck → 内核 → J-V
# ---------------------------------------------------------------------------
section("端到端（真实内核）")

from pvsim.solver import Simulator

result = Simulator().run(spec)
check("跑出 37 个偏压点", len(result.points) == 37, str(len(result.points)))
check("run_id 带 deck 的 run 名", result.run_id.startswith("mapbi3_ref"),
      result.run_id)
check("按 deck 输出 9 个场量", len(result.fields.names) == 9,
      str(result.fields.names))
check("场表形状与网格一致",
      all(arr.shape == (37, result.fields.n_node)
          for arr in result.fields.data.values()))
check("J-V 数组与点数一致", result.jv.V.size == 37)

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
