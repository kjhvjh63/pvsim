"""``.mat`` 材料库测试：解析 / 索引 / 与 DeckCompiler 集成。

Run: python python/tests/test_material_db.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import fields as dc_fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.param import LayerSpec, MaterialDb, MatParseError, parse_mat
from pvsim.param.schema import SpecError
from pvsim.script import DeckCompiler

MAT_DIR = ROOT.parent / "csrc" / "material_db"
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


SAMPLE = """# 演示材料
material name=DEMO
  permittivity=12.5 eg300=1.5D+00 chi3=4.0 \\
  nc300=1e19 nv300=2e19 mun=100 mup=50 \\
  taun0=1e-9 taup0=2e-9 auger=on user.default=DEMO
"""


# ---------------------------------------------------------------------------
# 1. 解析
# ---------------------------------------------------------------------------
section("解析")

specs = parse_mat(SAMPLE)
check("一个文件一条材料", len(specs) == 1, str(len(specs)))
demo = specs[0]
check("材料名", demo.name == "DEMO")
check("材料参数 9 项", len(demo) == 9, str(len(demo)))

params = demo.as_dict()
check("permittivity → eps_r", params["eps_r"] == 12.5)
check("D 指数 eg300 → Eg_eV", params["Eg_eV"] == 1.5, str(params.get("Eg_eV")))
check("chi3 → chi_eV", params["chi_eV"] == 4.0)
check("nc300/nv300 → Nc/Nv",
      params["Nc_cm3"] == 1e19 and params["Nv_cm3"] == 2e19)
check("mun/mup 映射", params["mun_cm2Vs"] == 100.0 and params["mup_cm2Vs"] == 50.0)
check("taun0/taup0 → taun/taup",
      params["taun_s"] == 1e-9 and params["taup_s"] == 2e-9)

extras = dict(demo.extra)
check("未识别键保留在 extra", extras.get("auger") == "on")
check("user.* 键保留", extras.get("user.default") == "DEMO")
check("extra 不进参数", "auger" not in params and "user.default" not in params)

MULTI = """material name=A
  eg=1.1
material name=B
  eg300=2.2 eg=9.9
"""
multi = parse_mat(MULTI)
check("一个文件多条材料", len(multi) == 2)
check("eg 回退可用", multi[0].as_dict()["Eg_eV"] == 1.1)
check("eg300 优先于 eg", multi[1].as_dict()["Eg_eV"] == 2.2)
check("未采用的同义键进 extra",
      ("eg", "9.9") in multi[1].extra, str(multi[1].extra))

check("参数出现在 material 之前报错",
      isinstance(_try(lambda: parse_mat("permittivity=1\n"), MatParseError),
                 MatParseError))
check("缺 name 报错",
      isinstance(_try(lambda: parse_mat("material name=\n permittivity=1\n"),
                      MatParseError), MatParseError))
check("重复参数报错",
      isinstance(_try(lambda: parse_mat(
          "material name=X\n eg=1\n eg=2\n"), MatParseError), MatParseError))
check("非数值报错",
      isinstance(_try(lambda: parse_mat("material name=X\n eg=abc\n"),
                      MatParseError), MatParseError))
check("无任何可识别参数报错",
      isinstance(_try(lambda: parse_mat("material name=X\n auger=on\n"),
                      MatParseError), MatParseError))
check("空文本返回空元组", parse_mat("") == ())

err = _try(lambda: parse_mat("material name=X\n eg=abc\n", source="x.mat"),
           MatParseError)
check("错误信息带来源文件", err is not None and "x.mat" in str(err))

# ---------------------------------------------------------------------------
# 2. 材料库
# ---------------------------------------------------------------------------
section("材料库")

db = MaterialDb(parse_mat(SAMPLE))
check("库长度", len(db) == 1)
check("lookup 命中", db.lookup("DEMO")["Eg_eV"] == 1.5)
check("lookup 大小写不敏感", db.lookup("demo") is not None)
check("lookup 未命中返回 None", db.lookup("NOPE") is None)
check("contains 大小写不敏感", "demo" in db and "DEMO" in db)
check("keys 统一大写", db.keys() == ("DEMO",), str(db.keys()))
check("names 保留原样", db.names() == ("DEMO",))
check("get 未命中抛 KeyError",
      isinstance(_try(lambda: db.get("NOPE"), KeyError), KeyError))
check("重复 add 抛 ValueError",
      isinstance(_try(lambda: db.add(demo), ValueError), ValueError))

check("参数可直接构造 LayerSpec",
      LayerSpec(name="x", thick_nm=10.0, **db.lookup("DEMO")).eps_r == 12.5)

# ---------------------------------------------------------------------------
# 3. 随仓库附带的 .mat
# ---------------------------------------------------------------------------
section("随仓库附带的 .mat")

check("material_db 目录存在", MAT_DIR.is_dir(), str(MAT_DIR))
shipped = MaterialDb.from_directory(MAT_DIR)
check("两个文件共 12 种材料", len(shipped) == 12, str(len(shipped)))
check("含硅/GaAs 等通用材料",
      {"SI", "GAAS", "TIO2", "FTO", "ITO", "AU", "AG"} <= set(shipped.keys()))
check("含钙钛矿扩展材料",
      {"MAPBI3", "SPIRO", "C60", "NIO", "SNO2"} <= set(shipped.keys()))

mapi = shipped.get("MAPBI3").as_dict()
check("MAPBI3 带隙 1.6", mapi["Eg_eV"] == 1.6)
check("MAPBI3 介电 24", mapi["eps_r"] == 24.0)
check("MAPBI3 寿命 8 ns", mapi["taun_s"] == 8e-9 and mapi["taup_s"] == 8e-9)
check("SI 参数", shipped.lookup("si")["Eg_eV"] == 1.12)

allowed = {f.name for f in dc_fields(LayerSpec)}
check("所有参数都是 LayerSpec 字段",
      all(set(s.as_dict()) <= allowed for s in shipped), str(shipped.keys()))

check("load_file 返回条数",
      MaterialDb().load_file(MAT_DIR / "silvaco.mat") == 7)
check("重复加载同名材料报错",
      isinstance(_try(lambda: MaterialDb.from_files(
          MAT_DIR / "silvaco.mat", MAT_DIR / "silvaco.mat"), ValueError),
          ValueError))

# ---------------------------------------------------------------------------
# 4. 与 DeckCompiler 集成
# ---------------------------------------------------------------------------
section("与 DeckCompiler 集成")

DECK_TEXT = DECK_PATH.read_text(encoding="utf-8")
deck_spec = DeckCompiler(material_db=shipped).compile_text(DECK_TEXT)
layers = deck_spec.device.layers
check("用 .mat 编译示例 deck",
      [l.material for l in layers] == ["FTO", "TiO2", "MAPbI3", "Spiro", "Au"])
check("材料参数来自 .mat",
      layers[2].Eg_eV == 1.6 and layers[1].eps_r == 10.0)
check("deck 覆盖仍优先于 .mat 之外的值",
      layers[2].chi_eV == 3.93)

MINIMAL = ("region material=FTO thickness=100\n"
           "region material=MAPbI3 thickness=400\n"
           "region material=Au thickness=80\n")

with tempfile.TemporaryDirectory() as tmp:
    custom = Path(tmp) / "custom.mat"
    custom.write_text(
        "material name=MAPBI3\n"
        "  permittivity=30 eg300=1.75 chi3=3.9 \\\n"
        "  nc300=1e18 nv300=1e18 mun=2 mup=2\n",
        encoding="utf-8",
    )
    custom_db = MaterialDb.from_files(custom)
    custom_spec = DeckCompiler(material_db=custom_db).compile_text(MINIMAL)
    custom_layer = custom_spec.device.layers[1]
    check("自定义 .mat 生效",
          custom_layer.Eg_eV == 1.75 and custom_layer.eps_r == 30.0,
          f"Eg={custom_layer.Eg_eV}, eps={custom_layer.eps_r}")

check("空库回退内建材料表",
      DeckCompiler(material_db=MaterialDb()).compile_text(MINIMAL)
      .device.layers[1].Eg_eV == 1.6)

check("库与内建表都没有时抛 SpecError",
      isinstance(_try(lambda: DeckCompiler(material_db=MaterialDb())
                      .compile_text("region material=UNOBTANIUM thickness=10\n"
                                    + MINIMAL), SpecError), SpecError))

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
