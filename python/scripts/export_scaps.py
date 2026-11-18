"""把 deck 导出成 SCAPS-1D 对照材料（.script + 参数清单 .md）。

用法::

    python python/scripts/export_scaps.py
    python python/scripts/export_scaps.py decks/device_example.tst \\
        --out-dir refs --name pvsim_mapbi3

产出两个文件：

* ``<name>.script``：SCAPS 脚本面板可直接载入，按与 pvsim 相同的温度、
  光照和电压范围跑 IV，并存成 ``<name>.iv``；
* ``<name>_scaps_params.md``：逐层参数清单，照表填入 SCAPS 的 Layer /
  Doping / Defects 面板。

之后在 SCAPS 里跑完，把 ``.iv`` 交给::

    python python/scripts/compare_reference.py -r refs/<name>.iv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # python/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.dataio.scaps_export import export_scaps   # noqa: E402
from pvsim.script import DeckCompiler                # noqa: E402

DEFAULT_DECK = ROOT.parent / "decks" / "device_example.tst"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="导出 SCAPS-1D 对照脚本与参数清单",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("deck", nargs="?", default=str(DEFAULT_DECK), help="deck (.tst)")
    p.add_argument("--out-dir", default=str(ROOT.parent / "refs"),
                   help="输出目录")
    p.add_argument("--name", default=None, help="产出文件名（默认取 deck 的 run 名）")
    args = p.parse_args(argv)

    try:
        spec = DeckCompiler().compile_text(
            Path(args.deck).read_text(encoding="utf-8"))
    except Exception as exc:                      # noqa: BLE001 - CLI 统一出口
        print(f"[错误] deck 编译失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    name = args.name or spec.output.run_name
    script, table = export_scaps(spec, args.out_dir, iv_name=name)
    print(f"SCAPS 脚本      : {script}")
    print(f"参数清单        : {table}")
    print("")
    print("下一步：")
    print(f"  1. 按 {table.name} 在 SCAPS 里建层栈/掺杂/缺陷；")
    print(f"  2. Script 面板载入 {script.name} 并运行，得到 {name}.iv；")
    print(f"  3. python python/scripts/compare_reference.py -r "
          f"{Path(args.out_dir) / (name + '.iv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
