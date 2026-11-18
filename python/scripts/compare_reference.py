"""P2 外部对照：把本平台仿真结果与 SCAPS / Silvaco 参考曲线逐项比对。

用法::

    python python/scripts/compare_reference.py -r refs/scaps_case.iv
    python python/scripts/compare_reference.py decks/device_example.tst \\
        -r refs/atlas_case.log --nodes 201 --jsc-tol 3 --voc-tol 20
    python python/scripts/compare_reference.py -r refs/case.csv --out out/compare.md

参考文件格式
------------
* ``.log`` / ``.ssf``：Silvaco SSF（单位自动换算）；
* ``.iv`` / ``.dat`` / ``.txt`` / ``.csv``：两列或多列文本，有表头时按列名
  挑电压/电流列，没有就取前两列；``#`` 之后为注释。

符号约定由 :mod:`pvsim.dataio.reference` 统一处理：外部短路电流为正、
内核为负，读取后自动归一，比较时不会再受符号影响。

退出码：0 = 全部指标与曲线落在容差内；1 = 有超差；2 = 数据/编译错误。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # python/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace                   # noqa: E402

from pvsim.dataio import SilvacoLogWriter, load_reference   # noqa: E402
from pvsim.dataio.reference import compare        # noqa: E402
from pvsim.script import DeckCompiler             # noqa: E402
from pvsim.solver.simulator import Simulator      # noqa: E402

DEFAULT_DECK = ROOT.parent / "decks" / "device_example.tst"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="本平台仿真 vs SCAPS/Silvaco 参考曲线的量化对照",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("deck", nargs="?", default=str(DEFAULT_DECK),
                   help="仿真 deck（.tst）")
    p.add_argument("-r", "--reference", required=True,
                   help="参考 J-V 文件（.log/.ssf/.iv/.dat/.txt/.csv）")
    p.add_argument("--nodes", type=int, default=None,
                   help="覆盖 deck 的网格点数（例如 201）")
    p.add_argument("--pin", type=float, default=None,
                   help="覆盖入射功率 mW/cm²（默认沿用 deck 的 optics intensity）")
    p.add_argument("--jsc-tol", type=float, default=5.0, help="Jsc 相对容差 %%")
    p.add_argument("--voc-tol", type=float, default=30.0, help="Voc 绝对容差 mV")
    p.add_argument("--ff-tol", type=float, default=0.03, help="FF 绝对容差")
    p.add_argument("--pce-tol", type=float, default=8.0, help="PCE 相对容差 %%")
    p.add_argument("--rmse-tol", type=float, default=1.0,
                   help="J-V 曲线 RMSE 容差 mA/cm²")
    p.add_argument("--save-sim", default=None,
                   help="把本次仿真曲线另存为 SSF .log（可用 DeckBuild 打开）")
    p.add_argument("--out", default=None, help="把对照报告写成 Markdown")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        spec = DeckCompiler().compile_text(
            Path(args.deck).read_text(encoding="utf-8"))
        if args.nodes is not None:
            spec = replace(spec, device=replace(spec.device, nodes=int(args.nodes)))
        if args.pin is not None:
            spec = replace(spec, optics=replace(spec.optics,
                                                intensity_mW_cm2=float(args.pin)))
        result = Simulator().run(spec)
    except Exception as exc:                      # noqa: BLE001 - CLI 统一出口
        print(f"[错误] 仿真失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    sim_jv = result.jv
    pin = float(spec.optics.intensity_mW_cm2)
    try:
        ref_jv = load_reference(args.reference, pin_mW_cm2=pin)
    except Exception as exc:                      # noqa: BLE001
        print(f"[错误] 参考曲线读取失败：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    try:
        report = compare(
            sim_jv, ref_jv,
            jsc_tol_pct=args.jsc_tol,
            voc_tol_mV=args.voc_tol,
            ff_tol_abs=args.ff_tol,
            pce_tol_pct=args.pce_tol,
            rmse_tol=args.rmse_tol,
        )
    except ValueError as exc:
        print(f"[错误] 无法对照：{exc}", file=sys.stderr)
        return 2

    lines = [
        f"deck      : {args.deck}",
        f"reference : {args.reference}",
        f"仿真 run   : {result.run_id}",
        f"参考 PCE   : {ref_jv.PCE:.3f} %   Jsc={ref_jv.Jsc:.4f} "
        f"Voc={ref_jv.Voc:.4f} FF={ref_jv.FF:.4f}",
        "",
    ] + report.lines()

    print("\n".join(lines))

    if args.save_sim:
        path = SilvacoLogWriter().write(
            sim_jv, Path(args.save_sim), title=spec.output.run_name,
            run_id=result.run_id)
        print(f"\n仿真曲线已保存：{path}")

    if args.out:
        md = ["# 外部对照报告", "", "```text", *lines, "```", ""]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md), encoding="utf-8", newline="\n")
        print(f"报告已写入：{out}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
