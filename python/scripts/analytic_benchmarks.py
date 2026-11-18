"""解析基准与外部对照。

不依赖任何外部软件，用几个可解析/普适的物理关系验证求解结果：
  1. 细致平衡上限：Jsc ≤ q∫G dx（吸收的光子数上限）
  2. 暗态理想因子：指数区斜率给出的 n_id 应落在 1–2
  3. 理想因子的网格无关性：101/refine=1 与 201/refine=2 之差 < 0.05
  4. Jsc 与光强近似线性（50/100 mW/cm² 之比与 2 的偏差 < 3%）
  5. 吸收层内总电流守恒：JTOTAL 沿 x 的展布 < 1e-3
  6. 暗态零偏无净电流：|J(0)| < 1e-8 mA/cm²

--export DIR 会把 1 sun 的 J-V 写成 SSF/.log（DeckBuild 可直接打开），
用于和 SCAPS / Silvaco 等外部结果逐点对照。

用法:
  python python/scripts/analytic_benchmarks.py
  python python/scripts/analytic_benchmarks.py --export out/
退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # python/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.dataio.silvaco_log import SilvacoLogWriter   # noqa: E402
from pvsim.script import DeckCompiler                    # noqa: E402
from pvsim.solver.simulator import Simulator             # noqa: E402

DECK = ROOT.parent / "decks" / "device_example.tst"
Q_E = 1.602176634e-19
K_B = 8.617333262e-5   # eV/K
# numpy 2.0 把 trapz 改名为 trapezoid，这里兼容两个版本
_trapz = getattr(np, "trapezoid", None) or np.trapz

_failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failed += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="pvsim 解析基准")
    ap.add_argument("--export", default=None, metavar="DIR",
                    help="导出 1 sun J-V 的 SSF/.log 到该目录")
    args = ap.parse_args()

    base = DeckCompiler().compile_text(DECK.read_text(encoding="utf-8"))
    Vt = K_B * base.device.T_K          # eV 单位下的 kT/q

    def run(node_ref=None, intensity=None, v0=0.0, v1=1.3, npts=27):
        spec = base
        if node_ref is not None:
            spec = replace(spec, device=replace(spec.device, nodes=node_ref[0],
                                                interface_refine=node_ref[1]))
        if intensity is not None:
            spec = replace(spec, optics=replace(spec.optics,
                                                intensity_mW_cm2=intensity))
        spec = replace(spec, sweep=replace(spec.sweep, v_start=v0, v_stop=v1,
                                           n_points=npts))
        return Simulator().run(spec)

    print("1. 细致平衡上限")
    sun = run(intensity=100.0)
    x = sun.fields.x_nm
    g = sun.fields["G_OPT"][0]
    qG = Q_E * _trapz(g, x * 1e-7) * 1e3      # mA/cm²
    ratio = sun.jv.Jsc / qG
    check("Jsc ≤ q∫G dx", 0.0 < ratio <= 1.0 + 1e-9,
          f"Jsc={sun.jv.Jsc:.4f}, qG={qG:.4f}, 占比={ratio:.4f}")
    check("Jsc 达到上限的 80% 以上", ratio >= 0.8, f"{ratio:.4f}")

    print("2. 暗态理想因子")
    dark = run(intensity=0.0, v0=0.1, v1=0.7, npts=25)
    V, J = dark.V, dark.J
    m = J > 0.0
    slope = np.polyfit(V[m], np.log(J[m]), 1)[0]     # V⁻¹
    n_id = 1.0 / (Vt * slope)
    check("1.0 ≤ n_id ≤ 2.0", 1.0 <= n_id <= 2.0, f"n_id={n_id:.4f}")

    print("3. 理想因子的网格无关性")
    fine = run(node_ref=(201, 2.0), intensity=0.0, v0=0.1, v1=0.7, npts=25)
    Vf, Jf = fine.V, fine.J
    mf = Jf > 0.0
    n_fine = 1.0 / (Vt * np.polyfit(Vf[mf], np.log(Jf[mf]), 1)[0])
    check("|Δn_id| < 0.05", abs(n_id - n_fine) < 0.05,
          f"101/1: {n_id:.4f}, 201/2: {n_fine:.4f}")

    print("4. Jsc 与光强线性")
    half = run(intensity=50.0)
    ratio_i = sun.jv.Jsc / half.jv.Jsc
    check("Jsc(100)/Jsc(50) ≈ 2（±3%）", abs(ratio_i - 2.0) <= 0.06,
          f"比值={ratio_i:.4f}")

    print("5. 吸收层内电流守恒")
    bias = run(intensity=100.0, v0=1.0, v1=1.0, npts=1)
    thick = [L.thick_nm for L in base.device.layers]
    k_abs = int(np.argmax(thick))
    x0 = sum(thick[:k_abs]) + 10.0
    x1 = sum(thick[:k_abs + 1]) - 10.0
    jt = bias.fields["JTOTAL"][0]
    mask = (x > x0) & (x < x1)
    mean_j = float(jt[mask].mean())
    spread = float((jt[mask].max() - jt[mask].min()) / abs(mean_j))
    check("JTOTAL 展布 < 1e-3", spread < 1e-3,
          f"{x0:.0f}–{x1:.0f} nm, J={mean_j:.4f} mA/cm², 展布={spread:.2e}")

    print("6. 暗态零偏无净电流")
    zero = run(intensity=0.0, v0=0.0, v1=0.0, npts=1)
    j0 = zero.points[0].J_mA_cm2
    check("|J(0)| < 1e-8 mA/cm²", abs(j0) < 1e-8, f"J={j0:.3e}")

    print("\n参考曲线（供外部对照）")
    for inten in (10.0, 50.0, 100.0):
        r = run(intensity=inten).jv
        print(f"  {inten:5.0f} mW/cm²: Jsc={r.Jsc:7.4f}  Voc={r.Voc:.5f}  "
              f"FF={r.FF:.4f}  PCE={r.PCE:.3f}")

    if args.export:
        out = Path(args.export)
        path = SilvacoLogWriter().write(sun.jv, out / "jv_1sun.log",
                                        run_id="analytic_benchmarks",
                                        meta={"deck": DECK.name,
                                              "nodes": str(base.device.nodes),
                                              "refine": f"{base.device.interface_refine:g}"})
        print(f"\n已导出: {path}")

    print(f"\n{'全部通过' if _failed == 0 else f'{_failed} 项未通过'}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
