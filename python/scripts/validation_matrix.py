"""4-case 验证矩阵：亮/暗 × 本征/掺杂 × 网格。

默认 refine=1（界面加密，接触层可解析，网格无关）；--refine 0 只作
粗网格快速预览，接触积累层无法解析，结果会随节点数漂移。
数据源是仓库里的 decks/device_example.tst，保证与文档示例一致。

用法:
  python python/scripts/validation_matrix.py
  python python/scripts/validation_matrix.py --nodes 101,201,397
  python python/scripts/validation_matrix.py --refine 2 --nodes 201

退出码 0 = 全部点收敛；1 = 有未收敛点。
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # python/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvsim.script import DeckCompiler            # noqa: E402
from pvsim.solver.simulator import Simulator     # noqa: E402

DECK = ROOT.parent / "decks" / "device_example.tst"


def doped_layers(layers):
    """给 TiO2 掺 n、Spiro 掺 p，各 1e16（与原矩阵一致）。"""
    return tuple(
        replace(L, Nd_cm3=1e16) if L.name == "TiO2" else
        replace(L, Na_cm3=1e16) if L.name == "Spiro" else L
        for L in layers
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="pvsim 4-case 验证矩阵")
    ap.add_argument("--nodes", default="101,201",
                    help="逗号分隔的目标节点数（默认 101,201）")
    ap.add_argument("--refine", type=float, default=1.0,
                    help="interface_refine（默认 1.0；0=粗网格预览）")
    args = ap.parse_args()

    nodes_list = [int(s) for s in args.nodes.split(",") if s.strip()]
    base = DeckCompiler().compile_text(DECK.read_text(encoding="utf-8"))

    cases = (("亮本征", base.device.layers, 100.0),
             ("暗本征", base.device.layers, 0.0),
             ("亮掺杂", doped_layers(base.device.layers), 100.0),
             ("暗掺杂", doped_layers(base.device.layers), 0.0))

    all_ok = True
    for label, layers, intensity in cases:
        for nodes in nodes_list:
            spec = replace(
                base,
                device=replace(base.device, layers=tuple(layers),
                               nodes=nodes, interface_refine=args.refine),
                optics=replace(base.optics, intensity_mW_cm2=intensity),
            )
            t0 = time.time()
            res = Simulator().run(spec)
            jv = res.jv
            conv = sum(p.converged for p in res.points)
            ok = conv == len(res.points)
            all_ok = all_ok and ok
            print(f"{label} nodes={nodes} refine={args.refine:g} "
                  f"n={res.fields.n_node} 收敛{conv}/{len(res.points)} "
                  f"Jsc={jv.Jsc:.4f} Voc={jv.Voc:.4f} "
                  f"FF={jv.FF:.4f} PCE={jv.PCE:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"\n{'全部收敛' if all_ok else '存在未收敛点'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
