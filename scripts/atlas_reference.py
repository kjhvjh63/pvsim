"""用本机 Silvaco Atlas 复现参考 J-V，并与本平台对照。

流程：

1. 把 ``decks/atlas2d_800nm.in`` 连同折射率文件、``AM15.spec`` 复制到临时目录；
2. 调用 ``atlas.exe`` 跑 2D、1 sun 的电压扫描；
3. 解析 Atlas 的 SSF 日志（``p`` 参数表 + ``d`` 数据行，第 15 列电压、第 17 列电流）；
4. 打印 Jsc/Voc/FF/PCE；``--compare`` 时再跑一遍 ``decks/cs2agbibr6_800nm_ref.tst``。

用法::

    python scripts/atlas_reference.py
    python scripts/atlas_reference.py --compare
    python scripts/atlas_reference.py --atlas D:/sedatools/exe/atlas.exe \\
                                     --nk-dir D:/交互界面_data/datasets/nk

单位换算：2D 的电流是 A/µm（每 µm 深度），接触宽 10 µm、深度 1 µm，
面积取 ``--area-cm2``（默认 1e-7）。beam 的 ``min.w/max.w`` 相对 ``x.origin``，
写错会让光束只覆盖半个器件、电流正好减半（详见 deck 头注释）。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pvsim.solver.simulator import JVData          # noqa: E402

ATLAS_DEFAULT = r"D:\sedatools\exe\atlas.exe"
NK_DEFAULT = r"D:\交互界面_data\datasets\nk"
DATA_SUFFIXES = (".txt", ".spec")


def _parse_ssf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 Silvaco SSF 日志的 ``d`` 数据行，返回 (电压, 电流 A/µm)。"""
    rows = [[float(x) for x in ln.split()[1:]]
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.startswith("d ")]
    if not rows:
        raise RuntimeError(f"{path.name} 里没有 d 数据行，Atlas 可能没跑起来")
    a = np.array(rows)
    if a.shape[1] < 17:
        raise RuntimeError(f"{path.name} 列数 {a.shape[1]}，少于预期的 17 列")
    return a[:, 14], a[:, 16]


def run_atlas(deck: Path, atlas: str, nk_dir: Path, area_cm2: float) -> JVData:
    if not Path(atlas).exists():
        raise FileNotFoundError(f"找不到 atlas.exe：{atlas}")
    # 不用 TemporaryDirectory：Atlas 退出后仍会短暂占用日志文件，自动清理会抛异常
    work = Path(tempfile.mkdtemp(prefix="atlas_ref_"))
    try:
        for f in nk_dir.iterdir():
            if f.suffix.lower() in DATA_SUFFIXES:
                shutil.copy2(f, work / f.name)
        shutil.copy2(deck, work / deck.name)
        proc = subprocess.run([atlas, deck.name], cwd=work,
                              capture_output=True, text=True, errors="ignore")
        out = work / "atlas2d_800nm.log"
        if not out.exists():
            raise RuntimeError("Atlas 未生成日志：\n" + (proc.stdout or "")[-2000:])
        V, I = _parse_ssf(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    order = np.argsort(V)
    return JVData.from_points(V[order], I[order] / area_cm2 * 1000.0, 100.0)


def run_pvsim(deck: Path) -> JVData:
    from pvsim.script import DeckCompiler
    from pvsim.solver import Simulator
    text = deck.read_text(encoding="utf-8")
    return Simulator().run(DeckCompiler().compile_text(text)).jv


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas 2D 参考复现")
    ap.add_argument("--deck", default=str(ROOT / "decks" / "atlas2d_800nm.in"))
    ap.add_argument("--atlas", default=ATLAS_DEFAULT)
    ap.add_argument("--nk-dir", default=NK_DEFAULT)
    ap.add_argument("--area-cm2", type=float, default=1e-7)
    ap.add_argument("--compare", action="store_true", help="同时跑本平台 deck 并对照")
    args = ap.parse_args()

    jv = run_atlas(Path(args.deck), args.atlas, Path(args.nk_dir), args.area_cm2)
    print(f"Atlas 2D（1 sun）: Jsc={jv.Jsc:7.2f}  Voc={jv.Voc:.4f}  "
          f"FF={jv.FF:.3f}  PCE={jv.PCE:6.2f}%")

    if args.compare:
        ours = run_pvsim(ROOT / "decks" / "cs2agbibr6_800nm_ref.tst")
        for name, a, b in (("Jsc", jv.Jsc, ours.Jsc), ("Voc", jv.Voc, ours.Voc),
                           ("FF", jv.FF, ours.FF), ("PCE", jv.PCE, ours.PCE)):
            print(f"  {name:<4} Atlas={a:9.4f}  本平台={b:9.4f}  相对={(b - a) / a * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
