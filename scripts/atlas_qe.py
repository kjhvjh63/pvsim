"""低光单色 QE 对照：Atlas 与本平台逐波长比较。

Atlas 侧：把 deck 的 AM1.5 光源换成单色 ``beam wavelength=<µm>``，强度
``solve b1=<W/cm²>``（默认 1e-4，低光下 QE 与强度无关），读 V=0 阴极电流，
再按 1 sun 电流归一得到 Jsc(λ)，QE = Jsc/(qΦ)。

本平台侧：用 2 点单色谱（λ±0.5 nm）在 V=0 求解，QE = Jsc/(qΦ)。
低光下收集率≈100%，因此 QE 就等于钙钛矿层的吸收率。

用法::

    python scripts/atlas_qe.py                # 只算 Atlas 的 QE
    python scripts/atlas_qe.py --compare      # 再与平台相干 / atlas.ray 对照
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
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
DECK_DEFAULT = ROOT / "decks" / "atlas2d_800nm.in"
PLATFORM_DECK = ROOT / "decks" / "cs2agbibr6_800nm_ref.tst"
NK_DEFAULT = Path(r"D:/交互界面_data/datasets/nk")
ATLAS_DEFAULT = r"D:\sedatools\exe\atlas.exe"
Q_E = 1.602176634e-19
HC = 6.62607015e-34 * 299792458.0
WL_DEFAULT = "450,500,550,600,650,700,750,800,850,900,950,1000"


def run_atlas(text: str, atlas: str, nk_dir: Path, tag: str) -> float:
    """跑一遍 deck，返回 V=0 时的阴极电流（qid 21）。"""
    work = Path(tempfile.mkdtemp(prefix=f"atlas_qe_{tag}_"))
    for f in nk_dir.iterdir():
        if f.suffix.lower() in (".txt", ".spec"):
            shutil.copy2(f, work / f.name)
    (work / "qe.in").write_text(text, encoding="utf-8")
    proc = subprocess.run([atlas, "qe.in"], cwd=work, capture_output=True,
                          text=True, errors="ignore")
    state = work / "state.log"
    if not state.exists():
        raise RuntimeError("Atlas 未生成解文件：\n"
                           + ((proc.stdout or "") + (proc.stderr or ""))[-1500:])
    idx = None
    for line in state.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("p "):
            t = line.split()
            ids = t[2:2 + int(t[1])]
            idx = ids.index("21")
        elif line.startswith("d ") and idx is not None:
            return float(line.split()[1:][idx])
    raise RuntimeError("解文件里找不到阴极电流（qid 21）")


def qe_atlas(deck: Path, atlas: str, nk_dir: Path, b1: float,
             wavelengths) -> dict[float, float]:
    """逐波长算 Atlas QE：单色电流 / 1 sun 电流 × 27.27 mA/cm² 再除 qΦ。"""
    base = deck.read_text(encoding="utf-8")
    base = base.replace("log outf=atlas2d_800nm.log", "log outf=state.log")
    base = base.replace("solve vcathode=0 vstep=0.02 vfinal=2 name=cathode",
                        "solve vcathode=0 vstep=0.02 vfinal=0.02 name=cathode")
    i_sun = abs(run_atlas(base, atlas, nk_dir, "sun"))
    out = {}
    for lam in wavelengths:
        mono = base.replace(
            "beam num=1 x.origin=5 y.origin=-0.5 angle=90 "
            "power.file=AM15.spec min.w=-5 max.w=5",
            "beam num=1 x.origin=5 y.origin=-0.5 angle=90 "
            f"wavelength={lam / 1000.0:.4f} min.w=-5 max.w=5")
        mono = mono.replace("solve b1=1\n", f"solve b1={b1}\n")
        j = 27.27 * abs(run_atlas(mono, atlas, nk_dir, str(lam)) / i_sun)
        phi = b1 / (HC / (lam * 1e-9))
        out[lam] = j * 1e-3 / (Q_E * phi)
    return out


def qe_platform(deck: Path, wavelengths, p_w: float) -> dict[str, dict[float, float]]:
    """平台侧 QE（相干 / atlas.ray）。用 2 点单色谱把总通量定为 2·Φ。"""
    from dataclasses import replace
    from pvsim.script import DeckCompiler
    from pvsim.solver import Simulator

    text = deck.read_text(encoding="utf-8").replace(
        "solve vstart=0 vfinal=2.0 vstep=0.02",
        "solve vstart=0 vfinal=0.02 vstep=0.02")
    base = DeckCompiler().compile_text(text)
    out = {"coh": {}, "ray": {}}
    for lam in wavelengths:
        phi = p_w / (HC / (lam * 1e-9))
        for tag, kw in (("coh", dict(coherent=True, atlas_ray=False)),
                        ("ray", dict(coherent=False, atlas_ray=True))):
            opt = replace(base.optics, wl_nm=np.array([lam - 0.5, lam + 0.5]),
                          flux=np.array([phi, phi]), intensity_mW_cm2=100.0, **kw)
            jv = Simulator().run(replace(base, optics=opt)).jv
            out[tag][lam] = jv.Jsc * 1e-3 / (Q_E * 2.0 * phi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="低光 QE 对照")
    ap.add_argument("--deck", default=str(DECK_DEFAULT))
    ap.add_argument("--platform-deck", default=str(PLATFORM_DECK))
    ap.add_argument("--atlas", default=ATLAS_DEFAULT)
    ap.add_argument("--nk-dir", default=str(NK_DEFAULT))
    ap.add_argument("--b1", type=float, default=1e-4, help="Atlas 单色强度 W/cm²")
    ap.add_argument("--power", type=float, default=1e-4,
                    help="平台单色强度 W/cm²")
    ap.add_argument("--wavelengths", default=WL_DEFAULT)
    ap.add_argument("--compare", action="store_true", help="与平台对照")
    args = ap.parse_args()

    wls = [float(x) for x in args.wavelengths.split(",") if x.strip()]
    a = qe_atlas(Path(args.deck), args.atlas, Path(args.nk_dir),
                 args.b1, wls)
    if not args.compare:
        print(f"{'λ(nm)':>7} {'QE_Atlas':>9}")
        for lam in wls:
            print(f"{lam:7.0f} {a[lam]:9.4f}")
        return 0

    p = qe_platform(Path(args.platform_deck), wls, args.power)
    print(f"{'λ(nm)':>7} {'Atlas':>7} {'coh':>7} {'ray':>7} "
          f"{'ray-Atlas':>10} {'coh-Atlas':>10}")
    dc, dr = [], []
    for lam in wls:
        dc.append(p["coh"][lam] - a[lam])
        dr.append(p["ray"][lam] - a[lam])
        print(f"{lam:7.0f} {a[lam]:7.4f} {p['coh'][lam]:7.4f} "
              f"{p['ray'][lam]:7.4f} {dr[-1]:+10.4f} {dc[-1]:+10.4f}")
    print(f"\nRMS：atlas.ray {np.sqrt(np.mean(np.square(dr))):.4f}，"
          f"相干 {np.sqrt(np.mean(np.square(dc))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
