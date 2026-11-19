"""从 Atlas 的解文件里直接取光学量，与本平台逐波长对照。

Atlas 的 ``save`` 解文件（SSF 2.0）包含两张光学表：

* ``R`` 行（``O`` 行之后）：逐波长光线表，5 条/波长，依次是
  真空区与 FTO/TiO2/PVK/NiO 四个区。第二列是该区**入射功率**，
  因此
      A_all(λ) = 1 - P(NiO)/P(真空)
      A_pvk(λ) = (P(PVK) - P(NiO)) / P(真空)
* ``Q 209 "Photon absorption rate"`` + ``n`` 行：逐节点产生率 G(x)。
  注意保存网格较粗（PVK 内约 20 nm），前表面陡降会被梯形积分低估
  约 10%，G(x) 只用于看形状，绝对量以 R 表为准。

Atlas 的 BEAM 是**非相干光线追踪**：A(λ) 光滑无干涉条纹，且光线表
不扣前表面反射（300–600 nm 的 A_all 正好是 1.0）。

用法::

    python scripts/atlas_optics.py                # 只打印 Atlas 光学量
    python scripts/atlas_optics.py --compare      # 再与本平台相干/非相干对照
"""
from __future__ import annotations

import argparse
import ctypes as ct
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
Q = 1.602176634e-19
QID_GEN = 12          # s 行 qids 里 "209 Photon absorption rate" 的位置


def run_atlas(deck: Path, atlas: str, nk_dir: Path) -> Path:
    """跑一遍 Atlas 并返回保存的 state.str 路径（临时目录，调用方负责清理）。"""
    work = Path(tempfile.mkdtemp(prefix="atlas_optics_"))
    for f in nk_dir.iterdir():
        if f.suffix.lower() in (".txt", ".spec"):
            shutil.copy2(f, work / f.name)
    text = deck.read_text(encoding="utf-8")
    if "save outf=state.str" not in text:
        text = text.replace("quit", "save outf=state.str\nquit")
    text = text.replace("outf=atlas2d_800nm.log", "outf=state.log")
    (work / "optics.in").write_text(text, encoding="utf-8")
    proc = subprocess.run([atlas, "optics.in"], cwd=work, capture_output=True,
                          text=True, errors="ignore")
    state = work / "state.str"
    if not state.exists():
        raise RuntimeError("Atlas 未生成 state.str：\n"
                           + ((proc.stdout or "") + (proc.stderr or ""))[-1500:])
    return state


def parse_radecay(state: Path):
    """R 表 → (λ_nm, A_all, A_pvk)。"""
    lines = state.read_text(encoding="utf-8", errors="ignore").splitlines()
    oi = next(i for i, l in enumerate(lines) if l.startswith("O "))
    by = {}
    for l in lines[oi + 1:]:
        if not l.startswith("R "):
            continue
        t = l.split()
        by.setdefault(float(t[4]), {})[int(t[3])] = float(t[5])
    lam, a_all, a_pvk = [], [], []
    for lu, d in sorted(by.items()):
        p0 = d.get(0, 0.0)
        if p0 <= 0:
            continue
        lam.append(lu * 1e3)
        a_all.append(1.0 - d.get(4, 0.0) / p0)
        a_pvk.append((d.get(3, 0.0) - d.get(4, 0.0)) / p0)
    return np.array(lam), np.array(a_all), np.array(a_pvk)


def parse_gx(state: Path):
    """节点表 col=209 → (y_nm, G)。只取左边界一列（解在 x 方向均匀）。"""
    lines = state.read_text(encoding="utf-8", errors="ignore").splitlines()
    nodes, vals = {}, {}
    for l in lines:
        if l.startswith("c "):
            t = l.split()
            nodes[int(t[1])] = (float(t[2]), float(t[3]))
        elif l.startswith("n "):
            t = l.split()
            vals[int(t[1])] = float(t[2 + QID_GEN])
    pts = sorted((nodes[i][1] * 1e3, vals[i]) for i in nodes
                 if abs(nodes[i][0]) < 1e-9)
    y = np.array([a for a, _ in pts])
    g = np.array([b for _, b in pts])
    return y, g


def platform_curves(deck: Path):
    """复刻卡的 A(λ)（coherent / 非相干）与 G(x)。"""
    from dataclasses import replace
    from pvsim.script import DeckCompiler
    from pvsim.kernel.abi import PvkLib
    from pvsim.kernel import structs as st

    base = DeckCompiler().compile_text(deck.read_text(encoding="utf-8"))
    lib = PvkLib()
    dev, keep = st.device_from_spec(base.device)
    ctx = lib.create()
    lib.build_mesh(ctx, dev)
    n = lib.node_count(ctx)
    x = np.asarray(lib.get_x(ctx, n)) * 1e7

    out = {}
    for coherent in (True, False):
        spec = replace(base.optics, coherent=coherent)
        opt = st.optics_from_spec(spec)
        a = (ct.c_double * opt.n_wl)()
        g = (ct.c_double * n)()
        rc = lib._lib.pvk_optics_compute(ctx, ct.byref(dev), ct.byref(opt), a, g)
        lib._check(ctx, rc, "optics_compute")
        out["coh" if coherent else "inc"] = (
            spec.wl_nm.copy(), np.asarray(list(a)), x, np.asarray(list(g)))
    lib.destroy(ctx)
    return base, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas 光学量导出与对照")
    ap.add_argument("--deck", default=str(DECK_DEFAULT))
    ap.add_argument("--atlas", default=ATLAS_DEFAULT)
    ap.add_argument("--nk-dir", default=str(NK_DEFAULT))
    ap.add_argument("--compare", action="store_true", help="与本平台复刻卡对照")
    args = ap.parse_args()

    state = run_atlas(Path(args.deck), args.atlas, Path(args.nk_dir))
    try:
        lam, a_all, a_pvk = parse_radecay(state)
        y, g = parse_gx(state)

        print("Atlas 光线表（非相干、不扣前表面反射）")
        print(f"{'λ(nm)':>7} {'A_all':>7} {'A_pvk':>7}")
        for t in (400, 500, 600, 700, 800, 850, 900, 950, 1000, 1050, 1100):
            i = int(np.argmin(abs(lam - t)))
            print(f"{lam[i]:7.0f} {a_all[i]:7.4f} {a_pvk[i]:7.4f}")

        from pvsim.param.schema import OpticsSpec
        spec = OpticsSpec.default_am15g()
        wl, phi = spec.wl_nm, spec.flux * np.gradient(spec.wl_nm)
        aa = np.interp(wl, lam, a_all, left=a_all[0], right=0.0)
        ap_ = np.interp(wl, lam, a_pvk, left=a_pvk[0], right=0.0)
        print(f"\nq∫A_all·φ = {Q*np.sum(aa*phi)*1e3:.3f} mA/cm²（300–1100 nm）")
        print(f"q∫A_pvk·φ = {Q*np.sum(ap_*phi)*1e3:.3f} mA/cm²")
        print(f"节点 G(x)：y {y[0]:.0f}–{y[-1]:.0f} nm，"
              f"q∫G = {Q*np.trapz(g, y*1e-7)*1e3:.2f} mA/cm²"
              "（粗网格低估 ~10%，只作形状参考）")

        if args.compare:
            _, curves = platform_curves(PLATFORM_DECK)
            print(f"\n{'λ(nm)':>7} {'A_all':>7} {'coh':>7} {'inc':>7}")
            for t in (400, 500, 600, 700, 800, 900, 1000, 1050):
                i = int(np.argmin(abs(lam - t)))
                wlc, ac, _, _ = curves["coh"]
                _, ai, _, _ = curves["inc"]
                jc = int(np.argmin(abs(wlc - t)))
                print(f"{lam[i]:7.0f} {a_all[i]:7.4f} {ac[jc]:7.4f} {ai[jc]:7.4f}")
            root = Path(tempfile.gettempdir())
            np.savez(root / "atlas_optics.npz", lam=lam, a_all=a_all,
                     a_pvk=a_pvk, y=y, g=g)
            print(f"\n曲线已存 {root / 'atlas_optics.npz'}")
        return 0
    finally:
        shutil.rmtree(state.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
