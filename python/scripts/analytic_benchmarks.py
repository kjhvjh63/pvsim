"""解析基准与外部对照。

不依赖任何外部软件，用几个可解析/普适的物理关系验证求解结果：
  1. 细致平衡上限：Jsc ≤ q∫G dx（吸收的光子数上限）
  2. 暗态理想因子：指数区斜率给出的 n_id 应落在 1–2
  3. 理想因子的网格无关性：101/refine=1 与 201/refine=2 之差 < 0.05
  4. Jsc 与光强近似线性（50/100 mW/cm² 之比与 2 的偏差 < 3%）
  5. 吸收层内总电流守恒：JTOTAL 沿 x 的展布 < 1e-3
  6. 暗态零偏无净电流：|J(0)| < 1e-8 mA/cm²
  7. 亮态 Voc 光照斜率锚点：0.3→1 sun 的 dVoc/dlnG/Vt 应落在 2.5–3.6。
     注意：单能级 SRH 高注入的理论斜率是 2，实测 >2 是 A′ 接触模型
     （整个 ETL/HTL 归入钉扎层、无限表面复合）的固有后果——Voc 处
     环流电流（=Jsc）把吸收层边缘撕出随光强增大的 n/p 失衡，
     SRH=np/(τ(n+p)) 中 (n+p) 增速超过 e^(V/2Vt)，压低了 SRH 随 V
     的增速。已排除网格伪影（201/2 同值）与串联电阻（Voc 处 J=0）。
     此锚点只防行为回退，不代表物理正确；待外部软件对照后再定取舍。
  8. 厚吸收层粗网格锚点（1500nm）：曾因 40nm TiO2 在 101 节点网格下
     只剩 1 个节点、被电极 bc 吞并，导致 cpl 钉扎侵入吸收层，
     Jsc 塌到暗电流量级却被报成收敛。现要求 @101 与 @201 的 Jsc
     相差 <20%，且 1500nm 的 Jsc 不低于 500nm 的 70%。
  9. Atlas 2D 对齐锚点：Cs2AgBiBr6 800nm 复刻卡与 Silvaco Atlas 的
     Jsc/Voc/FF 偏差（5% / 100 mV / 0.10）。辐射复合（copt=2.3e-9）、
     相干薄膜光学（TMM，optics 的 coherent 开关）、背电极反射与 nk 超范围
     端点钳制均已在内核实现；q∫G=31.39（含 100 nm Au 的复折射率），
     实测 Jsc 26.98（Atlas 27.27，-1.09%）。
 10. 低光锚点：同一复刻卡 1 sun → 0.01 mW 的 Jsc 严格线性（±2%）、
     Voc 有限且单调、Voc 以下（V≤1.2 V）各档全部收敛。锁住连续方程
     尺度化重构与精确 Bernoulli（expm1）求值的成果。复刻卡扫到 2.0 V，
     强注入区（>1.36 V）Newton 仍会停滞，属已知数值限制，不在此锚点内。
 11. 光学模式自检：atlas.ray 与独立 Python 单程 Beer-Lambert 复算一致
     （A(800)/A(1000) ±2%），coherent 的 q∫G 与 A(800) 在合理区间。
     锁住新增的相干/对照光学路径，防止单位或索引回归。

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

    print("7. 亮态 Voc 光照斜率锚点（A′ 接触固有 n>2，见模块文档）")
    third = run(intensity=30.0)
    n_light = (sun.jv.Voc - third.jv.Voc) / np.log(100.0 / 30.0) / Vt
    check("2.5 ≤ n_light ≤ 3.6", 2.5 <= n_light <= 3.6, f"n_light={n_light:.3f}")

    print("8. 厚吸收层粗网格锚点（1500nm，防 cpl 钉扎侵入吸收层）")
    def run_thick(nodes):
        ls = list(base.device.layers)
        ls[k_abs] = replace(ls[k_abs], thick_nm=1500.0)
        spec = replace(base, device=replace(base.device, layers=tuple(ls),
                                            nodes=nodes, interface_refine=1.0))
        spec = replace(spec, sweep=replace(spec.sweep, v_start=0.0, v_stop=0.0,
                                           n_points=1))
        return Simulator().run(spec).points[0].J_mA_cm2
    j15_fine, j15_coarse = run_thick(201), run_thick(101)
    check("1500nm@201 |Jsc| ≥ 0.7×500nm", abs(j15_fine) >= 0.7 * abs(sun.jv.Jsc),
          f"Jsc(1500)={j15_fine:.4f}, Jsc(500)={sun.jv.Jsc:.4f}")
    check("1500nm@101 与 @201 偏差 <20%",
          abs(abs(j15_coarse) - abs(j15_fine)) / abs(j15_fine) < 0.2,
          f"Jsc(101)={j15_coarse:.4f}, Jsc(201)={j15_fine:.4f}")

    print("9. Atlas 2D 对齐锚点（Cs2AgBiBr6 800nm 复刻卡）")
    # 参考值来自本机 Silvaco Atlas（decks/atlas2d_800nm.in，1 sun）：
    # Jsc 27.27 / Voc 1.1822 / FF 0.810。复跑方式见 scripts/atlas_reference.py。
    # 内核已实现辐射复合（copt）与相干薄膜光学（复折射率 TMM、完美背镜，
    # 缺 nk 的尾部接触层不传播），nk 超范围端点钳制用于消除 898 nm 假台阶。
    # 当前 q∫G=31.39（相干 + 100 nm Au），Jsc 26.98（Atlas 27.27，-1.09%）、
    # Voc 1.1822、FF 0.8092。缺 Au nk 时回退完美背镜：q∫G 31.56、Jsc 27.11。
    ref_deck = (Path(__file__).resolve().parents[2] / "decks" /
                "cs2agbibr6_800nm_ref.tst")
    ref_spec = DeckCompiler().compile_text(ref_deck.read_text(encoding="utf-8"))
    res9 = Simulator().run(ref_spec)
    r9 = res9.jv
    AT_JSC, AT_VOC, AT_FF = 27.27, 1.1822, 0.810
    check("Jsc 与 Atlas 差 <5%", abs(r9.Jsc - AT_JSC) / AT_JSC < 0.05,
          f"ours={r9.Jsc:.2f}, atlas={AT_JSC:.2f}")
    check("Voc 与 Atlas 差 <100 mV",
          np.isfinite(r9.Voc) and abs(r9.Voc - AT_VOC) < 0.10,
          f"ours={r9.Voc:.4f}, atlas={AT_VOC:.4f}")
    check("FF 与 Atlas 差 <0.10",
          np.isfinite(r9.FF) and abs(r9.FF - AT_FF) < 0.10,
          f"ours={r9.FF:.4f}, atlas={AT_FF:.4f}")

    print("10. 低光锚点（800nm 复刻卡，1 sun → 1 mW）")
    # 锁住低光成果：Jsc 必须与光强严格线性（误差来自 SRH/双分子复合的
    # 物理偏离，应 <1%），Voc 必须有限且随光强单调。收敛性只看 Voc 以下
    # （V≤1.2 V）：复刻卡扫到 2.0 V，>1.36 V 的强注入区 Newton 会停滞，
    # 那是已知数值限制；修好 Bernoulli 后 0–1.2 V 各档全部收敛。
    def run_ref(inten: float):
        spec = replace(ref_spec,
                       optics=replace(ref_spec.optics, intensity_mW_cm2=inten))
        return Simulator().run(spec)

    r10, r1 = run_ref(10.0), run_ref(1.0)
    r01, r001 = run_ref(0.1), run_ref(0.01)
    for name, res_k, ratio in (("10", r10, 0.1), ("1", r1, 0.01),
                               ("0.1", r01, 1e-3), ("0.01", r001, 1e-4)):
        lin = res_k.jv.Jsc / r9.Jsc
        check(f"Jsc({name})/Jsc(100) = {ratio:g}（±2%）",
              abs(lin - ratio) <= 0.02 * ratio, f"比值={lin:.6g}")
    vocs = (r9.Voc, r10.jv.Voc, r1.jv.Voc, r01.jv.Voc, r001.jv.Voc)
    check("Voc 有限且随光强单调",
          all(np.isfinite(v) for v in vocs)
          and all(vocs[i] > vocs[i + 1] for i in range(4)),
          " > ".join(f"{v:.4f}" for v in vocs))
    v_cut = 1.2
    n_min = min(sum(p.converged for p in r.points if p.V <= v_cut)
                for r in (res9, r10, r1, r01, r001))
    n_tot = sum(1 for p in res9.points if p.V <= v_cut)
    check(f"V≤{v_cut:g} V 各档全部收敛", n_min == n_tot,
          f"最少 {n_min}/{n_tot}")

    print("11. 光学模式自检（atlas.ray / coherent）")
    # atlas.ray 是「单程、无界面反射」的对照路径；用独立 Python 复算同一套
    # nk 的 Beer-Lambert 吸收，二者必须在 2% 内一致（防止单位/索引回归）。
    import ctypes as ct
    from pvsim.kernel.abi import PvkLib
    from pvsim.kernel import structs as st

    nk_dir = Path(ref_spec.optics.nk_dir or "")
    if not nk_dir.exists():
        print("  [SKIP] nk.dir 不存在，跳过光学自检")
    else:
        def _tab(name: str):
            for pat in ("{}.txt", "{}_nk_wavelength.txt", "{}_nk.txt"):
                p = nk_dir / pat.format(name)
                if not p.exists():
                    continue
                rows = []
                for ln in p.read_text(errors="ignore").splitlines():
                    t = ln.split()
                    if len(t) < 2:
                        continue
                    try:
                        a, b = float(t[0]), float(t[1])
                        c = float(t[2]) if len(t) > 2 else 0.0
                    except ValueError:
                        continue
                    if a > 0:
                        rows.append((a, b, c))
                if len(rows) >= 2:
                    return rows
            return None

        def a_py(lam_nm: float) -> float:
            tau = 0.0
            for i, L in enumerate(ref_spec.device.layers):
                tab = _tab(L.material or L.name)
                if tab is None or (L.is_contact and i > 0):
                    continue
                lu = min(max(lam_nm * 1e-3, tab[0][0]), tab[-1][0])
                k = float(np.interp(lu, [r[0] for r in tab], [r[2] for r in tab]))
                tau += 4.0 * np.pi * k / (lam_nm * 1e-7) * L.thick_nm * 1e-7
            return 1.0 - np.exp(-tau)

        lib = PvkLib()
        dev, _keep = st.device_from_spec(ref_spec.device)
        ctx = lib.create()
        lib.build_mesh(ctx, dev)
        n_opt_nodes = lib.node_count(ctx)

        def a_kernel(lam_nm: float, **kw) -> float:
            sp = replace(ref_spec.optics, wl_nm=np.array([lam_nm]),
                         flux=np.array([1.0]), **kw)
            opt = st.optics_from_spec(sp)
            out = (ct.c_double * 1)()
            rc = lib._lib.pvk_optics_compute(ctx, ct.byref(dev), ct.byref(opt),
                                             out, None)
            lib._check(ctx, rc, "optics_compute")
            return float(out[0])

        for lam in (800.0, 1000.0):
            got = a_kernel(lam, coherent=False, atlas_ray=True)
            want = a_py(lam)
            check(f"atlas.ray A({lam:.0f}) 与独立单程复算一致（±2%）",
                  abs(got - want) <= 0.02 * max(want, 1e-6),
                  f"kernel={got:.4f}, python={want:.4f}")

        sp = replace(ref_spec.optics, coherent=True, atlas_ray=False)
        opt = st.optics_from_spec(sp)
        g = (ct.c_double * n_opt_nodes)()
        rc = lib._lib.pvk_optics_compute(ctx, ct.byref(dev), ct.byref(opt),
                                         None, g)
        lib._check(ctx, rc, "optics_compute")
        x = np.asarray(lib.get_x(ctx, n_opt_nodes))
        gg = np.asarray(list(g))
        qg = float(Q_E * np.sum(0.5 * (gg[1:] + gg[:-1]) * np.diff(x)) * 1e3)
        check("coherent q∫G ∈ [30, 33] mA/cm²", 30.0 <= qg <= 33.0, f"{qg:.2f}")
        a800 = a_kernel(800.0, coherent=True, atlas_ray=False)
        check("coherent A(800) ∈ [0.8, 1.0]", 0.8 <= a800 <= 1.0, f"{a800:.3f}")
        lib.destroy(ctx)

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
