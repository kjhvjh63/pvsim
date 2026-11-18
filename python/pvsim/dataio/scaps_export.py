"""把 deck / ProblemSpec 导出成 SCAPS-1D 对照材料（脚本 + 参数清单）。

为什么是"脚本 + 清单"两份
-------------------------
SCAPS 的脚本语言（3.0.x）里，层栈本身由 GUI 或 ``.def`` 文件定义，脚本命令
主要负责动作（温度、光照、IV 扫描、存结果）。官方手册能确认的层参数命令只有
``set layerN.thickness/Eg/chi/epsilon`` 与 ``set layerN.defect.*``；Nc/Nv、
迁移率、掺杂浓度、缺陷密度这些命令名未在手册片段中验证。因此这里：

* ``.script``：只写**已验证**的命令，保证能跑；结构与材料参数需在 GUI 建好；
* ``.md``：完整参数清单，逐项给出"本平台值 → SCAPS 面板字段"，照填即可。

两者配合，IV 设置（温度/电压范围/点数）与本平台完全一致，产出的 ``.iv``
可直接交给 ``compare_reference.py``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..param.schema import ProblemSpec

SCAPS_MAX_LAYERS = 7


@dataclass(frozen=True)
class ScapsLayer:
    """一层的 SCAPS 视角参数（厚度用 µm，SCAPS 的单位）。"""

    index: int
    name: str
    material: str
    thickness_um: float
    Eg_eV: float
    chi_eV: float
    eps_r: float
    Nc_cm3: float
    Nv_cm3: float
    mun: float
    mup: float
    Nd_cm3: float
    Na_cm3: float
    taun_s: float
    taup_s: float
    is_contact: bool

    @property
    def is_semiconductor(self) -> bool:
        return not self.is_contact


def layers_of(spec: ProblemSpec) -> Tuple[ScapsLayer, ...]:
    out: List[ScapsLayer] = []
    for i, layer in enumerate(spec.device.layers):
        out.append(ScapsLayer(
            index=i,
            name=layer.name,
            material=layer.material or layer.name,
            thickness_um=layer.thick_nm / 1000.0,
            Eg_eV=layer.Eg_eV,
            chi_eV=layer.chi_eV,
            eps_r=layer.eps_r,
            Nc_cm3=layer.Nc_cm3,
            Nv_cm3=layer.Nv_cm3,
            mun=layer.mun_cm2Vs,
            mup=layer.mup_cm2Vs,
            Nd_cm3=layer.Nd_cm3,
            Na_cm3=layer.Na_cm3,
            taun_s=layer.effective_tau_n_s,
            taup_s=layer.effective_tau_p_s,
            is_contact=layer.is_contact,
        ))
    return tuple(out)


def _g(value: float) -> str:
    return f"{value:g}"


def scaps_script(spec: ProblemSpec, *, iv_name: str = "pvsim_ref") -> str:
    """生成 SCAPS ``.script``：只含官方手册可确认的命令。"""
    sweep = spec.sweep
    lines = [
        "// pvsim -> SCAPS-1D 对照脚本",
        "// 用法：GUI 里按同名 .md 清单建好层栈/掺杂/缺陷后，本脚本负责",
        "//      按与 pvsim 相同的温度、光照、电压范围跑 IV 并存 .iv。",
        f"// pvsim deck 的入射功率 = {_g(spec.optics.intensity_mW_cm2)} mW/cm²，",
        "// 请确认 SCAPS 的 Spectrum 面板为 AM1.5G、对应积分功率一致。",
        "",
        f"action workingpoint.temperature {_g(spec.device.T_K)}",
        "action light",
        "",
        f"action iv.startV {_g(sweep.v_start)}",
        f"action iv.stopV {_g(sweep.v_stop)}",
        f"action iv.points {int(sweep.n_points)}",
        "action iv.doiv",
        "",
        f"save results.iv {iv_name}",
        "",
    ]
    return "\n".join(lines)


def parameter_table(spec: ProblemSpec, *, iv_name: str = "pvsim_ref") -> str:
    """生成 Markdown 参数清单（逐层 → SCAPS 面板字段）。"""
    layers = layers_of(spec)
    semi = [l for l in layers if l.is_semiconductor]
    lines = [
        "# SCAPS-1D 对照参数清单（由 pvsim deck 生成）",
        "",
        f"- 层数：{len(layers)}（其中半导体 {len(semi)} 层，SCAPS 上限 "
        f"{SCAPS_MAX_LAYERS}）",
        f"- 温度：{_g(spec.device.T_K)} K",
        f"- 入射功率：{_g(spec.optics.intensity_mW_cm2)} mW/cm²",
        f"- IV 扫描：{_g(spec.sweep.v_start)} → {_g(spec.sweep.v_stop)} V，"
        f"{int(spec.sweep.n_points)} 点",
        f"- 电极：首个/末个 layer 视为接触层；"
        f"边界类型 {spec.sweep.bc_type}，"
        f"功函数 {_g(spec.sweep.work_fn_anode)} / "
        f"{_g(spec.sweep.work_fn_cathode)} eV",
        "",
        "## 逐层参数",
        "",
        "| # | 名称 | 材料 | 厚度 (µm) | Eg (eV) | χ (eV) | εr | Nc (cm⁻³) "
        "| Nv (cm⁻³) | μn (cm²/Vs) | μp (cm²/Vs) | Nd (cm⁻³) | Na (cm⁻³) "
        "| τn (s) | τp (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for l in layers:
        lines.append(
            f"| {l.index + 1} | {l.name} | {l.material} | {_g(l.thickness_um)} "
            f"| {_g(l.Eg_eV)} | {_g(l.chi_eV)} | {_g(l.eps_r)} "
            f"| {_g(l.Nc_cm3)} | {_g(l.Nv_cm3)} | {_g(l.mun)} | {_g(l.mup)} "
            f"| {_g(l.Nd_cm3)} | {_g(l.Na_cm3)} | {_g(l.taun_s)} | {_g(l.taup_s)} |"
        )
    lines += [
        "",
        "## 填入 SCAPS 的位置",
        "",
        "1. **Layer properties**：厚度、Eg、χ、εr、Nc、Nv、μn、μp 按上表；"
        "接触层（is_contact=1）填到最外侧 layer，并在 Contact 面板设功函数。",
        "2. **Doping**：Nd → 施主浓度（N_D），Na → 受主浓度（N_A），"
        "其余填 0；类型按浓度非零项选择 n/p。",
        "3. **Defects**：单能级 SRH 用"
        " `set layerN.defect.singlelevel`，带隙中深处填 Et = Eg/2；"
        "由 τ 反推 Nt·σ：`τ = 1 / (σ · v_th · Nt)`，v_th ≈ 1e7 cm/s。"
        "本平台 taun/taup 为等效寿命，可直接令 σn=σp=1e-15 cm⁻² 后反解 Nt。",
        "4. **Script 面板**：载入同名 `.script`（已按 pvsim 的 IV 设置写好）。",
        "",
        "## 验证过的 SCAPS 层参数命令（可选，手填更快）",
        "",
        "```text",
    ]
    for l in layers:
        if not l.is_semiconductor:
            continue
        lines += [
            f"set layer{l.index + 1}.thickness {_g(l.thickness_um):<8} // µm",
            f"set layer{l.index + 1}.Eg        {_g(l.Eg_eV):<8} // eV",
            f"set layer{l.index + 1}.chi       {_g(l.chi_eV):<8} // eV",
            f"set layer{l.index + 1}.epsilon   {_g(l.eps_r):<8}",
        ]
    lines += [
        "```",
        "",
        f"> 上面只列半导体层（{len(semi)} 层）；接触层（"
        + "、".join(l.name for l in layers if l.is_contact)
        + "）在 SCAPS 里按电极/高掺杂层处理，功函数见顶部说明。",
        "> 其余参数（Nc/Nv/迁移率/掺杂/缺陷密度）的命令名未在官方手册片段中"
        "确认，建议在 GUI 面板填写，避免脚本报错。",
        "",
    ]
    return "\n".join(lines)


def export_scaps(spec: ProblemSpec, out_dir, *,
                 iv_name: str = "pvsim_ref") -> Tuple[object, object]:
    """把 .script 与 .md 写到目录；返回两个路径。"""
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    script_path = d / f"{iv_name}.script"
    table_path = d / f"{iv_name}_scaps_params.md"
    script_path.write_text(scaps_script(spec, iv_name=iv_name),
                           encoding="utf-8", newline="\n")
    table_path.write_text(parameter_table(spec, iv_name=iv_name),
                          encoding="utf-8", newline="\n")
    return script_path, table_path
