"""DeckCompiler：把 :class:`Deck` 编译成唯一的 ``ProblemSpec``。

职责
----
* 把 ``region`` 序列 + 内建材料表（或外部 material_db）组装成层栈；
* 把 ``doping`` / ``material`` 覆盖参数落到对应层；
* 把 ``optics`` / ``method`` / ``solve`` / ``contact`` / ``output`` 映射到
  对应的 Spec 子结构；
* 语义错误（材料不存在、region 编号重复、缺 thickness）抛 ``SpecError``；
  非致命问题（未知模型、log 落盘等）走 ``log`` 回调，不影响编译。

内建材料表是过渡方案：正式材料库落地后由 ``csrc/material_db/*.mat`` 提供，
``DeckCompiler(material_db=...)`` 的优先级高于内建表。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..param.schema import (
    VALID_FIELDS,
    DeviceSpec,
    LayerSpec,
    MaterialSet,
    OpticsSpec,
    OutputSpec,
    ProblemSpec,
    SolverOpts,
    SpecError,
    SweepSpec,
)
from . import ast_nodes as ast

# 电极材料（用于判断首/末层是否挂边界条件）
CONTACT_MATERIALS = frozenset({"FTO", "ITO", "AZO", "AU", "AG", "AL", "CU", "NI"})

# deck 场量别名 → VALID_FIELDS
FIELD_ALIASES: Dict[str, str] = {
    "E.FIELD": "EFIELD",
    "ELECTRIC.FIELD": "EFIELD",
    "CON.BAND": "CB",
    "EC": "CB",
    "VAL.BAND": "VB",
    "EV": "VB",
    "J": "JTOTAL",
    "J.TOTAL": "JTOTAL",
    "R": "R_NET",
    "G": "G_OPT",
    "PHI.N": "PHIN",
    "PHI.P": "PHIP",
}

# 内建材料参数（过渡用；数值与 schema.mapbi3_stack 一致）
BUILTIN_MATERIALS: Dict[str, Dict[str, float]] = {
    "FTO": dict(eps_r=9.0, Eg_eV=3.7, chi_eV=4.2, Nc_cm3=1e21, Nv_cm3=1e21,
                mun_cm2Vs=20.0, mup_cm2Vs=10.0),
    "ITO": dict(eps_r=9.0, Eg_eV=3.6, chi_eV=4.3, Nc_cm3=1e21, Nv_cm3=1e21,
                mun_cm2Vs=30.0, mup_cm2Vs=10.0),
    "TIO2": dict(eps_r=10.0, Eg_eV=3.2, chi_eV=4.0, Nc_cm3=5e20, Nv_cm3=5e20,
                 mun_cm2Vs=0.1, mup_cm2Vs=0.01),
    "MAPBI3": dict(eps_r=24.0, Eg_eV=1.6, chi_eV=3.93, Nc_cm3=2.2e18,
                   Nv_cm3=1.0e19, mun_cm2Vs=1.6, mup_cm2Vs=1.7,
                   taun_s=8e-9, taup_s=8e-9),
    "SPIRO": dict(eps_r=3.0, Eg_eV=3.0, chi_eV=2.5, Nc_cm3=5e19, Nv_cm3=5e19,
                  mun_cm2Vs=0.01, mup_cm2Vs=0.02),
    "C60": dict(eps_r=4.0, Eg_eV=1.8, chi_eV=4.0, Nc_cm3=1e18, Nv_cm3=1e18,
                mun_cm2Vs=1.0, mup_cm2Vs=1.0),
    "AU": dict(eps_r=1.0, Eg_eV=1.0, chi_eV=4.7, Nc_cm3=1e22, Nv_cm3=1e22,
               mun_cm2Vs=100.0, mup_cm2Vs=80.0),
    "AG": dict(eps_r=1.0, Eg_eV=1.0, chi_eV=4.3, Nc_cm3=1e22, Nv_cm3=1e22,
               mun_cm2Vs=100.0, mup_cm2Vs=80.0),
}

# 只作为材料属性的字段（掺杂是逐层的，不进 MaterialSet.overrides）
_INTRINSIC_FIELDS = ("eps_r", "Eg_eV", "chi_eV", "Nc_cm3", "Nv_cm3",
                     "mun_cm2Vs", "mup_cm2Vs", "taun_s", "taup_s")


class DeckCompiler:
    """``compile(deck) -> ProblemSpec``；``compile_text`` 是便捷入口。"""

    def __init__(self, material_db: Optional[Any] = None,
                 log: Optional[Callable[[str], None]] = None):
        self._db = material_db
        self._log = log or (lambda _msg: None)

    # ----- 入口 ---------------------------------------------------------
    def compile_text(self, text: str) -> ProblemSpec:
        from .parser import parse_deck
        return self.compile(parse_deck(text))

    def compile(self, deck: ast.Deck) -> ProblemSpec:
        for warning in deck.warnings:
            self._log(warning)

        layers, resolved = self._build_layers(deck)
        device = self._build_device(deck, layers)
        spec = ProblemSpec(
            device=device,
            optics=self._build_optics(deck),
            sweep=self._build_sweep(deck),
            material=MaterialSet(overrides={
                key: {k: v for k, v in params.items() if k in _INTRINSIC_FIELDS}
                for key, params in resolved.items()
            }),
            output=self._build_output(deck),
        )

        fatal, warnings = spec.validate_notes()
        if fatal:
            raise SpecError("deck 编译出的 ProblemSpec 未通过校验",
                            details=fatal + warnings)
        for warning in warnings:
            self._log(f"deck 警告：{warning}")
        return spec

    # ----- 材料 ---------------------------------------------------------
    def _lookup(self, key: str) -> Dict[str, float]:
        if self._db is not None:
            got = None
            if hasattr(self._db, "lookup"):
                got = self._db.lookup(key)
            elif isinstance(self._db, Mapping):
                got = self._db.get(key, self._db.get(key.upper()))
            if got:
                return dict(got)
        return dict(BUILTIN_MATERIALS.get(key, {}))

    # ----- 层栈 ---------------------------------------------------------
    def _build_layers(self, deck: ast.Deck) -> Tuple[Tuple[LayerSpec, ...],
                                                     Dict[str, Dict[str, float]]]:
        regions = deck.of(ast.RegionStmt)
        if not regions:
            raise SpecError("deck 缺少 region 语句",
                            details=["至少需要一个 region 定义层栈"])

        numbers: List[int] = []
        for i, region in enumerate(regions):
            num = region.number if region.number is not None else i + 1
            if num in numbers:
                raise SpecError(f"region 编号重复：{num}",
                                details=[f"已用编号 {numbers}"])
            numbers.append(num)

        overrides = {m.name.upper(): m.as_dict()
                     for m in deck.of(ast.MaterialStmt)}

        doping_by_region: Dict[int, List[ast.DopingStmt]] = {}
        for doping in deck.of(ast.DopingStmt):
            if doping.region is None:
                raise SpecError("doping 缺少 region=",
                                details=["1D 器件按 region 编号施加掺杂"])
            if doping.region not in numbers:
                raise SpecError(f"doping 指向不存在的 region {doping.region}",
                                details=[f"已有编号 {numbers}"])
            doping_by_region.setdefault(doping.region, []).append(doping)

        forced_contacts = {c.region for c in deck.of(ast.ContactStmt)
                          if c.region is not None}

        layers: List[LayerSpec] = []
        resolved: Dict[str, Dict[str, float]] = {}
        n = len(regions)

        for i, region in enumerate(regions):
            num = region.number if region.number is not None else i + 1
            if region.thickness_nm is None or region.thickness_nm <= 0:
                raise SpecError(
                    f"region[{num}] '{region.material}' 缺少正的 thickness=",
                    details=["厚度单位为 nm"],
                )

            key = region.material.upper()
            params = self._lookup(key)
            if not params:
                raise SpecError(
                    f"未知材料 '{region.material}'（region[{num}]）",
                    details=[f"内建材料：{', '.join(sorted(BUILTIN_MATERIALS))}",
                             "或在 deck 里用 material name=... eg=... 显式给参数"],
                )
            params.update(overrides.get(key, {}))
            resolved.setdefault(key, dict(params))

            for doping in doping_by_region.get(num, ()):
                field = "Nd_cm3" if doping.kind == "n" else "Na_cm3"
                params[field] = float(doping.conc)

            # 首/末层是边界条件挂载点（DeviceSpec 的强制要求），
            # contact 语句可以额外把中间层也标成挂载点
            is_contact = num in forced_contacts or i in (0, n - 1)
            layers.append(LayerSpec(
                name=region.name or region.material,
                material=region.material,
                thick_nm=float(region.thickness_nm),
                is_contact=is_contact,
                **params,
            ))

        for i, layer in ((0, layers[0]), (n - 1, layers[-1])):
            if layer.material.upper() not in CONTACT_MATERIALS:
                self._log(f"deck 提示：{'首' if i == 0 else '末'}层 "
                          f"'{layer.name}' 作为"
                          f"{'阳极' if i == 0 else '阴极'}挂载点（非典型电极材料）")
        return tuple(layers), resolved

    def _build_device(self, deck: ast.Deck,
                      layers: Tuple[LayerSpec, ...]) -> DeviceSpec:
        mesh = deck.last(ast.MeshStmt)
        nodes = mesh.nodes if mesh is not None and mesh.nodes is not None else 101
        refine = mesh.refine if mesh is not None and mesh.refine is not None else 0.0
        t_k = mesh.temperature if mesh is not None and mesh.temperature is not None else 300.0
        if nodes < 3:
            raise SpecError(f"mesh nodes={nodes} 太少，至少 3")
        return DeviceSpec(layers=layers, T_K=float(t_k), nodes=int(nodes),
                          interface_refine=float(refine))

    # ----- 光学 ---------------------------------------------------------
    def _build_optics(self, deck: ast.Deck) -> OpticsSpec:
        base = OpticsSpec.default_am15g()
        stmt = deck.last(ast.OpticsStmt)
        if stmt is None:
            return base

        if stmt.spectrum is not None and stmt.spectrum.lower() != "am15g":
            raise SpecError(f"optics spectrum='{stmt.spectrum}' 不支持",
                            details=["目前只支持 am15g"])

        changes: Dict[str, Any] = {}
        if stmt.intensity is not None:
            changes["intensity_mW_cm2"] = float(stmt.intensity)
        if stmt.bounces is not None:
            changes["bounces"] = int(stmt.bounces)
        if stmt.bandgap_mask is not None:
            changes["bandgap_mask"] = bool(stmt.bandgap_mask)
        if stmt.nk_dir is not None:
            changes["nk_dir"] = stmt.nk_dir
        return replace(base, **changes)

    # ----- 求解器 + 扫描 ------------------------------------------------
    def _build_sweep(self, deck: ast.Deck) -> SweepSpec:
        solver = self._build_solver(deck)

        solve = deck.last(ast.SolveStmt)
        v_start = 0.0
        v_stop = 1.8
        n_points = 37
        if solve is not None:
            if solve.v_start is not None:
                v_start = float(solve.v_start)
            if solve.v_stop is not None:
                v_stop = float(solve.v_stop)
            if solve.n_points is not None:
                n_points = int(solve.n_points)
            elif solve.v_step is not None:
                if solve.v_step <= 0:
                    raise SpecError(f"solve vstep={solve.v_step} 必须 > 0")
                n_points = int(round((v_stop - v_start) / solve.v_step)) + 1
            if solve.log:
                self._log("solve log：SSF 落盘由 dataio 负责，内核不直接写文件")

        bc_type, wf_anode, wf_cathode, an_fixed, ca_fixed = self._build_contacts(deck)
        return SweepSpec(
            v_start=v_start, v_stop=v_stop, n_points=max(1, n_points),
            anode_V_fixed=an_fixed, cathode_V_fixed=ca_fixed, sweep_side="ANODE",
            bc_type=bc_type, work_fn_anode=wf_anode, work_fn_cathode=wf_cathode,
            solver_opts=solver,
        )

    def _build_solver(self, deck: ast.Deck) -> SolverOpts:
        solver = SolverOpts()
        stmt = deck.last(ast.MethodStmt)
        if stmt is None:
            return solver

        changes: Dict[str, Any] = {}
        if stmt.method is not None:
            if stmt.method not in SolverOpts.VALID_METHODS:
                raise SpecError(
                    f"method '{stmt.method}' 不支持",
                    details=[f"可选：{', '.join(SolverOpts.VALID_METHODS)}"],
                )
            changes["method"] = stmt.method
        if stmt.max_iter is not None:
            changes["max_iter"] = int(stmt.max_iter)
        if stmt.tol is not None:
            changes["res_tol"] = float(stmt.tol)
        if stmt.damping is not None:
            changes["damping"] = float(stmt.damping)
        return replace(solver, **changes)

    def _build_contacts(self, deck: ast.Deck) -> Tuple[str, float, float, float, float]:
        bc_type = "OHMIC"
        wf_anode, wf_cathode = 4.7, 4.7
        an_fixed, ca_fixed = 0.0, 0.0

        for contact in deck.of(ast.ContactStmt):
            name = contact.name.lower()
            if name not in ("anode", "cathode"):
                raise SpecError(
                    f"contact name='{contact.name}' 非法",
                    details=["只支持 anode / cathode"],
                )
            if contact.workfunction is not None:
                bc_type = "SCHOTTKY"
                if name == "anode":
                    wf_anode = float(contact.workfunction)
                else:
                    wf_cathode = float(contact.workfunction)
            if contact.voltage is not None:
                if name == "anode":
                    an_fixed = float(contact.voltage)
                    self._log("contact anode voltage 会被 J-V 扫描覆盖"
                              "（SweepSpec.sweep_side=ANODE）")
                else:
                    ca_fixed = float(contact.voltage)
        return bc_type, wf_anode, wf_cathode, an_fixed, ca_fixed

    # ----- 输出 ---------------------------------------------------------
    def _build_output(self, deck: ast.Deck) -> OutputSpec:
        default = OutputSpec()
        stmt = deck.last(ast.OutputStmt)
        if stmt is None:
            return default

        fields: Tuple[str, ...] = default.fields
        if stmt.fields:
            resolved = []
            for raw in stmt.fields:
                name = FIELD_ALIASES.get(raw.upper(), raw.upper())
                if name not in VALID_FIELDS:
                    raise SpecError(
                        f"output fields 含未知场量 '{raw}'",
                        details=[f"可用：{', '.join(VALID_FIELDS)}"],
                    )
                resolved.append(name)
            fields = tuple(dict.fromkeys(resolved))   # 去重且保序

        run_name = stmt.run_name or default.run_name
        fmt = stmt.format.lower() if stmt.format else default.format
        if fmt not in ("npy", "ssf"):
            raise SpecError(f"output format='{stmt.format}' 非法",
                            details=["可选：npy / ssf"])
        if stmt.file:
            self._log(f"output file='{stmt.file}'：文件命名由 dataio 负责")
        return replace(default, fields=fields, run_name=run_name, format=fmt)
