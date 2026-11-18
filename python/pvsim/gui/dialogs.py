"""DeckBuild 风格参数对话框 → 回写 deck 语句。

设计
----
* 单一数据源仍是编辑器里的 deck 文本：对话框只负责"读出现有值 → 改 → 生成
  一行语句 → 替换/追加到文本"，不另存一份参数状态，避免两套参数漂移；
* 生成与替换逻辑全是纯函数（``build_*`` / ``replace_statements``），可在无
  显示环境测试；只有 ``ask_*`` 才碰 Tk；
* 生成的语句按 Atlas 习惯写成单行，替换时会连带吃掉旧语句的 ``\\`` 续行块。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..script import ast_nodes as ast
from ..script.parser import parse_deck

# 支持的体复合/光学模型（与 compiler._SUPPORTED_MODELS 对齐）
MODEL_CHOICES = ("srh", "srh auger", "srh optr")
METHOD_CHOICES = ("NEWTON", "NEWTON_WITH_WARMSTART", "GUMMEL")
FIELD_CHOICES = ("X", "PSI", "N", "P", "EFIELD", "JN", "JP", "JTOTAL",
                 "R_NET", "G_OPT", "CB", "VB", "PHIN", "PHIP")


# ---------------------------------------------------------------------------
# 纯函数：生成语句
# ---------------------------------------------------------------------------

def _g(value: float) -> str:
    return f"{value:g}"


def build_mesh_stmt(nodes: int, refine: float, space_mult: float,
                    temperature: float) -> List[str]:
    return [f"mesh nodes={int(nodes)} interface.refine={_g(refine)} "
            f"space.mult={_g(space_mult)} temperature={_g(temperature)}"]


def build_physics_stmts(models: str, intensity: float, bounces: int,
                        bandgap_mask: bool) -> Dict[str, List[str]]:
    mask = " bandgap.mask" if bandgap_mask else ""
    return {
        "models": [f"models {models.strip() or 'srh'}"],
        "optics": [f"optics spectrum=am15g intensity={_g(intensity)} "
                   f"bounces={int(bounces)}{mask}"],
    }


def build_solve_stmts(method: str, maxiter: int, tol: float, damping: float,
                      v_start: float, v_stop: float,
                      v_step: float) -> Dict[str, List[str]]:
    return {
        "method": [f"method {method.lower()} maxiter={int(maxiter)} "
                   f"tol={_g(tol)} damping={_g(damping)}"],
        "solve": [f"solve vstart={_g(v_start)} vfinal={_g(v_stop)} "
                  f"vstep={_g(v_step)}"],
    }


def build_output_stmt(fields: Sequence[str], fmt: str, run: str) -> List[str]:
    names = ",".join(dict.fromkeys(fields)) or "X,PSI,N,P,EFIELD,JTOTAL,G_OPT"
    tail = f" format={fmt}" if fmt else ""
    if run:
        tail += f" run={run}"
    return [f"output fields={names}{tail}"]


# ---------------------------------------------------------------------------
# 纯函数：文本替换
# ---------------------------------------------------------------------------

def _head(line: str) -> str:
    body = line.strip()
    if not body or body.startswith("#"):
        return ""
    return body.split(None, 1)[0].lower()


def replace_statements(text: str,
                       replacements: Sequence[Tuple[Sequence[str], List[str]]]) -> str:
    """把每个关键字对应的**最后一条**语句替换成新行；没有就追加到末尾。

    ``replacements`` 形如 ``[(("models",), ["models srh"]), ...]``；
    关键字命中后会自动吞掉该语句的 ``\\`` 续行块。
    """
    lines = text.splitlines()
    for keywords, new_lines in replacements:
        keys = {k.lower() for k in keywords}
        start: Optional[int] = None
        for i, line in enumerate(lines):
            if _head(line) in keys:
                start = i
        if start is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines.extend(new_lines)
            lines.append("")
            continue
        end = start
        while lines[end].rstrip().endswith("\\") and end + 1 < len(lines):
            end += 1
        lines[start:end + 1] = list(new_lines)
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# 纯函数：从现有 deck 读出对话框初值
# ---------------------------------------------------------------------------

def current_values(text: str) -> Dict[str, Any]:
    """解析现有 deck，返回各对话框字段的初值（解析失败则用默认值）。"""
    out: Dict[str, Any] = {
        "nodes": 101, "refine": 1.0, "space_mult": 1.0, "temperature": 300.0,
        "models": "srh", "intensity": 100.0, "bounces": 100,
        "bandgap_mask": True,
        "method": "NEWTON_WITH_WARMSTART", "maxiter": 60, "tol": 1e-8,
        "damping": 1.0, "v_start": 0.0, "v_stop": 1.8, "v_step": 0.05,
        "fields": [], "format": "npy", "run": "device",
    }
    try:
        deck = parse_deck(text)
    except Exception:            # noqa: BLE001 - 初值失败不该挡住开对话框
        return out

    mesh = deck.last(ast.MeshStmt)
    if mesh is not None:
        out["nodes"] = mesh.nodes if mesh.nodes is not None else out["nodes"]
        out["refine"] = mesh.refine if mesh.refine is not None else out["refine"]
        out["space_mult"] = (mesh.space_mult if mesh.space_mult is not None
                             else out["space_mult"])
        out["temperature"] = (mesh.temperature if mesh.temperature is not None
                              else out["temperature"])

    models = deck.last(ast.ModelStmt)
    if models is not None and models.models:
        out["models"] = " ".join(models.models)

    optics = deck.last(ast.OpticsStmt)
    if optics is not None:
        if optics.intensity is not None:
            out["intensity"] = optics.intensity
        if optics.bounces is not None:
            out["bounces"] = optics.bounces
        if optics.bandgap_mask is not None:
            out["bandgap_mask"] = optics.bandgap_mask

    method = deck.last(ast.MethodStmt)
    if method is not None:
        if method.method:
            out["method"] = method.method
        if method.max_iter is not None:
            out["maxiter"] = method.max_iter
        if method.tol is not None:
            out["tol"] = method.tol
        if method.damping is not None:
            out["damping"] = method.damping

    solve = deck.last(ast.SolveStmt)
    if solve is not None:
        if solve.v_start is not None:
            out["v_start"] = solve.v_start
        if solve.v_stop is not None:
            out["v_stop"] = solve.v_stop
        if solve.v_step is not None:
            out["v_step"] = solve.v_step

    output = deck.last(ast.OutputStmt)
    if output is not None:
        if output.fields:
            out["fields"] = list(output.fields)
        if output.format:
            out["format"] = output.format.lower()
        if output.run_name:
            out["run"] = output.run_name
    return out


# ---------------------------------------------------------------------------
# Tk：通用表单对话框
# ---------------------------------------------------------------------------

class FormDialog(tk.Toplevel):
    """一个 key→(label, 控件) 的模态表单；确定后 ``result`` 是 dict。"""

    def __init__(self, master: tk.Misc, title: str,
                 specs: Sequence[Tuple[str, str, Any, str]],
                 choices: Optional[Dict[str, Sequence[str]]] = None):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[Dict[str, Any]] = None
        self._vars: Dict[str, Any] = {}
        self._kinds: Dict[str, str] = {}
        choices = choices or {}

        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        for row, (key, label, default, kind) in enumerate(specs):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w",
                                             padx=(0, 8), pady=3)
            if kind == "bool":
                var: Any = tk.BooleanVar(value=bool(default))
                ttk.Checkbutton(body, variable=var).grid(row=row, column=1,
                                                         sticky="w")
            elif key in choices:
                var = tk.StringVar(value=str(default))
                ttk.Combobox(body, textvariable=var, values=list(choices[key]),
                             state="readonly", width=26).grid(row=row, column=1,
                                                              sticky="ew")
            else:
                var = tk.StringVar(value=_g(default) if isinstance(
                    default, (int, float)) else str(default))
                ttk.Entry(body, textvariable=var, width=28).grid(
                    row=row, column=1, sticky="ew")
            self._vars[key] = var
            self._kinds[key] = kind

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="确定", command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=self.destroy).pack(
            side=tk.RIGHT, padx=6)
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.transient(master.winfo_toplevel())
        self.grab_set()

    def _ok(self) -> None:
        data: Dict[str, Any] = {}
        for key, var in self._vars.items():
            kind = self._kinds[key]
            val = var.get()
            if kind == "bool":
                data[key] = bool(val)
            elif kind == "int":
                data[key] = int(float(val))
            elif kind == "float":
                data[key] = float(val)
            else:
                data[key] = str(val).strip()
        self.result = data
        self.destroy()


def _run_dialog(master: tk.Misc, title: str,
                specs: Sequence[Tuple[str, str, Any, str]],
                choices: Optional[Dict[str, Sequence[str]]] = None
                ) -> Optional[Dict[str, Any]]:
    dlg = FormDialog(master, title, specs, choices)
    master.wait_window(dlg)
    return dlg.result


def ask_mesh(master: tk.Misc, cur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _run_dialog(master, "网格设置（Mesh）", [
        ("nodes", "目标节点数", cur["nodes"], "int"),
        ("refine", "界面加密比例", cur["refine"], "float"),
        ("space_mult", "网格间距倍率", cur["space_mult"], "float"),
        ("temperature", "温度 (K)", cur["temperature"], "float"),
    ])


def ask_physics(master: tk.Misc, cur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _run_dialog(master, "物理模型（Physics）", [
        ("models", "模型", cur["models"], "str"),
        ("intensity", "光强 (mW/cm²)", cur["intensity"], "float"),
        ("bounces", "反射次数", cur["bounces"], "int"),
        ("bandgap_mask", "带隙掩模", cur["bandgap_mask"], "bool"),
    ], choices={"models": MODEL_CHOICES})


def ask_solve(master: tk.Misc, cur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _run_dialog(master, "求解设置（Solve）", [
        ("method", "方法", cur["method"], "str"),
        ("maxiter", "最大迭代", cur["maxiter"], "int"),
        ("tol", "残差容限", cur["tol"], "float"),
        ("damping", "阻尼系数", cur["damping"], "float"),
        ("v_start", "起始偏压 (V)", cur["v_start"], "float"),
        ("v_stop", "终止偏压 (V)", cur["v_stop"], "float"),
        ("v_step", "电压步长 (V)", cur["v_step"], "float"),
    ], choices={"method": METHOD_CHOICES})


def ask_output(master: tk.Misc, cur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = _run_dialog(master, "输出设置（Output）", [
        ("fields", "场量（逗号分隔）", ",".join(cur["fields"]) or
         "X,PSI,N,P,EFIELD,JTOTAL,G_OPT", "str"),
        ("format", "格式", cur["format"], "str"),
        ("run", "run 名", cur["run"], "str"),
    ], choices={"format": ("npy", "ssf")})
    if data is not None:
        data["fields"] = [f.strip().upper() for f in str(data["fields"]).split(",")
                          if f.strip()]
    return data
