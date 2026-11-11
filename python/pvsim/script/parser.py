"""deck 解析：Token 流 → :class:`Deck`。

容错原则
--------
* 语法错误（缺值、引号未闭合、非法字符）抛 :class:`DeckSyntaxError`；
* 未知语句 → 记 :class:`UnknownStmt` + 一条警告，继续解析；
* 已知语句里的未知参数 → 记警告并忽略该参数；
* 语义错误（材料不存在、region 编号重复、缺 thickness）留给编译器，
  因为那需要材料库/器件上下文。

语句子集（Atlas 风格，详见 ``docs/deck_syntax.md``）::

    title <自由文本>
    mesh nodes=101 interface.refine=1.0 space.mult=1.0
    region num=1 material=FTO thickness=200
    doping region=1 n.type conc=1e20
    contact name=anode workfunction=4.7
    material name=MAPBI3 eg=1.6 taun=8e-9
    models srh
    optics spectrum=am15g intensity=100 bounces=1 bandgap.mask
    method newton maxiter=60 tol=1e-8 damping=1.0
    solve vstart=0 vfinal=1.8 vstep=0.05
    log file=out/jv.log
    output fields=X,PSI,N,P,EFIELD,JTOTAL,G_OPT run=mapbi3
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import ast_nodes as ast
from .lexer import (
    COMMA,
    EOF,
    EQ,
    LPAREN,
    NEWLINE,
    NUMBER,
    RPAREN,
    STRING,
    WORD,
    DeckSyntaxError,
    Token,
    tokenize,
)

# deck 参数名 → LayerSpec 字段名
_PARAM_KEYS: Dict[str, str] = {
    "eg": "Eg_eV", "eg.ev": "Eg_eV",
    "chi": "chi_eV",
    "eps": "eps_r", "eps.r": "eps_r",
    "mun": "mun_cm2Vs", "mup": "mup_cm2Vs",
    "nc": "Nc_cm3", "nv": "Nv_cm3",
    "taun": "taun_s", "taup": "taup_s",
    "nd": "Nd_cm3", "na": "Na_cm3",
}

ValueMap = Dict[str, Any]     # key → Token | List[Token]


# ---------------------------------------------------------------------------
# 参数表工具
# ---------------------------------------------------------------------------

def _parse_args(tokens: Sequence[Token]) -> Tuple[ValueMap, List[str], List[Token]]:
    values: ValueMap = {}
    flags: List[str] = []
    positional: List[Token] = []

    j, n = 0, len(tokens)
    while j < n:
        tok = tokens[j]

        if tok.kind in (LPAREN, RPAREN):
            raise DeckSyntaxError("暂不支持括号，请用逗号分隔的列表",
                                  tok.line, tok.col)

        if tok.kind == WORD and j + 1 < n and tokens[j + 1].kind == EQ:
            key = tok.value.lower()
            j += 2
            if j >= n:
                raise DeckSyntaxError(f"参数 {key}= 缺少值", tok.line, tok.col)
            if key in values:
                raise DeckSyntaxError(f"参数 {key} 重复赋值", tok.line, tok.col)

            first = tokens[j]
            j += 1
            if first.kind not in (WORD, NUMBER, STRING):
                raise DeckSyntaxError(f"参数 {key}= 的值非法：{first.value!r}",
                                      first.line, first.col)

            if j < n and tokens[j].kind == COMMA:
                items = [first]
                while j < n and tokens[j].kind == COMMA:
                    comma = tokens[j]
                    j += 1
                    if j >= n or tokens[j].kind not in (WORD, NUMBER, STRING):
                        raise DeckSyntaxError(f"参数 {key} 的列表缺少元素",
                                              comma.line, comma.col)
                    items.append(tokens[j])
                    j += 1
                values[key] = items
            else:
                values[key] = first
            continue

        if tok.kind == WORD:
            flags.append(tok.value.lower())
        elif tok.kind in (NUMBER, STRING):
            positional.append(tok)
        else:
            raise DeckSyntaxError(f"意外的记号 {tok.value!r}", tok.line, tok.col)
        j += 1

    return values, flags, positional


def _get(values: ValueMap, *names: str) -> Optional[Any]:
    for name in names:
        if name in values:
            return values[name]
    return None


def _num(values: ValueMap, *names: str, default: Optional[float] = None) -> Optional[float]:
    item = _get(values, *names)
    if item is None:
        return default
    if isinstance(item, list) or item.kind != NUMBER:
        tok = item[0] if isinstance(item, list) else item
        raise DeckSyntaxError(f"参数 {names[0]} 需要数值，得到 {tok.value!r}",
                              tok.line, tok.col)
    return float(item.value)


def _int(values: ValueMap, *names: str, default: Optional[int] = None) -> Optional[int]:
    val = _num(values, *names, default=None)
    if val is None:
        return default
    if abs(val - round(val)) > 1e-9:
        item = _get(values, *names)
        tok = item[0] if isinstance(item, list) else item
        raise DeckSyntaxError(f"参数 {names[0]} 需要整数，得到 {val:g}",
                              tok.line, tok.col)
    return int(round(val))


def _word(values: ValueMap, *names: str, default: Optional[str] = None) -> Optional[str]:
    item = _get(values, *names)
    if item is None:
        return default
    if isinstance(item, list):
        raise DeckSyntaxError(f"参数 {names[0]} 需要单个值", item[0].line, item[0].col)
    if item.kind not in (WORD, STRING):
        raise DeckSyntaxError(f"参数 {names[0]} 需要字符串，得到 {item.value!r}",
                              item.line, item.col)
    return item.value


def _bool(values: ValueMap, *names: str, default: Optional[bool] = None) -> Optional[bool]:
    item = _get(values, *names)
    if item is None:
        return default
    if isinstance(item, list) or item.kind not in (WORD, NUMBER):
        tok = item[0] if isinstance(item, list) else item
        raise DeckSyntaxError(f"参数 {names[0]} 需要布尔值", tok.line, tok.col)
    text = item.value.lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    raise DeckSyntaxError(f"参数 {names[0]} 的布尔值非法：{item.value!r}",
                          item.line, item.col)


def _reject_unknown(values: ValueMap, flags: Sequence[str], allowed: Sequence[str],
                    line: int, warnings: List[str], keyword: str) -> None:
    allowed_set = set(allowed)
    for key in values:
        if key not in allowed_set:
            warnings.append(f"L{line}: {keyword} 不支持参数 '{key}'（已忽略）")
    for flag in flags:
        if flag not in allowed_set:
            warnings.append(f"L{line}: {keyword} 不支持参数 '{flag}'（已忽略）")


def _field_list(values: ValueMap, key: str) -> Tuple[str, ...]:
    item = _get(values, key)
    if item is None:
        return ()
    tokens = item if isinstance(item, list) else [item]
    return tuple(t.value.upper() for t in tokens)


# ---------------------------------------------------------------------------
# 各语句的处理函数
# ---------------------------------------------------------------------------

def _p_mesh(values: ValueMap, flags: List[str], word: Token,
            warnings: List[str]) -> ast.MeshStmt:
    _reject_unknown(values, flags,
                    ("nodes", "refine", "interface.refine", "space.mult",
                     "space", "temperature", "temp"),
                    word.line, warnings, "mesh")
    return ast.MeshStmt(
        nodes=_int(values, "nodes"),
        refine=_num(values, "refine", "interface.refine"),
        space_mult=_num(values, "space.mult", "space"),
        temperature=_num(values, "temperature", "temp"),
        line=word.line,
    )


def _p_region(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.RegionStmt:
    _reject_unknown(values, flags,
                    ("num", "number", "material", "thickness", "thick", "name"),
                    word.line, warnings, "region")
    material = _word(values, "material")
    if material is None:
        raise DeckSyntaxError("region 缺少 material=", word.line, word.col)
    return ast.RegionStmt(
        material=material,
        thickness_nm=_num(values, "thickness", "thick"),
        number=_int(values, "num", "number"),
        name=_word(values, "name"),
        line=word.line,
    )


def _p_doping(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.DopingStmt:
    _reject_unknown(values, flags,
                    ("region", "conc", "concentration", "doping",
                     "nd", "na", "type", "n.type", "p.type"),
                    word.line, warnings, "doping")

    kind = None
    conc = None
    if "n.type" in flags or _word(values, "type") == "n":
        kind = "n"
    if "p.type" in flags or _word(values, "type") == "p":
        if kind is not None:
            raise DeckSyntaxError("doping 同时指定了 n 型和 p 型", word.line, word.col)
        kind = "p"

    nd = _num(values, "nd")
    na = _num(values, "na")
    if nd is not None and na is not None:
        raise DeckSyntaxError("doping 不能同时给 nd= 与 na=", word.line, word.col)
    if nd is not None:
        kind, conc = "n", nd
    if na is not None:
        kind, conc = "p", na
    if kind is None:
        kind = "n"
    if conc is None:
        conc = _num(values, "conc", "concentration", "doping")
    if conc is None or conc <= 0:
        raise DeckSyntaxError("doping 需要正的 conc=", word.line, word.col)

    return ast.DopingStmt(kind=kind, conc=float(conc),
                          region=_int(values, "region"), line=word.line)


def _p_contact(values: ValueMap, flags: List[str], word: Token,
               warnings: List[str]) -> ast.ContactStmt:
    _reject_unknown(values, flags,
                    ("name", "region", "voltage", "bias", "workfunction",
                     "work.function", "workfun"),
                    word.line, warnings, "contact")
    name = _word(values, "name")
    if name is None:
        raise DeckSyntaxError("contact 缺少 name=anode|cathode", word.line, word.col)
    return ast.ContactStmt(
        name=name,
        region=_int(values, "region"),
        voltage=_num(values, "voltage", "bias"),
        workfunction=_num(values, "workfunction", "work.function", "workfun"),
        line=word.line,
    )


def _p_material(values: ValueMap, flags: List[str], word: Token,
                warnings: List[str]) -> ast.MaterialStmt:
    _reject_unknown(values, flags,
                    ("name", "material") + tuple(_PARAM_KEYS),
                    word.line, warnings, "material")
    name = _word(values, "name", "material")
    if name is None:
        raise DeckSyntaxError("material 缺少 name=", word.line, word.col)

    params: List[Tuple[str, float]] = []
    for key, field in _PARAM_KEYS.items():
        if key in values:
            num = _num(values, key)
            if num is not None:
                params.append((field, float(num)))
    if not params:
        warnings.append(f"L{word.line}: material '{name}' 没有任何参数覆盖")
    return ast.MaterialStmt(name=name, params=tuple(sorted(params)), line=word.line)


def _p_models(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.ModelStmt:
    _reject_unknown(values, (), (), word.line, warnings, "models")
    if not flags:
        warnings.append(f"L{word.line}: models 语句没有模型名")
    return ast.ModelStmt(models=tuple(flags), line=word.line)


def _p_optics(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.OpticsStmt:
    _reject_unknown(values, flags,
                    ("spectrum", "intensity", "bounces", "bandgap.mask",
                     "mask", "nk.dir"),
                    word.line, warnings, "optics")
    mask = _bool(values, "bandgap.mask", "mask")
    if mask is None and ("mask" in flags or "bandgap.mask" in flags):
        mask = True
    return ast.OpticsStmt(
        intensity=_num(values, "intensity"),
        bounces=_int(values, "bounces"),
        bandgap_mask=mask,
        spectrum=_word(values, "spectrum"),
        nk_dir=_word(values, "nk.dir"),
        line=word.line,
    )


def _p_method(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.MethodStmt:
    _reject_unknown(values, flags,
                    ("method", "maxiter", "max.iter", "tol", "res.tol",
                     "damping", "newton", "gummel", "warmstart", "block"),
                    word.line, warnings, "method")

    method = None
    if "gummel" in flags:
        method = "GUMMEL"
    if "newton" in flags:
        method = "NEWTON"
    if "warmstart" in flags:
        method = "NEWTON_WITH_WARMSTART"
    given = _word(values, "method")
    if given is not None:
        method = given.upper()

    return ast.MethodStmt(
        method=method,
        max_iter=_int(values, "maxiter", "max.iter"),
        tol=_num(values, "tol", "res.tol"),
        damping=_num(values, "damping"),
        line=word.line,
    )


def _p_solve(values: ValueMap, flags: List[str], word: Token,
             warnings: List[str]) -> ast.SolveStmt:
    _reject_unknown(values, flags,
                    ("vstart", "v.start", "vbegin", "vfinal", "vstop",
                     "v.stop", "vstep", "dv", "npoints", "n.points", "np", "log"),
                    word.line, warnings, "solve")
    return ast.SolveStmt(
        v_start=_num(values, "vstart", "v.start", "vbegin"),
        v_stop=_num(values, "vfinal", "vstop", "v.stop"),
        v_step=_num(values, "vstep", "dv"),
        n_points=_int(values, "npoints", "n.points", "np"),
        log=("log" in flags) or bool(_bool(values, "log", default=False)),
        line=word.line,
    )


def _p_output(values: ValueMap, flags: List[str], word: Token,
              warnings: List[str]) -> ast.OutputStmt:
    _reject_unknown(values, flags,
                    ("fields", "field", "run", "run.name", "name", "file", "format"),
                    word.line, warnings, "output")
    return ast.OutputStmt(
        fields=_field_list(values, "fields") or _field_list(values, "field"),
        run_name=_word(values, "run", "run.name", "name"),
        file=_word(values, "file"),
        format=_word(values, "format"),
        line=word.line,
    )


def _p_log(values: ValueMap, flags: List[str], word: Token,
           warnings: List[str]) -> ast.LogStmt:
    _reject_unknown(values, flags, ("file", "path", "outfile", "outf"),
                    word.line, warnings, "log")
    path = _word(values, "file", "path", "outfile", "outf")
    if not path:
        raise DeckSyntaxError("log 缺少 file=", word.line, word.col)
    return ast.LogStmt(path=path, line=word.line)


_HANDLERS: Dict[str, Callable[[ValueMap, List[str], Token, List[str]], Any]] = {
    "mesh": _p_mesh,
    "region": _p_region,
    "doping": _p_doping,
    "contact": _p_contact,
    "material": _p_material,
    "models": _p_models,
    "model": _p_models,
    "optics": _p_optics,
    "method": _p_method,
    "solve": _p_solve,
    "output": _p_output,
    "log": _p_log,
}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_deck(text: str) -> ast.Deck:
    """把 deck 文本解析成 :class:`Deck`（不抛未知语句错误）。"""
    toks = tokenize(text)
    statements: List[Any] = []
    warnings: List[str] = []
    title = ""

    i = 0
    while toks[i].kind != EOF:
        if toks[i].kind == NEWLINE:
            i += 1
            continue

        start = i
        while toks[i].kind not in (NEWLINE, EOF):
            i += 1
        span = toks[start:i]
        word = span[0]
        keyword = word.value.lower()

        if keyword == "title":
            title = " ".join(t.value for t in span[1:])
            continue

        values, flags, _positional = _parse_args(span[1:])
        handler = _HANDLERS.get(keyword)
        if handler is None:
            raw = " ".join(t.value for t in span)
            warnings.append(f"L{word.line}: 未知语句 '{word.value}' 已忽略")
            statements.append(ast.UnknownStmt(keyword=keyword, raw=raw, line=word.line))
            continue

        statements.append(handler(values, flags, word, warnings))

    return ast.Deck(statements=tuple(statements), warnings=tuple(warnings), title=title)
