"""script — Silvaco Atlas deck 兼容层（L3）。

    deck = parse_deck(text)                     # 文本 → AST（未知语句只警告）
    spec = DeckCompiler().compile(deck)         # AST → ProblemSpec
    # 或一步到位：
    spec = DeckCompiler().compile_text(text)

语句子集：``title / mesh / region / doping / contact / material / models /
optics / method / solve / log / output``；未知语句与未知参数不致命。
"""

from . import ast_nodes
from .ast_nodes import Deck
from .compiler import DeckCompiler
from .lexer import DeckSyntaxError, Token, tokenize
from .parser import parse_deck

__all__ = [
    "parse_deck", "Deck", "DeckCompiler", "DeckSyntaxError",
    "tokenize", "Token", "ast_nodes",
]
