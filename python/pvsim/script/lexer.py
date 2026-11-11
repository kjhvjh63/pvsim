"""deck 词法：把 .tst 文本切成 Token 流。

语法约定（Atlas 子集）
----------------------
* 关键字、参数名大小写不敏感（统一转小写比较，值保留原样）；
* ``#`` 到行尾为注释；
* 行尾 ``\\`` 表示续行（反斜杠后只允许空白）；
* 值可以是数值、单词或加引号的字符串（字符串不跨行）；
* 语句以换行分隔，空行忽略。

词法错误一律抛 :class:`DeckSyntaxError` 并带行/列号——这类错误没法"容错继续"，
而未知语句/未知参数是解析层的事（见 parser，warn 不 fatal）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

WORD = "WORD"
NUMBER = "NUMBER"
STRING = "STRING"
EQ = "EQ"
COMMA = "COMMA"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
NEWLINE = "NEWLINE"
EOF = "EOF"

_WS = " \t\r\f\v"
# 允许路径式的单词（out/jv.log、paper-a.dat）；反斜杠留给续行，Windows 路径请加引号
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]*")
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class Token:
    kind:  str
    value: str
    line:  int
    col:   int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})@L{self.line}"


class DeckSyntaxError(ValueError):
    """deck 文本本身不合法（引号未闭合、缺值、非法字符等）。"""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        text = f"L{line}:C{col}: {message}" if line else message
        super().__init__(text)
        self.message = message
        self.line = line
        self.col = col


def tokenize(text: str) -> List[Token]:
    """把 deck 全文切成 Token 列表（末尾一定有 NEWLINE + EOF）。"""
    toks: List[Token] = []
    i, n = 0, len(text)
    line, col = 1, 1

    while i < n:
        ch = text[i]

        if ch in _WS:
            i += 1
            col += 1
            continue

        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue

        if ch == "\\":
            j = i + 1
            while j < n and text[j] in _WS:
                j += 1
            if j >= n or text[j] != "\n":
                raise DeckSyntaxError("'\\' 只能出现在行尾用于续行", line, col)
            i, line, col = j + 1, line + 1, 1
            continue

        if ch == "\n":
            toks.append(Token(NEWLINE, "\n", line, col))
            i, line, col = i + 1, line + 1, 1
            continue

        if ch == "=":
            toks.append(Token(EQ, ch, line, col))
            i, col = i + 1, col + 1
            continue
        if ch == ",":
            toks.append(Token(COMMA, ch, line, col))
            i, col = i + 1, col + 1
            continue
        if ch == "(":
            toks.append(Token(LPAREN, ch, line, col))
            i, col = i + 1, col + 1
            continue
        if ch == ")":
            toks.append(Token(RPAREN, ch, line, col))
            i, col = i + 1, col + 1
            continue

        if ch in "\"'":
            quote = ch
            start_line, start_col = line, col
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise DeckSyntaxError("字符串未闭合", start_line, start_col)
                if text[j] == "\n":
                    raise DeckSyntaxError("字符串不能跨行", start_line, start_col)
                if text[j] == quote:
                    j += 1
                    break
                buf.append(text[j])
                j += 1
            toks.append(Token(STRING, "".join(buf), start_line, start_col))
            col += j - i
            i = j
            continue

        if ch.isdigit() or (ch in "+-." and i + 1 < n
                            and (text[i + 1].isdigit() or text[i + 1] == ".")):
            m = _NUM_RE.match(text, i)
            if m is None:      # 例如孤立的小数点
                raise DeckSyntaxError(f"非法字符 {ch!r}", line, col)
            toks.append(Token(NUMBER, m.group(0), line, col))
            col += m.end() - i
            i = m.end()
            continue

        m = _WORD_RE.match(text, i)
        if m is not None:
            toks.append(Token(WORD, m.group(0), line, col))
            col += m.end() - i
            i = m.end()
            continue

        raise DeckSyntaxError(f"非法字符 {ch!r}", line, col)

    if not toks or toks[-1].kind != NEWLINE:
        toks.append(Token(NEWLINE, "\n", line, col))
    toks.append(Token(EOF, "", line, col))
    return toks
