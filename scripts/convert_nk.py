"""把外部 n,k 文件规范成内核可读的 ``<name>.txt``（λ 用 µm，空白分隔）。

内核的 ``nk_parse`` 只认「首列 µm + 空白分隔 + 可选三列」；而常见数据源是
nm、逗号分隔、带表头（如 ``cfab/Au_Formatted.txt``）。本脚本做一次性转换，
不改内核解析逻辑。

用法::

    python scripts/convert_nk.py --src D:/交互界面_data/cfab/Au_Formatted.txt \
        --out-dir D:/交互界面_data/datasets/nk --name Au
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0]
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                nums = []
                break
        if len(nums) < 3:
            continue
        rows.append((nums[0], nums[1], nums[2]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="n,k 文件规范化（nm/逗号 → µm/空白）")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None, help="输出文件名（不含 .txt）")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or src.stem.replace("_Formatted", "").replace("_formatted", "")
    out = out_dir / f"{name}.txt"

    rows = load_rows(src)
    if len(rows) < 2:
        raise SystemExit(f"没有解析到有效数据行：{src}")
    scale = 1e-3 if sorted(r[0] for r in rows)[len(rows) // 2] > 10.0 else 1.0
    rows = sorted((lam * scale, n, k) for lam, n, k in rows)
    rows = [r for i, r in enumerate(rows) if i == 0 or r[0] > rows[i - 1][0]]

    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(rows)}\n")
        for lam, n, k in rows:
            f.write(f"{lam:.6g} {n:.6g} {k:.6g}\n")
    print(f"{src.name} → {out}（{len(rows)} 点，"
          f"{rows[0][0]:.4g}–{rows[-1][0]:.4g} µm，"
          f"{'nm→µm' if scale != 1.0 else 'µm 原样'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
