"""ResultStore — run 目录管理与原子提交。

解决的问题
----------
旧流程最典型的坑是"启动即清空 Outputs/"，一次失败的新运行会把上一次
的结果一起埋掉。这里改成：

    begin()  → 在 ``results/.tmp_<run_id>_<uuid>/`` 里写
    commit() → 写完后整目录原子改名为 ``results/<run_id>/``

读者永远只能看到已提交的 run；写一半失败时临时目录用 ``discard()``
清掉，旧结果毫发无损。

目录布局（layout_version = 1）
------------------------------
::

    <root>/results/<run_id>/
        spec.json      ProblemSpec.to_dict()，完整可回溯
        meta.json      run_id / 时间 / 收敛情况 / 布局版本 / 场量清单
        points.json    每个偏压点的摘要
        jv.json        J-V 指标（NaN 写成 null，供人读；加载时重算）
        fields/
            x_nm.npy
            PSI.npy
            N.npy
            ...

加载时 J-V 指标一律由 points 重新计算，避免磁盘上的派生值与数据不一致。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..param.schema import ProblemSpec
from ..solver.simulator import FieldTable, JVData, SimResult, SolvePoint
from .silvaco_log import SilvacoLogWriter

LAYOUT_VERSION = 1


# ---------------------------------------------------------------------------
# JSON 辅助
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """让 numpy 标量/数组能被 json 序列化。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def _nan_to_none(x: float) -> Optional[float]:
    return None if (isinstance(x, float) and math.isnan(x)) else float(x)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False,
                  default=_json_default, allow_nan=True)
        fh.write("\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# RunHandle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunHandle:
    """``begin()`` 返回的写句柄：指向临时目录，commit 后改名。"""

    run_id:    str
    tmp_dir:   Path
    final_dir: Path

    @property
    def path(self) -> Path:
        """写入期使用临时目录。"""
        return self.tmp_dir


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

class ResultStore:
    """管理 ``<root>/results/`` 下的 run 目录。

    只做文件落盘/读回，不碰求解；与 ``Simulator`` 解耦，方便 GUI 把
    求解结果异步存盘（工作线程算完把 ``SimResult`` 丢过来即可）。

    参数
    ----
    root : 工作区根目录（``results/`` 会在其下自动创建）
    results_dirname : 结果目录名，默认 ``"results"``
    """

    def __init__(self, root: Union[str, Path], results_dirname: str = "results"):
        self._root = Path(root)
        self._results = self._root / results_dirname
        self._results.mkdir(parents=True, exist_ok=True)

    # ----- 基本信息 ------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    @property
    def results_dir(self) -> Path:
        return self._results

    def list_runs(self) -> List[str]:
        """已提交的 run_id（按名字排序，忽略临时目录）。"""
        return sorted(
            p.name for p in self._results.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def exists(self, run_id: str) -> bool:
        return self._run_dir(run_id).is_dir()

    # ----- 写：begin → commit -------------------------------------------
    def begin(self, spec: ProblemSpec, run_id: Optional[str] = None) -> RunHandle:
        """申请一个新 run_id 并创建临时目录（此时对读者不可见）。

        ``run_id`` 省略时按 ``<run_name>_<YYYYmmdd_HHMMSS>`` 生成；与已有
        run 冲突时自动追加 ``_2``、``_3``……，不覆盖旧结果。
        """
        # 显式传 "" 视为错误（`run_id or ...` 会把空串静默当成 None）
        base = run_id if run_id is not None else \
            f"{spec.output.run_name}_{datetime.now():%Y%m%d_%H%M%S}"
        self._validate_run_id(base)
        rid = self._unique_run_id(base)

        tmp = self._results / f".tmp_{rid}_{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        return RunHandle(run_id=rid, tmp_dir=tmp, final_dir=self._results / rid)

    def commit(self, handle: RunHandle, result: SimResult) -> Path:
        """把结果写进临时目录，然后原子改名为正式 run 目录。

        目标目录已存在时抛 ``FileExistsError``（绝不覆盖历史结果）。
        返回正式目录路径。
        """
        if handle.final_dir.exists():
            raise FileExistsError(
                f"run '{handle.run_id}' 已存在：{handle.final_dir}"
            )
        if not handle.tmp_dir.is_dir():
            raise FileNotFoundError(
                f"临时目录不存在（可能已 commit/discard）：{handle.tmp_dir}"
            )

        self._write_payload(handle.tmp_dir, handle.run_id, result)

        # 同卷内的目录改名是原子的：读者要么看到完整 run，要么看不到
        try:
            os.replace(handle.tmp_dir, handle.final_dir)
        except OSError:
            # 极端情况下（如目标卷不同）退化为 rename
            handle.tmp_dir.rename(handle.final_dir)
        return handle.final_dir

    def discard(self, handle: RunHandle) -> None:
        """放弃写入：删掉临时目录，已提交的结果不受影响。"""
        shutil.rmtree(handle.tmp_dir, ignore_errors=True)

    def save(self, result: SimResult, run_id: Optional[str] = None) -> Path:
        """一步完成 begin + commit；失败时自动清理临时目录。"""
        handle = self.begin(
            result.spec,
            run_id=run_id if run_id is not None else result.run_id,
        )
        try:
            return self.commit(handle, result)
        except BaseException:
            self.discard(handle)
            raise

    # ----- 读 ------------------------------------------------------------
    def load(self, run_id: str) -> SimResult:
        """读回一个已提交的 run；不存在时抛 ``FileNotFoundError``。"""
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run '{run_id}' 不存在：{run_dir}")

        meta = _read_json(run_dir / "meta.json")
        if int(meta.get("layout_version", -1)) != LAYOUT_VERSION:
            raise ValueError(
                f"run '{run_id}' 的布局版本为 {meta.get('layout_version')}，"
                f"当前代码只支持 {LAYOUT_VERSION}"
            )

        spec = ProblemSpec.from_dict(_read_json(run_dir / "spec.json"))
        points = tuple(
            SolvePoint(**p) for p in _read_json(run_dir / "points.json")
        )

        fields_dir = run_dir / "fields"
        x_nm = np.load(fields_dir / "x_nm.npy")
        data = {name: np.load(fields_dir / f"{name}.npy")
                for name in meta.get("fields", [])}

        # 派生指标一律重算，保证与 points 一致
        jv = JVData.from_points(
            [p.V for p in points],
            [p.J_mA_cm2 for p in points],
            pin_mW_cm2=spec.optics.intensity_mW_cm2,
        )

        return SimResult(
            run_id=str(meta.get("run_id", run_id)),
            spec=spec,
            points=points,
            fields=FieldTable(x_nm=x_nm, data=data),
            jv=jv,
        )

    # ----- 内部 ----------------------------------------------------------
    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or run_id.strip() == "":
            raise ValueError("run_id 不能为空")
        if run_id in (".", "..") or "/" in run_id or "\\" in run_id:
            raise ValueError(f"run_id 不能包含路径分隔符：{run_id!r}")

    def _run_dir(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self._results / run_id

    def _unique_run_id(self, base: str) -> str:
        if not (self._results / base).exists():
            return base
        i = 2
        while (self._results / f"{base}_{i}").exists():
            i += 1
        return f"{base}_{i}"

    def _write_payload(self, tmp: Path, run_id: str,
                       result: SimResult) -> None:
        fields_dir = tmp / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)

        _write_json(tmp / "spec.json", result.spec.to_dict())

        _write_json(tmp / "meta.json", {
            "layout_version": LAYOUT_VERSION,
            "run_id":         run_id,
            "sim_run_id":     result.run_id,
            "created_at":     datetime.now().isoformat(timespec="seconds"),
            "converged":      bool(result.converged),
            "n_points":       len(result.points),
            "n_node":         int(result.fields.n_node),
            "fields":         list(result.fields.names),
            "pin_mW_cm2":     float(result.spec.optics.intensity_mW_cm2),
        })

        _write_json(tmp / "points.json", [asdict(p) for p in result.points])

        jv = result.jv
        _write_json(tmp / "jv.json", {
            "V":           np.asarray(jv.V, dtype=float).tolist(),
            "J":           np.asarray(jv.J, dtype=float).tolist(),
            "Jsc":         _nan_to_none(jv.Jsc),
            "Voc":         _nan_to_none(jv.Voc),
            "FF":          _nan_to_none(jv.FF),
            "PCE":         _nan_to_none(jv.PCE),
            "Pmax_mW_cm2": _nan_to_none(jv.Pmax_mW_cm2),
            "pin_mW_cm2":  float(jv.pin_mW_cm2),
        })

        if result.spec.output.format == "ssf":
            SilvacoLogWriter().write(jv, tmp / "jv.log", run_id=run_id)

        np.save(fields_dir / "x_nm.npy", np.asarray(result.fields.x_nm, dtype=float))
        for name, arr in result.fields.data.items():
            np.save(fields_dir / f"{name}.npy", np.asarray(arr, dtype=float))
