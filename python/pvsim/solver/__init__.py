"""solver — 求解编排层。

对外只暴露 :class:`Simulator`：吃 ``ProblemSpec`` / ``SweepPlan``，
产出 ``SimResult``。本层不解析 deck、不做绘图、不落盘（落盘归 dataio）。
"""

from .simulator import (
    LOG_DEBUG,
    LOG_ERROR,
    LOG_INFO,
    LOG_WARN,
    FieldTable,
    JVData,
    PlanRun,
    ProgressHook,
    SimResult,
    Simulator,
    SolvePoint,
)

__all__ = [
    "Simulator",
    "SimResult",
    "PlanRun",
    "FieldTable",
    "JVData",
    "SolvePoint",
    "ProgressHook",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARN",
    "LOG_ERROR",
]
