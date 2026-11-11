"""求解工作线程：GUI 主循环不阻塞。

线程只做三件事：调 ``Simulator.run``、把日志/进度/结果塞进 ``queue``、
退出。Tk 侧用 ``after()`` 轮询该队列（见 ``App._poll_queue``），
绝不跨线程直接碰控件。
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional

EVT_LOG = "log"
EVT_POINT = "point"
EVT_DONE = "done"
EVT_ERROR = "error"


@dataclass(frozen=True)
class SolveEvent:
    """线程 → GUI 的一条消息。"""

    kind:    str
    payload: Any = None
    level:   int = 0
    index:   int = 0
    total:   int = 0


class _QueueHook:
    """把 ``ProgressHook`` 回调转成队列事件。"""

    def __init__(self, out: "queue.Queue[SolveEvent]"):
        self._out = out

    def on_log(self, level: int, message: str) -> None:
        self._out.put(SolveEvent(EVT_LOG, message, level=level))

    def on_point(self, index: int, total: int, point: Any) -> None:
        self._out.put(SolveEvent(EVT_POINT, point, index=index, total=total))


class SolveWorker(threading.Thread):
    """单次求解线程；每次运行新建一个实例，不可复用。"""

    def __init__(self, spec: Any, simulator: Any,
                 out: Optional["queue.Queue[SolveEvent]"] = None,
                 name: str = "pvsim-solve"):
        super().__init__(name=name, daemon=True)
        self._spec = spec
        self._sim = simulator
        self.out: "queue.Queue[SolveEvent]" = out if out is not None else queue.Queue()

    def run(self) -> None:      # noqa: D401 - Thread API
        try:
            result = self._sim.run(self._spec, hook=_QueueHook(self.out))
        except BaseException as exc:      # 线程里任何异常都要变成事件
            self.out.put(SolveEvent(EVT_ERROR, exc))
            return
        self.out.put(SolveEvent(EVT_DONE, result))
