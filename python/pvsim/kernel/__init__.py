"""kernel package — ctypes 低层级绑定。

导入约定::

    from pvsim.kernel.abi import PvkLib
    from pvsim.kernel import structs as st

kernel 层不持有任何业务逻辑（ProblemSpec 解析、SimResult 组装
等都在 solver/ 层），只做"Python ↔ C" 类型翻译和 ABI 校验。
"""

from .abi import PvkLib, PvkDLError, ABIError, PVK_ABI_VERSION
from . import structs

__all__ = ["PvkLib", "PvkDLError", "ABIError", "PVK_ABI_VERSION", "structs"]
