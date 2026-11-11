"""绘图模块集合：import 即完成 ``REGISTRY`` 注册。

按设计文档拆成 electrical / optical / scans / structure / thesis_compare
五个模块，均已落地；新增图时在对应模块里 ``@REGISTRY.register``
即可被 GUI 菜单自动发现。
"""

from . import electrical, optical, scans, structure, thesis_compare  # noqa: F401

__all__ = ["electrical", "optical", "scans", "structure", "thesis_compare"]
