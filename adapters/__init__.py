"""数据适配器包。

这里统一导出 `load_dataset` 和 `DatasetBundle`，算法侧通常只需要从
`adapters` 导入这两个对象即可。
"""

from .load_dataset import DatasetBundle, load_dataset

__all__ = ["DatasetBundle", "load_dataset"]
