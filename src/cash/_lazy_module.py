"""A module imported on first use rather than at ``import cash``.

``psutil`` costs about 11 ms to import and is read only when a memory cap or a
process start time is actually needed. A module-level ``psutil = LazyModule("psutil")``
keeps the name a module attribute -- so ``monkeypatch.setattr(mod, "psutil", fake)``
and ``mock.patch("psutil.virtual_memory")`` both still reach the code -- while
the import waits for the first attribute read.
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stands in for module *name*; imports it on the first attribute read.

    Attributes are read from and written to the real module, never kept
    here, so a patch applied either way is seen by every user.
    """

    def __init__(self, name: str) -> None:
        self.__dict__["_lazy_name"] = name
        self.__dict__["_lazy_module"] = None

    def _load(self) -> Any:
        module = self.__dict__["_lazy_module"]
        if module is None:
            module = importlib.import_module(self.__dict__["_lazy_name"])
            self.__dict__["_lazy_module"] = module
        return module

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    # Writes go to the real module too, as they did when the name held it:
    # ``monkeypatch.setattr(mod.psutil, "virtual_memory", fake)`` patches psutil.
    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._load(), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._load(), attr)

    def __repr__(self) -> str:
        return f"<lazy module {self.__dict__['_lazy_name']!r}>"
