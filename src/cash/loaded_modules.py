"""Ask whether a library is loaded, without importing it (`loaded_module`).

cash never imports numpy, pandas or any other library for the user: it
looks whether the program has, and only then knows about the library's
values. ``sys.modules.get("numpy")`` is not that question. An import puts
the module into ``sys.modules`` before it runs the module's code, so that
the module can be imported by what it imports; while one thread is inside
``import numpy``, every other thread finds a module there that has no
``ndarray`` yet, and a cached call on that thread failed with
AttributeError from inside cash, about a library its function never
touched.

A library whose import has not finished is not loaded: no value of it can
have reached a cached function yet. Every look for a library by name goes
through `loaded_module`; `tests/test_tooling` holds the package to it.
"""

from __future__ import annotations

import sys
from types import ModuleType

__all__ = ["loaded_module"]


def loaded_module(name: str) -> ModuleType | None:
    """The module *name* when the program has imported it and that import
    has finished; None when it is absent or still being imported (by this
    thread or another)."""
    module = sys.modules.get(name)
    if module is None:
        return None
    # The import system sets this on the module's spec for as long as the
    # module's code runs, and it is what makes ``from numpy import x`` say
    # "partially initialized module". `is True`: a stand-in put into
    # sys.modules (a mock) answers any attribute with something truthy.
    if getattr(getattr(module, "__spec__", None), "_initializing", False) is True:
        return None
    return module
