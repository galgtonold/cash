"""An object read as ``module.OBJ`` is keyed by its class's code, however it is used.

``config.SETTINGS.rate`` (a property), ``config.SETTINGS + x``,
``config.SETTINGS["k"]``, ``len(config.SETTINGS)`` and
``config.SETTINGS.anything`` (``__getattr__``) run a method of
``SETTINGS``'s class that no name in the body points at. Only the class's
data was keyed, so editing that method served the old result.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

SHAPES = {
    "a property": ("    @property\n    def rate(self):\n        return 1\n", "x + config.SETTINGS.rate"),
    "an operator": ("    def __add__(self, other):\n        return other + 1\n", "config.SETTINGS + x"),
    "indexing": ("    def __getitem__(self, key):\n        return 1\n", "x + config.SETTINGS['k']"),
    "len()": ("    def __len__(self):\n        return 1\n", "x + len(config.SETTINGS)"),
    "__getattr__": ("    def __getattr__(self, name):\n        return 1\n", "x + config.SETTINGS.anything"),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_editing_the_method_used_recomputes(tmp_path, shape):
    member, expr = SHAPES[shape]
    main = f"import cash, config\n\n@cash.cache\ndef f(x):\n    return {expr}\n\nprint(f(3))\n"
    config = "class Settings:\n" + member + "\nSETTINGS = Settings()\n"
    first, after, uncached = edited_runs(
        tmp_path,
        {"main.py": main, "config.py": config},
        [("config.py", "1\n", "100\n")],
    )
    assert first == "4"
    assert after == uncached == "103"
