"""A class whose metaclass answers every attribute can be read by a cached function.

``class DefaultsMeta(type): def __getattr__(cls, name): return 1`` answers
``Defaults.__code__`` too. Looking for the code a call ran, cash took that ``1``
for a code object and used it as a weak key: the call crashed with
``TypeError: cannot create weak reference to 'int' object`` after the body ran.
"""

from __future__ import annotations

from tests.test_core.code_identity._edited_project import edited_runs

MAIN = """
    import cash
    from settings import Defaults

    @cash.cache
    def f(x):
        return x + Defaults.retries

    print(f(3))
    print(f(3))
"""

SETTINGS = """
    class DefaultsMeta(type):
        def __getattr__(cls, name):
            return 1

    class Defaults(metaclass=DefaultsMeta):
        pass
"""


def test_the_call_returns_its_value(tmp_path):
    first, again, uncached = edited_runs(tmp_path, {"main.py": MAIN, "settings.py": SETTINGS})
    assert first == "4\n4"
    assert again == uncached == "4\n4"
