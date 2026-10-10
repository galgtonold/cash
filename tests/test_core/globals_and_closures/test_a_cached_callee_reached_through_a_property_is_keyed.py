"""A cached function that calls another one through a property or a
``__getattr__`` of a module-level object of the user's is keyed by it.

``api.inner(x)`` with ``inner`` a property returning ``impl.inner``, or
``api``'s class forwarding names to ``impl`` with ``__getattr__``: cash reads
attributes without running the user's code, so the edge to the cached
``impl.inner`` was lost and editing it kept serving the caller's old result.
The getter's body is now read instead of run. What cash still cannot tell
warns KEY-UNRESOLVED-CALL rather than staying silent.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

SHAPES = {
    "property": """
        import impl
        class Api:
            @property
            def inner(self):
                return impl.inner
        api = Api()
    """,
    "getattr": """
        import impl
        class Api:
            def __getattr__(self, name):
                return getattr(impl, name)
        api = Api()
    """,
    "cached_property": """
        import functools
        import impl
        class Api:
            @functools.cached_property
            def inner(self):
                return impl.inner
        api = Api()
    """,
    "nested": """
        import impl
        class Api:
            @property
            def ops(self):
                return impl
        api = Api()
    """,
    # The getter reads the function off an instance: nothing to resolve
    # statically, so every cached function named `inner` counts.
    "opaque_getter": """
        import impl
        class Holder:
            def __init__(self):
                self.fn = impl.inner
        class Api:
            def __init__(self):
                self._h = Holder()
            @property
            def inner(self):
                return self._h.fn
        api = Api()
    """,
}

MAIN = """
import cash
from helpers import api

@cash.cache
def outer(x):
    return api.{call}(x) + 1

print(outer(5))
"""


def _impl(k: int, cached: bool = True) -> str:
    deco = "@cash.cache\n" if cached else ""
    return f"import cash\n{deco}def inner(x):\n    return x * {k}\n"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_editing_a_cached_callee_reached_through_a_property_recomputes(tmp_path, shape):
    (tmp_path / "helpers.py").write_text(textwrap.dedent(SHAPES[shape]))
    (tmp_path / "main.py").write_text(MAIN.format(call="ops.inner" if shape == "nested" else "inner"))
    (tmp_path / "impl.py").write_text(_impl(2))
    first = run_python("main.py", cwd=tmp_path)
    assert first.stdout.strip() == "11"
    assert "KEY-UNRESOLVED-CALL" not in first.stderr
    (tmp_path / "impl.py").write_text(_impl(3))
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "16"


def test_a_plain_function_reached_through_a_property_warns(tmp_path):
    # The key follows a cached callee this way, not a plain one: say so.
    (tmp_path / "helpers.py").write_text(textwrap.dedent(SHAPES["property"]))
    (tmp_path / "main.py").write_text(MAIN.format(call="inner"))
    (tmp_path / "impl.py").write_text(_impl(2, cached=False))
    done = run_python("main.py", cwd=tmp_path)
    assert done.stdout.strip() == "11"
    assert "KEY-UNRESOLVED-CALL" in done.stderr


def test_a_library_property_does_not_warn(tmp_path):
    # Positive control for the warning: a property of library code (a path's
    # parent) reaches no code of the user's.
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""
            import pathlib
            import cash

            here = pathlib.PurePosixPath("/a/b/c")

            @cash.cache
            def outer(x):
                return here.parent.joinpath(x).name

            print(outer("d"))
        """)
    )
    done = run_python("main.py", cwd=tmp_path)
    assert done.stdout.strip() == "d"
    assert "KEY-UNRESOLVED-CALL" not in done.stderr
