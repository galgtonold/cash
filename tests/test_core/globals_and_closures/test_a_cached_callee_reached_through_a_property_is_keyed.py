"""A cached function that calls another one through a property or a
``__getattr__`` of a module-level object of the user's is keyed by it.

``api.inner(x)`` with ``inner`` a property returning ``impl.inner``, or
``api``'s class forwarding names to ``impl`` with ``__getattr__``: cash reads
attributes without running the user's code, so the edge to the cached
``impl.inner`` was lost and editing it kept serving the caller's old result.
The getter's body is now read instead of run. A plain function reached this way is served on a trace of what the call
ran on 3.12+; before that the call runs uncached with KEY-UNRESOLVED-CALL,
unless ``depends_on`` names it.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

TRACED = sys.version_info >= (3, 12)

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
    (tmp_path / "helpers.py").write_text(textwrap.dedent(SHAPES[shape]), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN.format(call="ops.inner" if shape == "nested" else "inner"), encoding="utf-8")
    (tmp_path / "impl.py").write_text(_impl(2), encoding="utf-8")
    first = run_python("main.py", cwd=tmp_path)
    assert first.stdout.strip() == "11"
    assert "KEY-UNRESOLVED-CALL" not in first.stderr
    (tmp_path / "impl.py").write_text(_impl(3), encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "16"


MAIN_COUNTED = """
import cash
from helpers import api

@cash.cache
def outer(x):
    print("ran")
    return api.inner(x) + 1

print(outer(5))
"""


def _plain_through_a_property(tmp_path, main=MAIN_COUNTED):
    (tmp_path / "helpers.py").write_text(textwrap.dedent(SHAPES["property"]), encoding="utf-8")
    (tmp_path / "main.py").write_text(main, encoding="utf-8")
    (tmp_path / "impl.py").write_text(_impl(2, cached=False), encoding="utf-8")


@pytest.mark.skipif(not TRACED, reason="sys.monitoring is Python 3.12+")
def test_a_plain_function_reached_through_a_property_is_served_on_its_trace(tmp_path):
    # The key does not name the plain function the property hands out; the
    # trace of what the call ran does, so it is served until that is edited.
    _plain_through_a_property(tmp_path)
    first = run_python("main.py", cwd=tmp_path)
    assert first.stdout.split() == ["ran", "11"]
    assert "KEY-UNRESOLVED-CALL" not in first.stderr
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["11"]
    (tmp_path / "impl.py").write_text(_impl(3, cached=False), encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["ran", "16"]


@pytest.mark.skipif(TRACED, reason="on 3.12+ the call is served on its trace")
def test_before_3_12_a_plain_function_reached_through_a_property_runs_uncached(tmp_path):
    _plain_through_a_property(tmp_path)
    first = run_python("main.py", cwd=tmp_path)
    assert first.stdout.split() == ["ran", "11"]
    assert "KEY-UNRESOLVED-CALL" in first.stderr
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["ran", "11"]


def test_naming_it_in_depends_on_keys_it(tmp_path):
    _plain_through_a_property(
        tmp_path,
        MAIN_COUNTED.replace("import cash\n", "import cash, impl\n").replace(
            "@cash.cache\n", "@cash.cache(depends_on=[impl.inner])\n"
        ),
    )
    first = run_python("main.py", cwd=tmp_path)
    assert first.stdout.split() == ["ran", "11"]
    assert "KEY-UNRESOLVED-CALL" not in first.stderr
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["11"]
    (tmp_path / "impl.py").write_text(_impl(3, cached=False), encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["ran", "16"]


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
        """),
        encoding="utf-8",
    )
    done = run_python("main.py", cwd=tmp_path)
    assert done.stdout.strip() == "d"
    assert "KEY-UNRESOLVED-CALL" not in done.stderr
