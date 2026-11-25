"""`settings.WIDTH` read in a class body reaches the key.

A class body loads a module global by ``LOAD_NAME`` (``LOAD_FROM_DICT_OR_GLOBALS``
in a generic class), not ``LOAD_GLOBAL``, so the scan for ``module.ATTR`` reads
missed ``class Row: width = settings.WIDTH`` written inside a cached function
or a helper: a local dataclass or enum whose defaults come from a settings
module. Editing the constant then served the result computed with the old one.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

IN_THE_CACHED_FUNCTION = """
    import cash
    import settings

    @cash.cache
    def build(n):
        class Row:
            width = settings.WIDTH
        return Row.width * n

    print(build(1))
"""

IN_A_HELPER = """
    import cash
    import helper

    @cash.cache
    def build(n):
        return helper.make() * n

    print(build(1))
"""

HELPER = """
    import settings

    def make():
        class Row:
            width = settings.WIDTH
        return Row.width
"""

IN_A_GENERIC_CLASS = """
    import cash
    import settings

    @cash.cache
    def build(n):
        class Box[T]:
            width = settings.WIDTH
        return Box.width * n

    print(build(1))
"""

# Control: the same read outside a class body, keyed all along.
PLAIN_READ = """
    import cash
    import settings

    @cash.cache
    def build(n):
        return settings.WIDTH * n

    print(build(1))
"""

CASES = [
    pytest.param(IN_THE_CACHED_FUNCTION, id="class body in the cached function"),
    pytest.param(IN_A_HELPER, id="class body in a helper"),
    pytest.param(
        IN_A_GENERIC_CLASS,
        id="body of a generic class",
        marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="class Box[T] is Python 3.12 syntax"),
    ),
    pytest.param(PLAIN_READ, id="outside a class body"),
]


@pytest.mark.parametrize("main", CASES)
def test_editing_the_constant_between_runs_recomputes(tmp_path, main):
    (tmp_path / "main.py").write_text(textwrap.dedent(main), encoding="utf-8")
    (tmp_path / "helper.py").write_text(textwrap.dedent(HELPER), encoding="utf-8")
    (tmp_path / "settings.py").write_text("WIDTH = 1\n", encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "1"
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "1"
    (tmp_path / "settings.py").write_text("WIDTH = 10  # edited\n", encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == "10"


def test_rebinding_the_constant_in_the_process_recomputes(tmp_path):
    """Repeated calls first, so every layer that remembers a key is warm."""
    main = """
        import cash
        import settings

        @cash.cache
        def build(n):
            class Row:
                width = settings.WIDTH
            return Row.width * n

        seen = [build(1) for _ in range(5)]
        settings.WIDTH = 10
        seen += [build(1) for _ in range(3)]
        settings.WIDTH = 1
        seen += [build(1) for _ in range(3)]
        print(seen)
    """
    (tmp_path / "main.py").write_text(textwrap.dedent(main), encoding="utf-8")
    (tmp_path / "settings.py").write_text("WIDTH = 1\n", encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip() == str([1] * 5 + [10] * 3 + [1] * 3)


def test_a_write_in_a_class_body_is_the_calls_output_not_an_input():
    """`settings.SEEN = ...` in a class body is a write, like the same line
    in the function: what the call writes is not folded as what it read."""
    from cash.decorator.module_attrs import written_module_attrs

    def build():
        class Row:
            settings.SEEN = 1  # noqa: F821 - only compiled, never run

        return Row

    assert ("settings", "SEEN") in written_module_attrs(build.__code__)
