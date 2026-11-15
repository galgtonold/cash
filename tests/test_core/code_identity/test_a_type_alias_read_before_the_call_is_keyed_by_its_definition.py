"""A ``type X = ...`` alias is keyed by what defines it, read early or not.

An alias builds its ``__value__`` once, on its first read, and pickles by
name. A program that read the alias before the cached call (``ops.Solver.
__value__`` at import) left the call nothing to see being built: after an
edit to the alias the next process hit and served the value from before the
edit. Each case here runs a script twice per edit, in fresh processes, and
compares what it prints with the same script run uncached (plain Python).
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a `type` statement needs 3.12"),
]

#: Module source per case, before and after the edit.
CASES = {
    "literal": (
        'from typing import Literal, get_args\ntype Solver = Literal["arpack", "full"]\n'
        "def run():\n    return get_args(Solver.__value__)\n",
        'from typing import Literal, get_args\ntype Solver = Literal["arpack", "lobpcg"]\n'
        "def run():\n    return get_args(Solver.__value__)\n",
    ),
    "generic": (
        "type Solver[T] = list[T]\ndef run():\n    return repr(Solver[int].__value__), Solver.__type_params__[0].__name__\n",
        "type Solver[K] = dict[str, K]\n"
        "def run():\n    return repr(Solver[int].__value__), Solver.__type_params__[0].__name__\n",
    ),
    "global_it_reads": (
        'from typing import Literal, get_args\nMODES = Literal["a", "b"]\ntype Solver = MODES | None\n'
        "def run():\n    return repr(Solver.__value__)\n",
        'from typing import Literal, get_args\nMODES = Literal["a", "c"]\ntype Solver = MODES | None\n'
        "def run():\n    return repr(Solver.__value__)\n",
    ),
}

#: How the cached function reaches the alias.
MAINS = {
    # Through a helper of the user's package that reads it as a global.
    "helper": """
        @cash.cache
        def solve():
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return ops.run()
    """,
    # Read itself through the module (an attribute chain).
    "chain": """
        @cash.cache
        def solve():
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return repr(ops.Solver.__value__)
    """,
}

HEAD = """
import os
import sys
import cash
from aliaspkg import ops

if os.environ.get("EARLY") == "1":
    ops.Solver.__value__
"""

TAIL = """
for _ in range(3):
    print(solve())
"""


def _main(how: str) -> str:
    return HEAD + textwrap.dedent(MAINS[how]) + TAIL


def _run(tmp_path, source, how, early, *, plain=False):
    pkg = tmp_path / "aliaspkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops.py").write_text(source, encoding="utf-8")
    (tmp_path / "main.py").write_text(_main(how), encoding="utf-8")
    env = {"EARLY": "1" if early else "0"}
    if plain:
        env["CASH_DISABLE"] = "1"
    proc = run_python("main.py", cwd=tmp_path, env=env)
    return proc.stdout, proc.stderr.count("[RUN]")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("how", sorted(MAINS))
@pytest.mark.parametrize("early", [True, False], ids=["read_early", "not_read_early"])
def test_an_edited_alias_recomputes_and_an_unchanged_one_hits(tmp_path, case, how, early):
    before, after = CASES[case]
    for source, runs in ((before, 1), (before, 0), (after, 1), (after, 0)):
        want, _ = _run(tmp_path, source, how, early, plain=True)
        got, ran = _run(tmp_path, source, how, early)
        assert got == want, f"cash printed\n{got}\nplain Python printed\n{want}"
        assert ran == runs, f"expected {runs} run(s) of the body, saw {ran}"


def test_reading_the_alias_early_or_not_keys_alike(tmp_path):
    """The key is the alias's definition, whichever process built its value
    first: the run that read it early hits the entry the other stored."""
    before, _ = CASES["literal"]
    _, ran = _run(tmp_path, before, "helper", early=False)
    assert ran == 1
    _, ran = _run(tmp_path, before, "helper", early=True)
    assert ran == 0


def test_an_alias_replaced_in_the_process_recomputes(tmp_path):
    """The memoised hit paths of one process see a reload that rebinds the
    alias, read before the call or not."""
    before, after = CASES["literal"]
    pkg = tmp_path / "aliaspkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops.py").write_text(before, encoding="utf-8")
    script = textwrap.dedent(f"""
        import importlib
        import pathlib
        import sys
        import cash
        from aliaspkg import ops

        @cash.cache
        def solve():
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return ops.run()

        for early in (True, False):
            if early:
                ops.Solver.__value__
            for _ in range(3):
                print(solve() == ops.run())
            pathlib.Path(ops.__file__).write_text({after!r} if early else {before!r})
            importlib.reload(ops)
            if early:
                ops.Solver.__value__
            for _ in range(3):
                print(solve() == ops.run())
    """)
    (tmp_path / "main.py").write_text(script, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path)
    assert proc.stdout.split() == ["True"] * 12, proc.stdout + proc.stderr
