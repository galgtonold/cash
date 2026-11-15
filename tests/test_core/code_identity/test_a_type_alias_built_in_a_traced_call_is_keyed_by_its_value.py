"""A type alias built in a traced call is keyed by what it stands for.

Reported against scanpy checked out as the user's code: ``type Solver =
Literal[...]`` builds its value on first read, so a traced call reading it
(``get_args(Solver.__value__)``) ran code no function name finds and was
refused. That code only builds the alias's value, so the value stands for
it: an edit to the alias is seen, an unchanged one hits.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a `type` statement and sys.monitoring need 3.12"),
]

OPS = textwrap.dedent("""
    import operator
    from typing import Literal, get_args

    type Solver = Literal[{solvers}]


    def run(xs, how):
        return [getattr(operator, how)(x, 2) for x in xs], get_args(Solver.__value__)
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from aliaspkg import ops


    @cash.cache
    def solve(xs, how):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return ops.run(xs, how)


    print(solve([1, 2], "mul"))
""")


def _run(tmp_path, solvers):
    pkg = tmp_path / "aliaspkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops.py").write_text(OPS.format(solvers=solvers), encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_an_unchanged_alias_hits_and_an_edited_one_recomputes(tmp_path):
    out, err = _run(tmp_path, '"arpack", "full"')
    assert out == "([2, 4], ('arpack', 'full'))" and "[RUN]" in err
    out, err = _run(tmp_path, '"arpack", "full"')
    assert "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, '"arpack", "lobpcg"')
    assert out == "([2, 4], ('arpack', 'lobpcg'))" and "[RUN]" in err, f"served the result from before the edit:\n{err}"
