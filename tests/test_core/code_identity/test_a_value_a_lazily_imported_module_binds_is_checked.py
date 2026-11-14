"""A value a lazily imported module computed at its top level is not served stale.

A module first imported inside a call (``from .helpmod import K`` in a method
of an argument, or in the cached function's body) may compute what it binds
from outside its source: the environment, a file, another module's value.
Its file is then the same while the value it hands the call is not, so a run
with the environment changed must not be served the result of the old one.

The method of an argument (the carrier path) is served on a trace on Python
3.12 and newer, which checks the value each such name holds when the call
looks it up again; the body path keys the call once the module is loaded.
A module whose top level binds only constants still hits.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(600)]

TRACED = sys.version_info >= (3, 12)

HELPMOD = {
    "env": 'import os\n\nK = int(os.environ.get("LZ_K", "1"))\n',
    "other_module": "from .conf import BASE\n\nK = BASE * 2\n",
    "file": 'from pathlib import Path\n\nK = int(Path(__file__).with_name("k.txt").read_text())\n',
    "constant": "K = 6\n",
}

# What K is for a setting *v* of the environment, the other module or the file.
K_OF = {"env": lambda v: v, "other_module": lambda v: 2 * v, "file": lambda v: v, "constant": lambda v: 6}

USE = {
    "from_import": ("from {pkg}.helpmod import K", "K"),
    "attribute": ("from {pkg} import helpmod", "helpmod.K"),
}

BASE = textwrap.dedent("""
    class S:
        def total(self, xs):
            {imp}

            return sum(xs) * {val}
""")

MAIN = textwrap.dedent("""
    import os
    import sys

    import cash
    from lzpkg import conf

    conf.BASE = int(os.environ.get("LZ_BASE", "1"))
    from lzpkg.base import S


    @cash.cache
    def carrier(est, xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return est.total(xs)


    @cash.cache
    def body(xs):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        {imp}

        return sum(xs) * {val}


    for _ in range(2):
        print(carrier(S(), [1, 2]) if sys.argv[1] == "carrier" else body([1, 2]))
""")


def _write(tmp_path, source, use):
    imp, val = USE[use]
    pkg = tmp_path / "lzpkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "conf.py").write_text("BASE = 1\n", encoding="utf-8")
    (pkg / "helpmod.py").write_text(HELPMOD[source], encoding="utf-8")
    (pkg / "base.py").write_text(
        BASE.format(imp=imp.format(pkg="").replace("from  ", "from . "), val=val), encoding="utf-8"
    )
    main = MAIN.format(imp=imp.format(pkg="lzpkg"), val=val)
    (tmp_path / "main.py").write_text(main, encoding="utf-8")


def _run(tmp_path, path, setting, trace):
    (tmp_path / "lzpkg" / "k.txt").write_text(str(setting), encoding="utf-8")
    env = {"CASH_TRACE": trace, "CASH_SUMMARY": "1", "LZ_K": str(setting), "LZ_BASE": str(setting)}
    proc = run_python("main.py", path, cwd=tmp_path, env=env)
    return proc.stdout.split(), proc.stderr


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("use", list(USE))
@pytest.mark.parametrize("source", ["env", "other_module", "file"])
@pytest.mark.parametrize("path", ["carrier", "body"])
def test_a_value_bound_from_outside_the_source_recomputes(tmp_path, path, source, use, trace):
    _write(tmp_path, source, use)
    last = None
    for setting in (4, 4, 24, 24, 4):
        want = str(3 * K_OF[source](setting))
        out, err = _run(tmp_path, path, setting, trace)
        assert out == [want, want], f"setting {setting}: served a value bound before the change:\n{err}"
        if setting == last and (path == "body" or (trace == "1" and TRACED)):
            # Unchanged, it is served: on the trace, or once the body's
            # import has loaded the module.
            assert err.count("[RUN]") <= (1 if path == "body" else 0), err
        last = setting


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("use", list(USE))
@pytest.mark.parametrize("path", ["carrier", "body"])
def test_a_module_of_constants_still_hits(tmp_path, path, use, trace):
    _write(tmp_path, "constant", use)
    out, err = _run(tmp_path, path, 4, trace)
    assert out == ["18", "18"]
    out, err = _run(tmp_path, path, 4, trace)
    assert out == ["18", "18"]
    assert "[RUN]" not in err, f"an unedited run must hit:\n{err}"
