"""A numba-compiled helper is code: the function it compiles and its options.

``@njit`` returns a dispatcher object, not a function. Read as a data global
it was hashed attribute by attribute -- type contexts, overload tables, the
on-disk cache's weak dictionaries -- so a cached function calling ``kern(x)``
after ``from helpers import kern`` raised KEY-UNHASHABLE-DEFAULT and never
cached, and one calling ``helpers.kern(x)`` cached but spent seconds per
miss walking numba's own source. Passed as an argument it could not be
hashed at all.

Each run is a fresh interpreter over one cache folder, so what a later
process is served is what the test sees.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba is not installed")


HELPERS = """
import numba

FAST = False

@numba.njit
def square(x):
    return x * x

@numba.njit(fastmath=FAST)
def kern(a):
    s = 0.0
    for x in a:
        s += square(x)
    return s
"""

BY_NAME = """
import ast
import linecache
import os
import time
import numba
import numpy as np
import cash
from helpers import kern

# what the first call reads: numba's own source files, and sources parsed
NUMBA = os.path.dirname(numba.__file__)
numba_files = set()
# kept on the ast module: a global of this script the cached call reads, as
# the patched parse below does, is checked before each hit
ast.parses = 0
_getlines, _parse = linecache.getlines, ast.parse

def getlines(filename, *args, **kwargs):
    if filename.startswith(NUMBA):
        numba_files.add(filename)
    return _getlines(filename, *args, **kwargs)

def parse(*args, **kwargs):
    ast.parses += 1
    return _parse(*args, **kwargs)

linecache.getlines, ast.parse = getlines, parse

@cash.cache
def backtest(n):
    print("RAN", flush=True)
    return kern(np.arange(n, dtype=np.float64))

t0 = time.perf_counter()
print("RESULT", backtest(4), flush=True)
print("FIRST_CALL_S", time.perf_counter() - t0, len(numba_files), ast.parses, flush=True)
"""

BY_MODULE = """
import numpy as np
import cash
import helpers

@cash.cache
def backtest(n):
    print("RAN", flush=True)
    return helpers.kern(np.arange(n, dtype=np.float64))

print("RESULT", backtest(4), flush=True)
"""

AS_ARGUMENT = """
import numpy as np
import cash
from helpers import kern

@cash.cache
def apply(fn, n):
    print("RAN", flush=True)
    return fn(np.arange(n, dtype=np.float64))

print("RESULT", apply(kern, 4), flush=True)
"""

IN_A_REGISTRY = """
import numpy as np
import cash
from helpers import kern

KERNELS = {"sum": kern}

@cash.cache
def run(name, n):
    print("RAN", flush=True)
    return KERNELS[name](np.arange(n, dtype=np.float64))

print("RESULT", run("sum", 4), flush=True)
"""


def _project(tmp_path: Path, main: str) -> Path:
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    (project / "helpers.py").write_text(textwrap.dedent(HELPERS).lstrip(), encoding="utf-8")
    (project / "main.py").write_text(textwrap.dedent(main).lstrip(), encoding="utf-8")
    return project


def _edit(project: Path, old: str, new: str) -> None:
    path = project / "helpers.py"
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def _run(project: Path) -> tuple[bool, str, str]:
    out = run_python("main.py", cwd=project, cache_dir=project / ".cash")
    result = next(line.split()[1] for line in out.stdout.splitlines() if line.startswith("RESULT"))
    return "RAN" in out.stdout, result, out.stdout + out.stderr


def _assert_caches(project: Path, expected: str) -> str:
    ran, first, output = _run(project)
    assert ran and first == expected
    assert "KEY-" not in output, output
    ran, again, output = _run(project)
    assert not ran, f"the second process ran again instead of hitting:\n{output}"
    assert again == expected
    return output


# 0 + 1 + 4 + 9
SUM_OF_SQUARES = "14.0"


@needs_numba
@pytest.mark.timeout(120)
def test_a_jitted_function_imported_by_name_caches(tmp_path):
    output = _assert_caches(_project(tmp_path, BY_NAME), SUM_OF_SQUARES)
    seconds, numba_files, parses = next(
        line.split()[1:] for line in output.splitlines() if line.startswith("FIRST_CALL_S")
    )
    # The whole first call of the process -- analysis, key build and the
    # lookup -- once took about 8 seconds walking numba's own source. Checked
    # by the work it does: no numba source read, a couple of dozen parses
    # (24 today). The time is a loose second check: 0.2 s here, but 1.2 s on a
    # loaded Windows runner.
    assert int(numba_files) == 0, f"read {numba_files} of numba's source files"
    assert int(parses) < 100, f"parsed {parses} sources"
    assert float(seconds) < 4.0, seconds


@needs_numba
@pytest.mark.timeout(120)
def test_a_jitted_function_called_through_its_module_caches(tmp_path):
    _assert_caches(_project(tmp_path, BY_MODULE), SUM_OF_SQUARES)


@needs_numba
@pytest.mark.timeout(120)
def test_a_jitted_function_passed_as_an_argument_caches(tmp_path):
    project = _project(tmp_path, AS_ARGUMENT)
    _assert_caches(project, SUM_OF_SQUARES)
    _edit(project, "s += square(x)", "s += square(x) + 1")
    ran, result, _ = _run(project)
    assert ran and result == "18.0"


@needs_numba
@pytest.mark.timeout(120)
def test_a_jitted_function_in_a_registry_caches_and_follows_edits(tmp_path):
    project = _project(tmp_path, IN_A_REGISTRY)
    _assert_caches(project, SUM_OF_SQUARES)
    _edit(project, "return x * x", "return x * x * x")
    ran, result, _ = _run(project)
    assert ran and result == "36.0"


@needs_numba
@pytest.mark.timeout(120)
def test_editing_the_jitted_function_invalidates(tmp_path):
    project = _project(tmp_path, BY_NAME)
    _assert_caches(project, SUM_OF_SQUARES)
    _edit(project, "s += square(x)", "s += square(x) + 1")
    ran, result, _ = _run(project)
    assert ran and result == "18.0"


@needs_numba
@pytest.mark.timeout(120)
def test_editing_a_helper_the_jitted_function_calls_invalidates(tmp_path):
    project = _project(tmp_path, BY_MODULE)
    _assert_caches(project, SUM_OF_SQUARES)
    _edit(project, "return x * x", "return x * x * x")
    ran, result, _ = _run(project)
    assert ran and result == "36.0"


@needs_numba
@pytest.mark.timeout(120)
def test_changing_a_compile_option_invalidates(tmp_path):
    """``fastmath=FAST``: the decorator's text is the same, the machine code is not."""
    project = _project(tmp_path, BY_NAME)
    _assert_caches(project, SUM_OF_SQUARES)
    _edit(project, "FAST = False", "FAST = True")
    ran, result, _ = _run(project)
    assert ran and result == SUM_OF_SQUARES


class _FakeDispatcher:
    """The shape of a numba dispatcher, with no numba: ``py_func`` beside
    ``targetoptions``."""

    def __init__(self, fn, **options):
        self.py_func = fn
        self.targetoptions = options
        self.overloads = {}

    def __call__(self, *args):
        return self.py_func(*args)


def _compiled(x):
    return x + 1


def test_a_dispatcher_is_recognised_by_its_shape():
    from cash.jit_functions import jit_python_function

    fake = _FakeDispatcher(_compiled, fastmath=True, cache=True)
    assert jit_python_function(fake) is _compiled
    assert jit_python_function(_compiled) is None
    assert jit_python_function(len) is None


def test_the_cache_option_does_not_move_the_options():
    """Where numba keeps the compiled code does not change what it computes."""
    from cash.jit_functions import jit_options

    assert jit_options(_FakeDispatcher(_compiled, fastmath=True, cache=True)) == jit_options(
        _FakeDispatcher(_compiled, fastmath=True)
    )
    assert jit_options(_FakeDispatcher(_compiled, fastmath=True)) != jit_options(
        _FakeDispatcher(_compiled, fastmath=False)
    )
