"""A function a call imports inside a body, which a decorator wrapped around
a ``@cash.cache`` function (scanpy's ``@_accepts_legacy_random_state`` over
a cached ``pca``), is checked by what it stands for: each layer's code and
bindings and the cached function's own state. Its value by itself held the
cached wrapper, which cannot be digested, so a call that picks code by a
run-time name and reached it ran uncached every time. An edit to the cached
function or to the layer around it is still seen."""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a call that picks code by name is traced on 3.12+"),
    pytest.mark.timeout(300),
]

A = """
import functools
import cash


def accepts_legacy(pos):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs) + pos
        return wrapper
    return decorator


@accepts_legacy(0)
@cash.cache
def pca(x):
    return x * 2
"""

HELPERS = """
def get_pca(x):
    from .a import pca
    return pca(x) + 1
"""

B = """
import sys
import cash
from . import helpers


@cash.cache
def neighbors(x, how="pca"):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return getattr(helpers, "get_" + how)(x)
"""

MAIN = """
from wrappkg.b import neighbors
print(neighbors(3), neighbors(3))
"""


def _write(root, a=A):
    pkg = root / "wrappkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(textwrap.dedent(a), encoding="utf-8")
    (pkg / "helpers.py").write_text(textwrap.dedent(HELPERS), encoding="utf-8")
    (pkg / "b.py").write_text(textwrap.dedent(B), encoding="utf-8")
    (root / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")


def _run(root):
    out = run_python("main.py", cwd=root)
    return out.stdout.split(), out.stderr.count("[RUN]"), out.stderr


def test_the_call_is_stored_and_served_in_the_next_process(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path)[:2] == (["7", "7"], 1)
    got, runs, err = _run(tmp_path)
    assert (got, runs) == (["7", "7"], 0), err
    assert "cannot be checked" not in err


@pytest.mark.parametrize(
    ("edit", "now"),
    [
        (("return x * 2", "return x * 5"), "16"),
        (("return f(*args, **kwargs) + pos", "return f(*args, **kwargs) + pos + 10"), "17"),
        (("@accepts_legacy(0)", "@accepts_legacy(3)"), "10"),
    ],
    ids=["cached-function", "wrapping-layer", "layer-binding"],
)
def test_an_edit_to_what_it_stands_for_is_recomputed(tmp_path, edit, now):
    _write(tmp_path)
    _run(tmp_path)
    _write(tmp_path, A.replace(*edit))
    got, runs, err = _run(tmp_path)
    assert got == [now, now], err
    assert runs == 1, err
