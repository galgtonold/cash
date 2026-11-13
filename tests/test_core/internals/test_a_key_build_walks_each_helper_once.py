"""One key build walks each helper once, and keys the call as before.

A bound method whose ``self`` is an estimator, holding a dataset, keys the
code of both classes and what every one of their functions reads. Each
function's purity report was a walk of its own through all the helpers
below it, and nothing was shared between those walks: a NiMARE ``_fit``
read, parsed and audited the same few hundred helpers thousands of times on
its first call. Inside one key build (`helper_walk.shared_nodes`), a helper
walked again makes the changes its first walk made instead.

Pinned here: the reports, and the keys, are the ones walking every helper
again gives (the old path is the build with sharing turned off); an edit to
a helper reached only through a cycle still moves the key; and the build
walks far fewer helpers.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import types

import pytest

from cash.analysis import helper_walk
from cash.analysis.helper_walk import HelperWalk, shared_nodes
from tests._scripts import run_python

pytestmark = pytest.mark.core

# --- a NiMARE-shaped package: an estimator over a dataset, shared helpers ---

_HELPERS = 40


def _helpers_source(edit: int | None = None) -> str:
    """Helpers ``h0..h39`` calling three below each (shared subtrees), some
    calling back up (cycles); *edit* changes one constant of that helper."""
    lines = ["import math", "SCALE = 3", "TABLE = {'a': 1, 'b': 2}", ""]
    for i in range(_HELPERS):
        offset = 1000 if i == edit else i
        lines.append(f"def h{i}(x):")
        lines.append(f"    y = x * SCALE + {offset}")
        for c in (i + 1, i + 3, i + 7):
            if c < _HELPERS:
                lines.append(f"    y = y + (h{c}(x - 1) if x > 0 else 0)")
        if i % 9 == 8:
            # Back up the tree: h{i-6} reached again through a cycle.
            lines.append(f"    y = y + (h{i - 6}(x - 50) if x > 100 else 0)")
        if i % 5 == 0:
            lines.append("    y = y + TABLE['a'] + math.floor(1.5)")
        lines.append("    return y")
        lines.append("")
    # Reached ONLY through the cycle h8 -> back_edge -> h2 -> ... -> h8.
    lines += ["def back_edge(x):", f"    return h2(x - 60) + {1000 if edit == -1 else 1}", ""]
    lines[lines.index("def h8(x):") + 1] += "\n    y = y + (back_edge(x - 70) if x > 200 else 0)"
    return "\n".join(lines) + "\n"


_DATASET = """
from . import helpers as H


class Dataset:
    def __init__(self, n):
        self.n = n

{methods}
"""

_ESTIMATOR = """
import cash
from . import helpers as H

c = cash.Cash()


class Base:
    def __init__(self, dataset):
        self.dataset = dataset
        self.alpha = 0.05

{base}

class Middle(Base):
{middle}

class Estimator(Middle):
{top}
    @c.cache
    def _fit(self, k):
        return self.m0_0(k) + self.m2_3(k) + self.dataset.get_1(k)
"""


def _methods(prefix: str, count: int, step: int) -> str:
    return "\n".join(
        f"    def {prefix}_{m}(self, x=1):\n"
        f"        return H.h{(m * step) % _HELPERS}(x) + H.h{(m * step + 11) % _HELPERS}(x)\n"
        for m in range(count)
    )


def _write_package(root, edit: int | None = None) -> None:
    pkg = root / "nimish"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "helpers.py").write_text(_helpers_source(edit))
    (pkg / "dataset.py").write_text(_DATASET.format(methods=_methods("get", 12, 5)))
    (pkg / "estimator.py").write_text(
        _ESTIMATOR.format(base=_methods("m0", 8, 3), middle=_methods("m1", 8, 7), top=_methods("m2", 8, 13))
    )


#: Prints the key of ``_fit``, how many callables its build walked, then
#: what a call returns and whether it was a hit, with the helpers shared
#: within the build or (``old``) every walk on its own.
_KEY_SCRIPT = """
import contextlib, sys
from cash.analysis import helper_walk
from cash.decorator import runtime
if sys.argv[1] == "old":
    runtime.shared_nodes = contextlib.nullcontext
walked = [0]
real = helper_walk.HelperWalk._walk_one
def counting(self, *args):
    walked[0] += 1
    return real(self, *args)
helper_walk.HelperWalk._walk_one = counting
from nimish.estimator import Estimator
from nimish.dataset import Dataset
est = Estimator(Dataset(5))
key = Estimator._fit.explain(est, 2).cache_key
walks = walked[0]
result = est._fit(2)
print(key, walks, result, Estimator._fit.cache_info()["hits"])
"""


def _run(tmp_path, mode: str, cache: str = "cache", trace: str = "1") -> tuple[str, int, int, int]:
    """(key, callables walked, result, hits) of ``_fit`` in a fresh process."""
    out = run_python(
        "-W", "ignore", "-c", _KEY_SCRIPT, mode, cwd=tmp_path, cache_dir=tmp_path / cache, env={"CASH_TRACE": trace}
    )
    key, walks, result, hits = out.stdout.split()[-4:]
    return key, int(walks), int(result), int(hits)


@pytest.mark.parametrize("trace", ["1", "0"])
def test_the_key_is_the_one_walking_every_helper_again_gives(tmp_path, trace):
    _write_package(tmp_path)
    old_key, old_walks, old_result, _ = _run(tmp_path, "old", "old", trace=trace)
    new_key, new_walks, new_result, _ = _run(tmp_path, "new", "new", trace=trace)
    assert new_key == old_key
    assert new_result == old_result
    assert new_walks * 5 <= old_walks, f"walked {new_walks} callables, against {old_walks} unshared"


# A helper reached only through a method of ``self`` (``H.h3`` in
# ``Base.m0_1``): with a trace (3.12+) the call's trace names it; without one
# (before 3.12, ``CASH_TRACE=0``) the methods' helpers are in the key.
@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("edit", [3, 17, -1], ids=["shared-helper", "deep-helper", "only-through-a-cycle"])
def test_an_edit_to_a_walked_helper_misses(tmp_path, edit, trace):
    _write_package(tmp_path)
    _, _, before, _ = _run(tmp_path, "new", trace=trace)
    assert _run(tmp_path, "new", trace=trace)[3] == 1, "an unchanged estimator was not served from the cache"
    _write_package(tmp_path, edit=edit)
    fresh = _run(tmp_path, "old", "fresh", trace=trace)[2]
    key, _, after, hits = _run(tmp_path, "new", trace=trace)
    assert hits == 0, f"an edit to a helper was served the result from before it ({before})"
    assert after == fresh
    assert key == _run(tmp_path, "old", "fresh", trace=trace)[0]


# --- the reports, shape by shape ---

_CORPUS = """
import dataclasses
import functools
import os
import time

import cash


def now():
    return time.time()


def stamp(x):
    return x, now()


def by_value(x):
    fn = now
    return fn(), x


def leaf(x):
    return x + int(os.environ.get("CORPUS_MODE", "0"))


def ping(n):
    return pong(n - 1) if n else leaf(n)


def pong(n):
    return ping(n - 1) if n else 0


@cash.pure
def marked(x):
    return leaf(x) + ping(x)


@cash.stateful
def noisy(x):
    print(x)
    return leaf(x)


def uses_markers(x):
    return marked(x) + noisy(x) + leaf(x)


exec_built = {}
exec("def built(x):\\n    return x * 2", exec_built)
built = exec_built["built"]


def calls_built(x):
    return built(x) + ping(x)


@cash.Cash().cache
def cached_callee(x):
    return leaf(x)


def calls_cached(x):
    return cached_callee(x) + pong(x)


@dataclasses.dataclass
class Point:
    x: int
    y: int = dataclasses.field(default_factory=lambda: leaf(0))

    def norm(self):
        return ping(self.x) + self.y

    @classmethod
    def origin(cls):
        return cls(0, 0)


POINT = Point(1, 2)
STEPS = [lambda x: leaf(x) + 1, lambda x: ping(x)]


def uses_point(x):
    p = Point.origin()
    return p.norm() + POINT.norm() + STEPS[0](x)


def make(k):
    def scaled(x):
        return leaf(x) * k
    return scaled


SCALED = make(3)


def uses_closure(x):
    return SCALED(x) + stamp(x)[0]


def waived(x):
    print(x)  # @cash:assume-safe
    return leaf(x) + uses_point(x)


@functools.lru_cache
def memo(x):
    return leaf(x)


def uses_memo(x):
    return memo(x) + calls_built(x)


def late(x):
    import corpus_never_loaded_xyz
    return corpus_never_loaded_xyz.f(x) + leaf(x)


def everything(x):
    return (
        uses_markers(x) + calls_built(x) + calls_cached(x) + uses_point(x)
        + uses_closure(x) + waived(x) + uses_memo(x) + by_value(x)[1] + late(x)
    )
"""

_ROOTS = (
    "now",
    "stamp",
    "by_value",
    "ping",
    "uses_markers",
    "marked",
    "calls_built",
    "calls_cached",
    "uses_point",
    "uses_closure",
    "waived",
    "uses_memo",
    "late",
    "everything",
)


def _plain(value):
    """*value* with each reference a report holds replaced by its object."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_plain(v) for v in value)
    if isinstance(value, (frozenset, set)):
        return value
    if callable(value) and not isinstance(value, (type, types.FunctionType, types.CodeType)):
        try:
            return ("ref", _plain_target(value()))
        except TypeError:
            return value
    return value


def _plain_target(obj):
    if type(obj) is types.MethodType:
        return ("method", obj.__func__, id(obj.__self__))
    return obj


def _report(fn):
    report = HelperWalk(fn).run()
    return {f: _plain(getattr(report, f)) for f in report.__dataclass_fields__}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    (tmp_path / "walk_once_corpus.py").write_text(textwrap.dedent(_CORPUS))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CORPUS_MODE", "1")
    module = importlib.import_module("walk_once_corpus")
    yield module
    sys.modules.pop("walk_once_corpus", None)


@pytest.mark.parametrize("order", ["roots-first", "callers-first"])
def test_every_report_is_the_one_its_own_walk_gives(corpus, order):
    roots = [getattr(corpus, name) for name in _ROOTS]
    if order == "callers-first":
        roots.reverse()
    alone = [_report(fn) for fn in roots]
    walked = []
    real = HelperWalk._walk_one

    def counting(self, *args):
        walked.append(1)  # not the callable: a bound method must die with its walk
        return real(self, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HelperWalk, "_walk_one", counting)
        with shared_nodes():
            shared = [_report(fn) for fn in roots]
    for fn, a, s in zip(roots, alone, shared):
        assert s == a, f"the report of {fn.__name__} differs when its helpers are shared"
    assert any(r["issues"] for r in alone), "the corpus reports no findings: it pins nothing"
    assert len(walked) < sum(len(r["helper_source_hashes"]) for r in alone)


def test_sharing_ends_with_the_build(corpus):
    with shared_nodes():
        with shared_nodes():  # a nested build shares the outer one's
            assert helper_walk._SHARED_NODES.get() is not None
            HelperWalk(corpus.ping).run()
        assert helper_walk._SHARED_NODES.get()
    assert helper_walk._SHARED_NODES.get() is None


def test_outside_a_build_nothing_is_shared(corpus):
    walked = []
    real = HelperWalk._walk_one

    def counting(self, *args):
        walked.append(1)
        return real(self, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HelperWalk, "_walk_one", counting)
        HelperWalk(corpus.ping).run()
        first = len(walked)
        HelperWalk(corpus.ping).run()
    assert len(walked) == 2 * first
