"""A ``numpy.vectorize`` a global holds keys alike before and after its
first call.

``np.vectorize`` builds a ufunc from its function on its first call and
keeps it. ``scipy.stats.f`` holds several (``generic_moment``,
``vecentropy``), so ``f.moment(6, ...)`` anywhere in a process changed the
digest of ``f``: lmfit's ``conf_interval``, which reads ``f``, missed in a
new process with "global `lmfit.confidence.f` changed". What that ufunc
does is the function it was built from, which stays in the key.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

JOB = """
    import sys

    import numpy as np

    import cash


    def double(x):
        return 2 * x


    def triple(x):
        return 3 * x


    VEC = np.vectorize({"double": double, "triple": triple}[sys.argv[2]], otypes=[float])


    @cash.cache
    def scaled(n):
        return float(VEC(np.arange(n)).sum())


    if sys.argv[1] == "warm":
        VEC(np.arange(2))  # fills the ufunc before the key is built
    first = scaled(4)
    VEC(np.arange(3))
    print(first, scaled(4), float(VEC(np.arange(4)).sum()))
    info = scaled.cache_info()
    print(info["hits"], info["misses"], info["changed"])
"""

SCIPY_JOB = """
    import sys

    from scipy.stats import f

    import cash


    @cash.cache
    def f_compare(dchi):
        return float(f.cdf(dchi, 2, 40))


    if sys.argv[1] == "warm":
        f.moment(6, 50, 60)
        f.entropy(2, 3)
    print(f_compare(1.5) == float(f.cdf(1.5, 2, 40)))
    info = f_compare.cache_info()
    print(info["hits"], info["misses"], info["changed"])
"""


def _runs(folder: Path, job: str, *argvs: tuple[str, ...]) -> list[str]:
    folder.mkdir(exist_ok=True)
    (folder / "job.py").write_text(textwrap.dedent(job), encoding="utf-8")
    return [run_python("job.py", *argv, cwd=folder).stdout.strip().splitlines()[-1] for argv in argvs]


def test_a_vectorized_global_hits_whether_or_not_it_was_called_first(tmp_path):
    pytest.importorskip("numpy")
    folder = tmp_path / "vec"
    assert _runs(folder, JOB, ("cold", "double"), ("warm", "double"), ("cold", "double")) == [
        "1 1 {}",
        "2 0 {}",
        "2 0 {}",
    ]
    # The function it vectorizes is keyed: another one misses.
    out = run_python("job.py", "warm", "triple", cwd=folder).stdout.strip().splitlines()
    assert out[0] == "18.0 18.0 18.0"
    assert out[1].startswith(
        "1 1 {'global VEC changed; it no longer uses helper job.double; it now uses helper job.triple'"
    )


def test_a_scipy_distribution_read_by_a_cached_function_hits_after_its_moments_were_used(tmp_path):
    pytest.importorskip("scipy.stats")
    folder = tmp_path / "scipy"
    assert _runs(folder, SCIPY_JOB, ("cold",), ("warm",), ("cold",)) == ["0 1 {}", "1 0 {}", "1 0 {}"]


def _double(x):
    return 2 * x


def _triple(x):
    return 3 * x


class _Holder:
    def __init__(self, fn):
        self.vec = None
        self.fn = fn
        self.excluded = {"unused"}  # a set: the key opens the object up


def test_an_argument_holding_a_vectorized_function_keys_after_it_was_called(tmp_path):
    np = pytest.importorskip("numpy")
    import cash

    c = cash.Cash(cache_dir=str(tmp_path / "c"))

    @c.cache
    def total(holder, n):
        return float(holder.vec(np.arange(n)).sum())

    holder = _Holder(_double)
    holder.vec = np.vectorize(_double, otypes=[float])
    assert total(holder, 4) == 12.0
    # Called, it keeps a ufunc pickle cannot write: the key still builds.
    holder.vec(np.arange(2))
    assert total(holder, 4) == 12.0
    assert total.cache_info()["hits"] == 1
    # The function it vectorizes is keyed.
    other = _Holder(_triple)
    other.vec = np.vectorize(_triple, otypes=[float])
    other.vec(np.arange(2))
    assert total(other, 4) == float(other.vec(np.arange(4)).sum()) == 18.0
    assert total.cache_info()["misses"] == 2
