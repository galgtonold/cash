"""``cash.cache(callable_instance)`` keys what the instance holds.

A callable instance has no ``__qualname__``, so its function name was its
``repr`` -- ``<Scaler object at 0x...>`` -- a new namespace in every process,
where no entry was ever found again. And its attributes never reached the
key: ``sc.k = 5`` after the first call served the result computed with 2.
The bound-method form (``cash.cache(sc.__call__)``) was keyed correctly.
"""

from __future__ import annotations

import textwrap

import pytest

import cash
from cash.decorator.function_identity import func_key
from tests._scripts import run_python

pytestmark = pytest.mark.core


class Scaler:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return self.k * x


def test_a_changed_attribute_recomputes():
    """THE BUG: the second call returned 6, the result for k=2."""
    c = cash.Cash(backend=cash.InMemoryBackend())
    sc = Scaler(2)
    f = c.cache(sc)
    assert f(3) == 6
    sc.k = 5
    assert f(3) == 15
    sc.k = 2
    assert f(3) == 6


def test_the_name_has_no_address():
    name = func_key(Scaler(2))
    assert "0x" not in name
    assert name.endswith("Scaler[instance]")


MAIN = textwrap.dedent("""
    import sys, time, warnings
    warnings.simplefilter("ignore")
    import cash

    class Scaler:
        def __init__(self, k):
            self.k = k

        def __call__(self, x):
            print("RAN", file=sys.stderr)  # @cash:assume-safe
            time.sleep(0.2)
            return self.k * x

    print(cash.cache(Scaler(int(sys.argv[1])))(3))
""")


def _run(tmp_path, k):
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    out = run_python("main.py", k, cwd=tmp_path, cache_dir=tmp_path / "cache")
    return int(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def test_a_new_process_hits(tmp_path):
    assert _run(tmp_path, 2) == (6, 1)
    assert _run(tmp_path, 2) == (6, 0)
    assert _run(tmp_path, 5) == (15, 1)
