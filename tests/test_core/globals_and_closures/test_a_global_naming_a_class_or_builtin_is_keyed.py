"""A global or a capture that names a class, a builtin or a library function
is keyed by what it names.

``DTYPE = np.float32`` edited to ``np.float64``, ``AGG = sum`` to ``max``,
``OP = math.floor`` to ``math.ceil``: the helper walk keys the user's own
code, so a name bound to a library class or a C function reached no channel,
and the next run served the old result -- read directly, through a helper,
through ``from settings import AGG``, as ``settings.AGG`` or from a factory
helper's closure. Rebinding the name while the program runs was stale the
same way, and so was rebinding the class an ``isinstance`` or ``except``
tests against.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytest.importorskip("numpy")

SETTINGS = {
    1: "import math\nimport numpy as np\nDTYPE = np.float32\nAGG = sum\nOP = math.floor\nUFUNC = np.add\n",
    2: "import math\nimport numpy as np\nDTYPE = np.float64\nAGG = max\nOP = math.ceil\nUFUNC = np.multiply\n",
}

JOB = textwrap.dedent("""
    import numpy as np
    import cash
    import settings
    from settings import AGG, DTYPE, OP, UFUNC

    def make(fn):
        def apply(xs):
            return fn(xs)
        return apply

    agg = make(settings.AGG)

    def total(xs):
        return AGG(xs)

    @cash.cache
    def load(n):
        return str(np.arange(n).astype(DTYPE).dtype)

    @cash.cache
    def summarize(xs):
        return AGG(xs)

    @cash.cache
    def bucket(v):
        return OP(v)

    @cash.cache
    def through_helper(xs):
        return total(xs)

    @cash.cache
    def through_module(xs):
        return settings.AGG(xs)

    @cash.cache
    def through_local_import(xs):
        from settings import AGG as local
        return local(xs)

    @cash.cache
    def through_factory(xs):
        return agg(xs)

    def make_cached(op):
        @cash.cache
        def combine(a, b):
            return int(op(a, b))
        return combine

    combine = make_cached(settings.UFUNC)

    print(load(3), summarize([1, 5]), bucket(1.5), through_helper([1, 5]), through_module([1, 5]),
          through_local_import([1, 5]), through_factory([1, 5]), combine(2, 3))
""")


def test_editing_what_a_setting_names_recomputes_in_the_next_run(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    outs = []
    for version in (1, 2):
        (tmp_path / "settings.py").write_text(SETTINGS[version], encoding="utf-8")
        outs.append(run_python("job.py", cwd=tmp_path).stdout.strip())
    assert outs[0] == "float32 6 1 6 6 6 6 5"
    assert outs[1] == "float64 5 2 5 5 5 5 6"


REBOUND = textwrap.dedent("""
    import math
    import numpy as np
    import cash

    DTYPE = np.float32
    AGG = sum
    OP = math.floor

    class Small(Exception):
        pass

    class Large(Exception):
        pass

    KIND = int
    E = Small

    @cash.cache
    def load(n):
        return str(np.arange(n).astype(DTYPE).dtype)

    @cash.cache
    def summarize(xs):
        return AGG(xs)

    @cash.cache
    def bucket(v):
        return OP(v)

    @cash.cache
    def is_kind(v):
        return isinstance(v, KIND)

    @cash.cache
    def caught(v):
        try:
            raise Large(v)
        except E:
            return "caught"
        except Exception:
            return "other"

    def show():
        return " ".join(map(str, (load(3), summarize([1, 5]), bucket(1.5), is_kind(1.5), caught(1))))

    print(show())
    DTYPE, AGG, OP, KIND, E = np.float64, max, math.ceil, float, Large
    print(show())
    print(show(), summarize.cache_info()["hits"])
""")


def test_rebinding_the_name_in_this_process_recomputes(tmp_path):
    (tmp_path / "job.py").write_text(REBOUND, encoding="utf-8")
    lines = run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()
    assert lines[0] == "float32 6 1 False other"
    assert lines[1] == "float64 5 2 True caught"
    # Positive control: unchanged names still hit.
    assert lines[2] == "float64 5 2 True caught 1"
