"""A function of yours handed to a ``@cash.cache`` function, that changes a
global when it is called, runs on every call.

``run(record, 3)`` with ``record`` appending to a global list: the second,
identical call was a hit, so ``record`` did not run and the list missed a row,
with no warning. So did a ``map(record, xs)`` in the body, a bound method
changing a global, and ``record`` hidden in a ``functools.partial``. Such a
call now runs uncached, as plain Python does, and IMPURE-HANDED-CALLABLE says
so once.

Controls: a function that changes nothing is still served from the cache, and
``assume_safe=True`` keeps the call cached without a warning. Under
``strict=True`` the call raises, as every other finding does there.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SCRIPT = textwrap.dedent("""
    import functools, json, warnings
    import cash

    LOG = []
    BODY = []

    def record(v):
        LOG.append(v)
        return v * 2

    def pure(v):
        return v * 2

    class Model:
        def predict(self, v):
            LOG.append(v)
            return v + 1

    def apply_(f, v):
        return f(v)

    @cash.cache
    def run(f, v):
        BODY.append(1)
        return f(v)

    @cash.cache
    def run_map(f, xs):
        BODY.append(1)
        return list(map(f, xs))

    @cash.cache(assume_safe=True)
    def run_waived(f, v):
        BODY.append(1)
        return f(v)

    @cash.cache(strict=True)
    def run_strict(f, v):
        return f(v)

    MODEL = Model()
    CASES = {
        "handed": lambda: run(record, 3),
        "mapped": lambda: run_map(record, [1, 2]),
        "bound method": lambda: run(MODEL.predict, 4),
        "partial": lambda: run(functools.partial(apply_, record), 5),
        "pure": lambda: run(pure, 6),
        "assume_safe": lambda: run_waived(record, 7),
    }
    out = {}
    for name, call in CASES.items():
        LOG.clear()
        BODY.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first, second = call(), call()
        codes = sorted({str(w.message).split("]")[0].lstrip("[") for w in caught if str(w.message).startswith("[")})
        out[name] = {"rows": len(LOG), "bodies": len(BODY), "same": first == second, "codes": codes}
    try:
        run_strict(record, 8)
        out["strict"] = "no error"
    except cash.CashImpureFunctionError as e:
        out["strict"] = "raised: " + str(e)
    print("RESULT " + json.dumps(out))
""")

#: case -> rows LOG holds after two identical calls, as plain Python leaves it
CHANGES_STATE = {"handed": 2, "mapped": 4, "bound method": 2, "partial": 2}
#: the first case of each cached function: the warning is given once per function
WARNS_HERE = {"handed", "mapped"}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    where = tmp_path_factory.mktemp("handed")
    (where / "main.py").write_text(SCRIPT, encoding="utf-8")
    proc = run_python("main.py", cwd=where)
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("case", list(CHANGES_STATE))
def test_the_handed_function_runs_on_the_second_call(results, case):
    got = results[case]
    assert got["rows"] == CHANGES_STATE[case], got
    assert got["same"], got
    if case in WARNS_HERE:
        assert "IMPURE-HANDED-CALLABLE" in got["codes"], got


def test_a_function_that_changes_nothing_is_still_served_from_the_cache(results):
    got = results["pure"]
    assert got["bodies"] == 1, got
    assert "IMPURE-HANDED-CALLABLE" not in got["codes"], got


def test_assume_safe_keeps_the_call_cached_without_a_warning(results):
    got = results["assume_safe"]
    assert got["bodies"] == 1 and got["rows"] == 1, got
    assert "IMPURE-HANDED-CALLABLE" not in got["codes"], got


def test_strict_raises(results):
    assert results["strict"].startswith("raised: "), results["strict"]
    assert "strict=True" in results["strict"], results["strict"]
