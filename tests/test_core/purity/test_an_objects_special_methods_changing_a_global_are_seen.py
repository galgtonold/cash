"""A ``@cash.cache`` function whose body runs a special method or a property
of an object of yours that appends to a global list is not silent about it.

``BAG.value`` or ``BAG['k']`` read by a global name, a property of an object
handed in, a context manager handed in and entered with ``with cm:``: the
second, identical call was a hit, the append was skipped, and nothing was
said (the same append in a plain helper warns IMPURE-SIDE-EFFECTS). Now the
object read by name is read as a helper (IMPURE-SIDE-EFFECTS), and a call
handing the object in runs uncached (IMPURE-HANDED-CALLABLE).

Control: an object whose special methods change nothing is served from the
cache without a warning.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SCRIPT = textwrap.dedent("""
    import json, warnings
    import cash

    LOG = []

    class Bag:
        @property
        def value(self):
            LOG.append('prop')
            return 1
        def __getitem__(self, k):
            LOG.append('item')
            return 1

    class Plain:
        @property
        def value(self):
            return 1
        def __getitem__(self, k):
            return 1

    class timer:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            LOG.append('exit')
            return False

    BAG = Bag()
    PLAIN = Plain()

    @cash.cache
    def g_prop(v):
        return BAG.value + v

    @cash.cache
    def g_item(v):
        return BAG['k'] + v

    @cash.cache
    def g_arg_prop(b, v):
        return b.value + v

    @cash.cache
    def g_arg_cm(cm, v):
        with cm:
            return v * 2

    @cash.cache
    def g_plain(b, v):
        return b.value + b['k'] + PLAIN['k'] + v

    CASES = {
        "prop": lambda: g_prop(1),
        "item": lambda: g_item(1),
        "arg_prop": lambda: g_arg_prop(BAG, 1),
        "arg_cm": lambda: g_arg_cm(timer(), 1),
        "plain": lambda: g_plain(PLAIN, 1),
    }
    out = {}
    for name, call in CASES.items():
        LOG.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(), call()
        codes = sorted({str(w.message).split("]")[0].lstrip("[") for w in caught if str(w.message).startswith("[")})
        out[name] = {"rows": len(LOG), "codes": codes}
    print("RESULT " + json.dumps(out))
""")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    where = tmp_path_factory.mktemp("special")
    (where / "main.py").write_text(SCRIPT, encoding="utf-8")
    proc = run_python("main.py", cwd=where)
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("case", ["prop", "item"])
def test_an_object_read_by_name_is_reported_as_a_side_effect(results, case):
    assert "IMPURE-SIDE-EFFECTS" in results[case]["codes"], results[case]


@pytest.mark.parametrize("case", ["arg_prop", "arg_cm"])
def test_an_object_handed_in_runs_the_call_each_time(results, case):
    got = results[case]
    assert got["rows"] == 2, got
    assert "IMPURE-HANDED-CALLABLE" in got["codes"], got


def test_an_object_changing_nothing_is_served_without_a_warning(results):
    got = results["plain"]
    assert not got["codes"], got
