"""A repeated hit is served from the function's memo, and only while nothing
its key reads has moved.

With the extension, a hit of a function whose key reads only immutable
values, code and watched dicts is remembered per argument tuple and served
again without building the key (``cash.decorator.hit_memo``, ~0.5 us instead
of ~125 us). Each test below first checks that the memo IS serving (else the
test would pass on the ordinary path and prove nothing), then moves one thing
the key reads -- the function's code, a global, a closure cell, a default, a
class attribute, an argument edited in place, a file it read, the settings --
and checks that the next call answers as the ordinary path does.
"""

from __future__ import annotations

import importlib
import math
import sys
import threading
import time

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    # The functions count their runs in a global dict, which cash rightly
    # calls a side effect.
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

#: Body runs. A dict, not a list: a list global keeps a function off the
#: memo (the watchers cannot vouch for it), a dict is watched.
RUNS = {"n": 0}
RATE = 2
TABLE = {"a": 1, "b": 2}
ITEMS = [1, 2, 3]


def helper(x):
    return x + 100


class Settings:
    scale = 3


def times_rate(x):
    RUNS["n"] += 1
    return x * RATE


def via_helper(x):
    RUNS["n"] += 1
    return helper(x)


def via_class(x):
    RUNS["n"] += 1
    return x * Settings.scale


def via_table(name):
    RUNS["n"] += 1
    return TABLE.get(name)


def via_list(i):
    RUNS["n"] += 1
    return ITEMS[i]


def with_default(x, k=10):
    RUNS["n"] += 1
    return x + k


def with_kwdefault(x, *, k=10):
    RUNS["n"] += 1
    return x + k


def plus_one(x):
    RUNS["n"] += 1
    return x + 1


def plus_two(x):
    RUNS["n"] += 1
    return x + 2


def typed(x):
    RUNS["n"] += 1
    return (x, type(x).__name__)


def summed(d):
    RUNS["n"] += 1
    return sum(v if isinstance(v, int) else sum(v) for v in d.values())


def listing(n):
    RUNS["n"] += 1
    return list(range(n))


def total(xs):
    RUNS["n"] += 1
    return sum(xs)


def reads(path):
    RUNS["n"] += 1
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(autouse=True)
def _put_back():
    global RATE, TABLE, ITEMS, helper
    saved = (RATE, TABLE, ITEMS, helper, Settings.scale)
    saved_code = {f: (f.__code__, f.__defaults__, f.__kwdefaults__) for f in (plus_one, with_default, with_kwdefault)}
    RUNS["n"] = 0
    yield
    RATE, TABLE, ITEMS, helper, Settings.scale = saved
    for f, (code, defaults, kwdefaults) in saved_code.items():
        f.__code__, f.__defaults__, f.__kwdefaults__ = code, defaults, kwdefaults


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def fast_hits() -> int:
    return hit_memo.status()["fast_hits"]


def served_fast(f, *args, **kwargs):
    """Call *f*; return its result and whether the memo served it."""
    before_hits, before_runs = fast_hits(), RUNS["n"]
    result = f(*args, **kwargs)
    return result, fast_hits() > before_hits and RUNS["n"] == before_runs


def warm(f, *args, **kwargs):
    """Call *f* until the memo serves it (miss, remembered, served).

    Not a fixed number of calls: a dict the memo watches that another
    thread changes (a background write's first import adds to
    ``sys.modules``) sends it back to scan, at a doubling backoff. A traced
    caller needs five calls with nothing moving, and two such changes in a
    loaded run made six too few. Whatever the memo serves is still
    checked by the test."""
    deadline = time.monotonic() + 5.0
    while True:
        result, fast = served_fast(f, *args, **kwargs)
        if fast:
            return result
        if time.monotonic() > deadline:
            break
    raise AssertionError(f"{f.__name__}{args} was never served from the memo")


# -- served, and counted as an ordinary hit is ----------------------------


def test_a_repeated_hit_is_served_and_counted_as_a_hit(c):
    f = c.cache(times_rate)
    assert warm(f, 21) == 42
    misses, runs = f.cache_info()["misses"], RUNS["n"]
    hits = f.cache_info()["hits"]
    for _ in range(500):
        assert served_fast(f, 21) == (42, True)
    info = f.cache_info()
    assert info["hits"] == hits + 500
    assert info["misses"] == misses
    assert RUNS["n"] == runs
    assert info["total_time_saved"] > 0


def test_a_fast_hit_logs_what_an_ordinary_hit_logs(c):
    f = c.cache(plus_one)
    warm(f, 1)
    ordinary = [e for e in c.drain_decorator_calls() if e["cache_hit"]][0]
    for _ in range(3):
        assert served_fast(f, 1) == (2, True)
    events = c.drain_decorator_calls()
    assert len(events) == 3
    for event in events:
        assert event.keys() == ordinary.keys()
        assert event["cache_hit"] is True
        assert event["cache_key"] == ordinary["cache_key"]
        assert event["time_saved"] == ordinary["time_saved"]
    assert [e["timestamp"] for e in events] == sorted(e["timestamp"] for e in events)


def test_an_int_a_float_and_a_bool_are_not_one_key(c):
    f = c.cache(typed)
    warm(f, 1)
    assert f(1.0) == (1.0, "float")
    assert f(True) == (True, "bool")
    assert f(1) == (1, "int")
    warm(f, 0.0)
    assert f(-0.0) == (-0.0, "float") and math.copysign(1, f(-0.0)[0]) == -1
    assert f("1") == ("1", "str")
    assert f(b"1") == (b"1", "bytes")
    assert f((1,)) == ((1,), "tuple")
    assert f((1.0,)) == ((1.0,), "tuple") and type(f((1.0,))[0][0]) is float


def test_a_keyword_and_a_positional_call_answer_the_same(c):
    f = c.cache(with_default)
    warm(f, 1)
    assert f(1, k=5) == 6
    assert f(1, 5) == 6
    assert f(x=1) == 11
    assert f(1) == 11


def test_a_mutable_result_handed_out_is_a_copy(c):
    f = c.cache(listing)
    first = warm(f, 3)
    first.append("edited")
    assert served_fast(f, 3) == ([0, 1, 2], True)


# -- every change the key reads ends the memo's answer --------------------


_MODULE_V1 = """
def helper(x):
    return x + 1


def work(x):
    return helper(x)
"""

_MODULE_V2 = """
def helper(x):
    return x + 22


def work(x):
    return helper(x) * 10
"""


@pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheStoreFailedWarning")
def test_editing_the_code_is_seen(c, tmp_path, monkeypatch):
    """A helper edited in its file and reloaded (what an edit-and-rerun in
    a session does): the cached caller answers with the new helper, and the
    edited function decorated again answers with its new body."""
    path = tmp_path / "hitpath_edited_module.py"
    path.write_text(_MODULE_V1, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import hitpath_edited_module as mod

    try:
        f = c.cache(mod.work)
        assert warm(f, 1) == 2
        path.write_text(_MODULE_V2, encoding="utf-8")
        importlib.invalidate_caches()
        importlib.reload(mod)
        assert f(1) == 23
        assert c.cache(mod.work)(1) == 230
    finally:
        sys.modules.pop("hitpath_edited_module", None)


def test_a_rebound_global_is_seen(c):
    global RATE
    f = c.cache(times_rate)
    assert warm(f, 5) == 10
    RATE = 3
    assert f(5) == 15
    assert warm(f, 5) == 15


def test_a_replaced_helper_is_seen(c):
    global helper
    f = c.cache(via_helper)
    assert warm(f, 1) == 101
    helper = lambda x: x + 200
    assert f(1) == 201


def test_a_class_attribute_it_reads_is_seen(c):
    f = c.cache(via_class)
    assert warm(f, 2) == 6
    Settings.scale = 5
    assert f(2) == 10


def test_a_global_dict_edited_in_place_is_seen(c):
    f = c.cache(via_table)
    assert warm(f, "a") == 1
    TABLE["a"] = 7
    assert f("a") == 7


def test_a_global_list_keeps_the_function_on_the_ordinary_path(c):
    f = c.cache(via_list)
    for _ in range(5):
        assert served_fast(f, 0)[1] is False
    ITEMS[0] = 9
    assert f(0) == 9


def test_a_closure_cell_is_seen(c):
    def make():
        factor = 2

        def scaled(x):
            RUNS["n"] += 1
            return x * factor

        def set_factor(v):
            nonlocal factor
            factor = v

        return scaled, set_factor

    scaled, set_factor = make()
    f = c.cache(scaled)
    assert warm(f, 4) == 8
    set_factor(10)
    assert f(4) == 40


def test_a_changed_default_is_seen(c):
    f = c.cache(with_default)
    assert warm(f, 1) == 11
    with_default.__defaults__ = (20,)
    assert f(1) == 21


def test_a_changed_keyword_default_is_seen(c):
    f = c.cache(with_kwdefault)
    assert warm(f, 1) == 11
    with_kwdefault.__kwdefaults__ = {"k": 30}
    assert f(1) == 31


def test_a_small_list_argument_is_keyed_by_value(c):
    f = c.cache(total)
    xs = [1, 2, 3]
    assert warm(f, xs) == 6
    assert served_fast(f, [1, 2, 3]) == (6, True)
    xs.append(4)
    assert f(xs) == 10
    assert f([1, 2, 3.0]) == 6.0 and type(f([1, 2, 3.0])) is float
    assert f((1, 2, 3)) == 6


def test_a_big_argument_edited_in_place_is_seen(c):
    f = c.cache(summed)
    d = {f"k{i}": i for i in range(1000)}
    total = sum(range(1000))
    assert warm(f, d) == total
    d["k0"] = 1000
    assert f(d) == total + 1000
    assert warm(f, d) == total + 1000


def test_a_small_argument_edited_deep_inside_is_seen(c):
    f = c.cache(summed)
    d = {"a": [1, 2], "b": 3}
    assert warm(f, d) == 6
    d["a"].append(10)
    assert f(d) == 16


def test_an_equal_argument_that_is_another_object_answers_the_same(c):
    f = c.cache(summed)
    d = {"a": 1, "b": 2}
    assert warm(f, d) == 3
    assert f({"a": 1, "b": 2}) == 3
    assert f({"a": 1, "b": 5}) == 6


def test_a_file_it_read_is_seen(c, tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("one", encoding="utf-8")
    f = c.cache(reads)
    assert warm(f, str(path)) == "one"
    runs = RUNS["n"]
    path.write_text("two!", encoding="utf-8")
    assert f(str(path)) == "two!"
    assert RUNS["n"] == runs + 1


def test_a_ttl_is_honoured(c):
    f = c.cache(ttl=0.3)(plus_one)
    assert warm(f, 1) == 2
    runs = RUNS["n"]
    time.sleep(0.4)
    assert f(1) == 2
    assert RUNS["n"] == runs + 1


def test_cache_clear_is_seen(c):
    f = c.cache(plus_one)
    warm(f, 1)
    f.cache_clear()
    runs = RUNS["n"]
    assert f(1) == 2
    assert RUNS["n"] == runs + 1
    assert f.cache_info()["misses"] == 1


def test_disabling_cash_after_fast_hits_runs_the_body(c):
    f = c.cache(plus_one)
    warm(f, 1)
    c.config.disable = True
    try:
        runs = RUNS["n"]
        assert f(1) == 2
        assert f(1) == 2
        assert RUNS["n"] == runs + 2
    finally:
        c.config.disable = False


def test_two_cash_instances_keep_to_themselves(c, tmp_path):
    other = Cash(cache_dir=str(tmp_path / "other"), register_magic=False)
    warm(c.cache(plus_one), 1)
    g = other.cache(plus_one)
    runs = RUNS["n"]
    assert g(1) == 2
    assert RUNS["n"] == runs + 1, "one instance's memo answered for another"


def test_a_result_cache_if_rejects_is_never_served(c):
    f = c.cache(cache_if=lambda r: r > 100)(plus_one)
    for _ in range(5):
        assert f(1) == 2
    assert RUNS["n"] == 5


# -- what does not end it ----------------------------------------------------


def test_a_miss_of_another_function_keeps_the_memo_serving(c):
    """A miss stores, learns and may evict, but only for its own function:
    what it changes in the world the watchers see, and an entry it evicts
    the RAM check sees."""

    def tripled(x):  # touches nothing plus_one reads (RUNS is in its world)
        return x * 3

    f = c.cache(plus_one)
    other = c.cache(tripled)
    for i in range(50):
        other(-i)  # past what cash imports once, on its first misses
    warm(f, 1)
    served = []
    for i in range(40):
        other(i)  # a miss each
        result, fast = served_fast(f, 1)
        assert result == 2
        served.append(fast)
    # A module imported meanwhile (``sys.modules`` is watched whole) costs a
    # step or two; a miss elsewhere costs none.
    assert sum(served) >= 35, served


def test_a_module_level_loop_keeps_the_memo_serving(c, tmp_path):
    """A script's loop rebinds module globals (``for x in ...``) on every
    step; a function whose own ``x`` is local is not moved by that."""
    from tests._scripts import run_python

    script = tmp_path / "hitpath_loop.py"
    script.write_text(
        "import tempfile\n"
        "from cash import Cash\n"
        "from cash.decorator import hit_memo\n"
        "c = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)\n"
        "@c.cache\n"
        "def f(x):\n"
        "    y = x * 2\n"
        "    return y\n"
        "for _ in range(3):\n"
        "    f(1)\n"
        "before = hit_memo.status()['fast_hits']\n"
        "for x in range(1000):\n"
        "    y = f(1)\n"
        "print(hit_memo.status()['fast_hits'] - before)\n",
        encoding="utf-8",
    )
    done = run_python(script, cwd=tmp_path)
    assert int(done.stdout.split()[-1]) >= 990


def test_a_memo_the_world_kept_off_scans_again_after_a_while(c):
    """The backoff counts calls, and a scan that served nothing doubles it, up
    to 1,024. On a Windows runner 1,024 ordinary hits of a function that reads
    a file took seconds, so a world that moved now and then kept it off the
    memo for good. A scan is also due a quarter second after the last one."""
    f = c.cache(plus_one)
    assert warm(f, 1) == 2
    (memo,) = [m for m in hit_memo._MEMOS if m.state is hit_memo._STATE_OF[f]]
    memo.backoff = 1024  # as ten scans that served nothing leave it
    sys.modules["_cash_test_world_moves"] = None
    del sys.modules["_cash_test_world_moves"]
    deadline = time.monotonic() + 3.0
    calls = 0
    while time.monotonic() < deadline:
        calls += 1
        if served_fast(f, 1) == (2, True):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("never served again in 3 s")
    assert calls < 1024


def test_a_callees_miss_ends_its_callers_remembered_hits(c):
    inner = c.cache(times_rate)

    def outer_body(x):
        return inner(x) + 1

    outer = c.cache(outer_body)
    assert warm(outer, 2) == 5
    inner(7)  # a miss of the callee
    assert served_fast(outer, 2) == (5, False)
    assert warm(outer, 2) == 5


# -- inside another cached call, the ordinary path records the dependency --


def test_a_call_inside_another_cached_call_takes_the_ordinary_path(c):
    global RATE
    inner = c.cache(times_rate)
    warm(inner, 3)

    def outer_body(x):
        return inner(x) + 1

    outer = c.cache(outer_body)
    assert outer(3) == 7
    RATE = 10
    assert outer(3) == 31, "outer did not learn that it depends on inner's global"


def test_threads_serving_one_function_count_every_hit(c):
    f = c.cache(plus_one)
    warm(f, 1)
    before = f.cache_info()["hits"]

    def hammer():
        for _ in range(2000):
            assert f(1) == 2

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert f.cache_info()["hits"] == before + 8000


# -- the switches -----------------------------------------------------------

_SCRIPT = """
import tempfile
from cash import Cash
from cash.decorator import hit_memo

RATE = {"k": 2}
c = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)


@c.cache
def scaled(x, items=(1, 2)):
    return x * RATE["k"] + sum(items)


out = []
for i in range(200):
    out.append(scaled(i % 5))
    out.append(scaled(i % 3, items=[i % 2, 3]))
    if i == 100:
        RATE["k"] = 3
info = scaled.cache_info()
print(out, info["hits"], info["misses"])
print("FAST", hit_memo.status().get("fast_hits", 0), "VERIFIED", hit_memo.VERIFIED[0])
"""


def _run(tmp_path, env):
    from tests._scripts import run_python

    script = tmp_path / "hitpath_switch.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    done = run_python(script, cwd=tmp_path, env=env)
    answer, counters = done.stdout.strip().splitlines()[-2:]
    fast, verified = counters.split()[1::2]
    return answer, int(fast), int(verified)


def test_switched_off_and_verified_runs_answer_alike(tmp_path):
    """``CASH_HITPATH=0`` serves nothing fast; ``CASH_HITPATH_VERIFY=1``
    builds the key behind every fast hit and finds it the same; and all
    three answer, and count hits and misses, alike."""
    on, fast_on, _ = _run(tmp_path, {})
    off, fast_off, _ = _run(tmp_path, {"CASH_HITPATH": "0"})
    checked, _, verified = _run(tmp_path, {"CASH_HITPATH_VERIFY": "1"})
    assert on == off == checked
    assert fast_on > 300 and fast_off == 0
    assert verified > 300


# -- the memo key (C) --------------------------------------------------------


def _key(*args, **kwargs):
    return hit_memo._cw.any_key(args, kwargs)


def test_the_memo_key_tells_types_and_float_bits_apart():
    values = [1, 1.0, True, -0.0, 0.0, 0, False, None, "1", b"1", (1,), [1], {"1": 1}, 1 + 0j]
    keys = [_key(v) for v in values]
    assert None not in keys
    assert len(set(keys)) == len(values)
    assert _key(float("nan")) == _key(float("nan"))


def test_the_memo_key_keeps_positional_keyword_and_order_apart():
    assert _key(1, 2) != _key(2, 1)
    assert _key(1) != _key(x=1)
    assert _key(a=1, b=2) != _key(b=2, a=1)  # dict order: one more entry, never a wrong one
    assert _key([1, [2]]) != _key([[1], 2])
    assert _key((1, 2)) != _key([1, 2])


def test_the_memo_key_refuses_what_it_cannot_tell_by_value():
    class Box:
        pass

    assert _key(Box()) is None
    assert _key([Box()]) is None
    assert _key(frozenset({1})) is None
    assert _key(list(range(5000))) is None  # past the budget, and not watched
    assert _key(*range(21)) is not None  # more arguments than the quick key packs
    assert _key(1, Box()) is None


def test_a_state_that_is_not_one_is_never_served():
    assert hit_memo.fast_call([], (1,), {}) is hit_memo.MISS
    with pytest.raises(TypeError):
        hit_memo.fast_call(hit_memo.new_state(), [1], {})
