"""pytest setting and putting back its assertion hook keeps every fast path.

Around every test item pytest's assertion plugin binds
``_pytest.assertion.util._reprcompare`` (and ``_assertion_pass`` when a
plugin implements that hook) to a function of its own and puts ``None``
back afterwards. A callable rebound in a module namespace moves the world
for every memo, so under pytest each test began with an empty memo, and a
function the suite called once or twice per test was never served from its
fast path: PyThaiNLP's ``word_dict_trie`` with ``@lru_cache`` swapped for
``@cash.cache`` took 1673 hits across a test walk, none of them fast. No key
reads those two names unless code cash scanned names them, as the warnings
display hooks (`hit_memo._WARNING_HOOKS`), so they are quiet the same way,
and a function whose own code names one still sees each change.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from cash import Cash
from cash.decorator import hit_memo
from tests._scripts import run_python

util = pytest.importorskip("_pytest.assertion.util")

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
]


def reads_the_hook(x):
    # Code that names the hook: a rebinding must still reach it.
    return (x, util._reprcompare is None)


def _callbinrepr(op, left, right):
    return None


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


@pytest.fixture(autouse=True)
def _hooks_kept():
    saved = util._reprcompare, util._assertion_pass
    yield
    util._reprcompare, util._assertion_pass = saved


def _fast(cached, *args):
    """Call *cached*; was the call served from the fast path?"""
    before = hit_memo.status()["fast_hits"]
    result = cached(*args)
    return result, hit_memo.status()["fast_hits"] > before


def _settle(cached, *args):
    """Call until a call is served from the fast path (another test's
    leftovers can move the world now and then)."""
    deadline = time.monotonic() + 5 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while not _fast(cached, *args)[1]:
        assert time.monotonic() < deadline, "not served from the fast path: the test proves nothing"
        time.sleep(0.01)


#: A fresh interpreter: no code scanned in it named the hook (`world_names`
#: never forgets a name, so an earlier test in this process could have).
_SCRIPT = textwrap.dedent(
    """
    import time
    import _pytest.assertion.util as util
    from cash import Cash
    from cash.decorator import hit_memo

    def unrelated(x):
        return x + 1

    def callbinrepr(op, left, right):
        return None

    def fast(cached, *args):
        before = hit_memo.status()["fast_hits"]
        result = cached(*args)
        return result, hit_memo.status()["fast_hits"] > before

    f = Cash(backend="memory", register_magic=False).cache(unrelated)
    deadline = time.monotonic() + 5.0
    while not fast(f, 1)[1]:
        assert time.monotonic() < deadline, "never served from the fast path"
        time.sleep(0.01)
    served = []
    for _ in range(5):
        # `_pytest.assertion.pytest_runtest_protocol`: set before each test
        # item runs, put back after it.
        util._reprcompare = callbinrepr
        util._assertion_pass = callbinrepr
        util._reprcompare = None
        util._assertion_pass = None
        served.append(fast(f, 1))
    print(served)
    assert served == [(2, True)] * 5, served
    """
)


def test_the_assertion_hook_set_and_put_back_keeps_a_fast_path(tmp_path):
    (tmp_path / "main.py").write_text(_SCRIPT)
    run_python("main.py", cwd=tmp_path)


def test_code_that_names_the_hook_still_sees_it_change(c):
    util._reprcompare = None
    g = c.cache(reads_the_hook)
    _settle(g, 1)
    util._reprcompare = _callbinrepr
    result, fast = _fast(g, 1)
    assert not fast
    assert result == reads_the_hook(1) == (1, False)
