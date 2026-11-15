"""A side effect on a line a traced call did not run does not refuse it.

A call that picks code by a run-time name is judged on what its trace says
it ran (`cash.decorator.code_trace`): a finding a hit would skip (a write to
a module-level list) in a function that ran refused the call, even when the
line it is on sat in a branch the call did not take. Found on
imbalanced-learn checked out as the user's code: ``ret[col] = ...`` in
``ArraysTransformer._transfrom_one`` and ``y.index = X.index`` in
``transform``, on branches a NumPy input never takes, refused every cached
``resample(sampler, X, y)``.

On Python 3.12+ the lines of code holding such a finding are watched on a
miss (``sys.monitoring`` LINE events, each line disabled after its first
callback), and a finding counts only when its line ran. Arguments, or a
global the trace checks, that take the branch run it and are refused. Below
3.12 such a call runs uncached, as before.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time

import pytest

TRACED = sys.version_info >= (3, 12)
needs_monitoring = pytest.mark.skipif(not TRACED, reason="sys.monitoring is Python 3.12+")

_PLUGINS = """
LOG = []
FLAG = False


def maybe_log(x, flag):
    y = x * 2
    if flag:
        LOG.append(x)
    return y


def by_global(x):
    y = x + 1
    if FLAG:
        LOG.append(x)
    return y


def spanning(x, flag):
    y = (
        x * 3
        if not flag
        else LOG.append(x)
    )
    return y


def nested_unused(x):
    def later():
        LOG.append(x)

    return x + 4


def branchy(n):
    t = 0
    for i in range(n):
        if i % 3 == 0:
            t += 1
        elif i % 3 == 1:
            t -= 1
        else:
            t += 2
        if i < 0:
            LOG.append(i)
    return t


def plain(n):
    t = 0
    for i in range(n):
        if i % 3 == 0:
            t += 1
        elif i % 3 == 1:
            t -= 1
        else:
            t += 2
    return t
"""

_PICK = """
@cash.cache
def pick(name, *args):
    RUNS.append(name)  # @cash:assume-safe
    return getattr(plugins, name)(*args)
"""


@pytest.fixture
def load(tmp_path, request):
    """Write ``plugins.py`` and a module holding the cached ``pick``, import
    both, and count the runs of its body (``RUNS``). Named after the test,
    so an entry an earlier test stored is not this one's."""
    module = "m_" + "".join(c if c.isalnum() else "_" for c in request.node.name)
    names = ("plugins", module)
    sys.path.insert(0, str(tmp_path))

    def _load():
        for name in names:
            sys.modules.pop(name, None)
        (tmp_path / "plugins.py").write_text(textwrap.dedent(_PLUGINS), encoding="utf-8")
        (tmp_path / f"{module}.py").write_text("import cash, plugins\nRUNS = []\n" + _PICK, encoding="utf-8")
        importlib.invalidate_caches()
        return importlib.import_module(module)

    yield _load
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


@needs_monitoring
def test_a_side_effect_in_a_branch_the_call_did_not_take_is_stored_and_hits(load):
    m = load()
    assert m.pick("maybe_log", 5, False) == 10
    assert m.pick("maybe_log", 5, False) == 10
    assert m.RUNS == ["maybe_log"]
    assert m.plugins.LOG == []


@needs_monitoring
def test_arguments_that_take_the_branch_run_it_every_time(load):
    m = load()
    assert m.pick("maybe_log", 5, False) == 10
    assert m.pick("maybe_log", 5, True) == 10
    assert m.pick("maybe_log", 5, True) == 10
    assert m.plugins.LOG == [5, 5]
    assert m.RUNS == ["maybe_log"] * 3
    # ``LOG``, which the function reads, moved: the first entry is not served.
    assert m.pick("maybe_log", 5, False) == 10
    assert m.RUNS == ["maybe_log"] * 4


@needs_monitoring
def test_the_lines_are_watched_again_on_the_next_miss_of_the_same_function(load):
    """Each line's event is disabled after its first callback: a later miss
    arms them again, so a branch taken only then is seen."""
    m = load()
    assert m.pick("maybe_log", 1, False) == 2
    assert m.pick("maybe_log", 2, False) == 4
    assert m.pick("maybe_log", 3, True) == 6
    assert m.pick("maybe_log", 3, True) == 6
    assert m.plugins.LOG == [3, 3]
    assert m.RUNS == ["maybe_log"] * 4


@needs_monitoring
def test_a_global_the_trace_checks_that_takes_the_branch_runs_it_and_is_refused(load):
    """Same arguments, same key: the trace's item for ``FLAG`` moves, the
    call runs, takes the branch and is not stored, so it does not serve the
    entry stored while ``FLAG`` was off, nor one of its own."""
    m = load()
    assert m.pick("by_global", 1) == 2
    assert m.pick("by_global", 1) == 2
    m.plugins.FLAG = True
    assert m.pick("by_global", 1) == 2
    assert m.pick("by_global", 1) == 2
    assert m.plugins.LOG == [1, 1]
    assert m.RUNS == ["by_global"] * 3
    m.plugins.FLAG = False
    assert m.pick("by_global", 1) == 2
    assert m.pick("by_global", 1) == 2
    assert m.RUNS == ["by_global"] * 4


@needs_monitoring
def test_a_statement_over_several_lines_is_judged_by_the_line_of_its_effect(load):
    m = load()
    assert m.pick("spanning", 2, False) == 6
    assert m.pick("spanning", 2, False) == 6
    assert m.RUNS == ["spanning"]
    assert m.pick("spanning", 2, True) is None
    assert m.pick("spanning", 2, True) is None
    assert m.plugins.LOG == [2, 2]


@needs_monitoring
def test_a_nested_function_that_never_ran_does_not_refuse(load):
    m = load()
    assert m.pick("nested_unused", 1) == 5
    assert m.pick("nested_unused", 1) == 5
    assert m.RUNS == ["nested_unused"]


def _miss_seconds(m, name: str, n: int) -> float:
    t0 = time.perf_counter()
    m.pick(name, n)
    return time.perf_counter() - t0


@needs_monitoring
def test_a_million_passes_over_a_branchy_function_cost_one_callback_per_line(load):
    """Watching lines costs a callback per line once per miss, not per pass:
    counted, which CI noise cannot move, with a loose bound on time beside
    it against the same loop with no finding to watch."""
    from cash.decorator import code_trace

    m = load()
    n = 1_000_000
    _miss_seconds(m, "plain", 10)
    _miss_seconds(m, "branchy", 10)
    lines = len({line for *_, line in m.plugins.branchy.__code__.co_lines() if line is not None})
    recorder = code_trace._recorder()
    before = recorder.line_events
    watched = min(_miss_seconds(m, "branchy", n + k) for k in range(2))
    callbacks = recorder.line_events - before
    assert 0 < callbacks <= 2 * lines, f"{callbacks} line callbacks over 2 misses of a {lines}-line function"
    unwatched = min(_miss_seconds(m, "plain", n + k) for k in range(2))
    assert m.RUNS.count("branchy") == 3
    assert watched < unwatched * 1.5 + 0.25, f"watched {watched:.3f}s, unwatched {unwatched:.3f}s"


@pytest.mark.skipif(TRACED, reason="the fallback below Python 3.12, where sys.monitoring is missing")
def test_below_python_3_12_such_a_call_runs_every_time(load):
    m = load()
    with pytest.warns(Warning):
        assert m.pick("maybe_log", 5, False) == 10
    assert m.pick("maybe_log", 5, False) == 10
    assert m.RUNS == ["maybe_log", "maybe_log"]
