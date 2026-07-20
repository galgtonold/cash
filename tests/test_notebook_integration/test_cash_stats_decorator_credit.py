"""CAS-222: %cash_stats must credit @cash.cache hits, not report them as a cost.

P2's round-10 finding: in an ML session (expensive work behind the decorator,
which is the docs' own recommendation), a warm pass that avoided a 30s fit via a
decorator hit made %cash_stats print "cash cost you 27.3s" — the exact inverse
of cash's value — while `cache_info()` and `explain()` were both correct.

Root cause, established by inspection with a sleep-mocked trainer so the compute
cost is a known constant: `_update_session_stats` walked the statement-metrics
stream only, so a decorator hit contributed nothing to gross saved. The value
came from the decorator, so the wrapping statement only did a fast lookup and
read as cheap COMPUTED work.

(The *dramatic* -27s magnitude also needed ~29s of phantom overhead, which a
controlled mock never reproduces and which P2's session had via the CAS-218
discovery thrash — now fixed. This suite pins the reporting gap, which is the
part reproducible in-process.)

Compute cost is mocked with ``time.sleep`` so the numbers are deterministic; the
assertions are on sign and structure, never wall-clock exact.
"""
import json
import re

import pytest

pytestmark = pytest.mark.libraries

COST = 2.0  # stands in for a slow fit, scaled for test speed

SETUP = "import cash\n%cash_on"
DEFINE = (
    "import time\n"
    "@cash.cache\n"
    "def train(x):\n"
    f"    time.sleep({COST})\n"
    "    return x * 3\n"
    "print('defined')"
)
STATS = "%cash_stats json"


def _stats(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    assert m, f"no JSON in %cash_stats output: {text!r}"
    return json.loads(m.group(0))


def test_decorator_hit_credited_and_verified_in_one_session(nb_runner):
    """Miss then hit in the SAME session: the saving is credited AND verified.

    Both calls carry ``# @cash:no-cache`` so the wrapping statement never caches
    the result -- the value crosses the second call only through the decorator,
    which is exactly the case the old accounting could not see. The first call
    measures the compute this session, so the second (a hit) is creditable as
    verified under the CAS-157 rule, and the headline net is a real win.
    """
    nb_runner.create_notebook([
        SETUP,
        DEFINE,
        "a = train(7)  # @cash:no-cache\nprint('a=', a)",   # miss: sleeps, measured
        "b = train(7)  # @cash:no-cache\nprint('b=', b)",   # hit: no sleep, same key
        STATS,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    s = _stats(nb_runner.get_output(5))

    # The decorator hit's saving reaches gross AND verified...
    assert s['total_time_saved'] >= COST * 0.8, s
    assert s['total_verified_saved'] >= COST * 0.8, s
    # ...so the honest, verified headline is a win, not a cost.
    assert s['net_time_saved'] > 0, s
    assert 'statements_cacheable_hit' in s and s['statements_cacheable_hit'] >= 1, s


def test_decorator_hit_after_restart_is_gross_not_a_phantom_cost(nb_runner):
    """The post-restart / post-reset case: no miss to measure, so no VERIFIED
    credit -- but the report must still not claim a cost.

    ``%cash_stats reset`` drops the session's measured baselines, mimicking a
    kernel restart where the compute happened in a prior session. The hit can
    then only be an *upper-bound* saving, and the fix's job here is narrower but
    critical: the gross/upper-bound must reflect the hit, so the summary reads
    "at best a win" instead of the old "cash cost you N".
    """
    nb_runner.create_notebook([
        SETUP,
        DEFINE,
        "cold = train(9)  # @cash:no-cache\nprint('cold=', cold)",  # miss
        "%cash_stats reset",                                        # forget baselines
        "warm = train(9)  # @cash:no-cache\nprint('warm=', warm)",  # hit
        STATS,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    s = _stats(nb_runner.get_output(6))

    # Gross reflects the saving even though it can't be verified this session...
    assert s['total_time_saved'] >= COST * 0.8, s
    # ...so the generous bound is a clear win -- the old bug printed a cost here.
    assert s['net_time_saved_upper_bound'] > 0, s
    # Verified stays honest: the reset forgot the measurement, so no verified credit.
    assert s['total_verified_saved'] == 0.0, s


def test_plain_statement_caching_stats_unchanged(nb_runner):
    """Guard: a session with NO decorator must behave exactly as before.

    The fix adds a branch that is a strict no-op when a statement carries no
    decorator_calls, so ordinary statement-level caching stats must be untouched.
    """
    nb_runner.create_notebook([
        SETUP,
        "import time\n"
        "def slow():\n"
        f"    time.sleep({COST})\n"
        "    return 123\n"
        "print('defined')",
        "val = slow()\nprint('val=', val)",   # cold: computes
        STATS,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _stats(nb_runner.get_output(4))
    # No decorator anywhere, so no decorator crediting can have fired.
    assert cold['total_time_saved'] == 0.0, cold
    assert cold['statements_cacheable_hit'] == 0, cold

    nb_runner.run_all()   # warm: statement restores
    warm = _stats(nb_runner.get_output(4))
    # The saving is the ordinary statement restore, credited exactly as before.
    assert warm['total_time_saved'] >= COST * 0.8, warm
    assert warm['statements_restored'] >= 1, warm
