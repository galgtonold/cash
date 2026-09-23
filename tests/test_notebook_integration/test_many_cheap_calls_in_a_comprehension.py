"""A comprehension of many cheap calls is not slowed down by caching each one.

r23s4 read a folder with ``docs = pd.DataFrame([read_doc(p) for p in paths])``:
5,030 calls of a function reading one small file. Cached one by one -- a key,
a lookup, a store and a file tracker each, ~14 ms a call around ~1.6 ms of
work -- the cell went from 8.4 s to 71.6 s. Past 50 calls the site's calls
are timed plain on a few samples; when caching one costs more than three
times its work (the numbers a ``for`` loop is run as one unit by), the rest
run plain.

Observed through the decision trace (``call_site_decided``), not wall time.
A trivial callee makes the verdict certain: the key alone outweighs it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

N = 300
SETUP = "def norm(v):\n    return v * 2\nvalues = list(range(%d))" % N
USE = "out = [norm(v) for v in values]\nprint('SUM', sum(out))"


def _decided(trace, source):
    return [e for e in trace.events("call_site_decided", phase="run_all") if e.get("source") == source]


def test_a_comprehension_of_cheap_calls_runs_them_plain(upstream_trace, nb_runner):
    t = upstream_trace([SETUP, USE], lambda r: None)

    assert f"SUM {N * (N - 1)}" in nb_runner.get_output(2), nb_runner.get_output(2)
    decided = _decided(t, "norm(v)")
    assert decided and decided[0]["plain"] is True, decided


def test_a_short_comprehension_is_not_judged(upstream_trace, nb_runner):
    """Under 50 calls nothing is sampled: a handful of elements is exactly
    where per-element reuse pays."""
    t = upstream_trace(["def norm(v):\n    return v * 2\nvalues = list(range(20))", USE], lambda r: None)

    assert "SUM 380" in nb_runner.get_output(2)
    assert _decided(t, "norm(v)") == []
