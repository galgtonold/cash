"""A cached function's ttl reaches the cached callers that reach it as a value.

A caller that calls a ttl'd cached function by name refreshes at least as
often as it (`FunctionRegistry.effective_ttl`). One that reaches it as an
argument (``passed(rates)``) or through a dict (``SOURCES[name]()``) had it
in its key but not its ttl: after the ttl ran out the caller kept serving the
value computed before, with no warning.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

JOB = """
    import time
    import cash

    @cash.cache(ttl=1)
    def rates():
        return time.time()  # @cash:assume-safe  (stands in for a fetch)

    SOURCES = {"rates": rates}

    @cash.cache
    def passed(source):
        return source()

    @cash.cache
    def via_dict(name):
        return SOURCES[name]()

    first = passed(rates), via_dict("rates")
    print(first == (passed(rates), via_dict("rates")))  # within the ttl: the stored answers
    time.sleep(1.2)
    again = passed(rates), via_dict("rates")
    print([a != b for a, b in zip(first, again)])
"""


def test_a_caller_refreshes_when_the_ttl_of_what_it_reached_runs_out(tmp_path):
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    out = run_python("job.py", cwd=tmp_path).stdout.split("\n")
    assert out[:2] == ["True", "[True, True]"]
