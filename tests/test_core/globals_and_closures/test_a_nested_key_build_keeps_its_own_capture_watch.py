"""Each key build keeps its own record of the captures it folded.

A global handed to a call is folded into the key provisionally, then hashed
again after the body ran: if the call itself moved it, the name stops being
folded (see ``test_capture_fold_bare_arg``). The names to re-check were kept as
scratch state on the ``Cash`` instance and reset by every key build. Another
key build in between -- a second thread, or a cached call made by a
``dynamic_depends_on`` resolver while the outer key is being built -- replaced
the outer call's record with its own, so the outer call judged the wrong names.
An accumulator was then never demoted, and the function missed on every call.
"""

from __future__ import annotations

import linecache
import uuid
import warnings

import cash


def test_a_cached_call_in_a_resolver_does_not_replace_the_outer_record(tmp_path):
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def lookup():
        return 1

    def resolver():
        lookup()
        return []

    def mutate(x):
        x.append(len(x))
        return len(x)

    ns = {"__name__": "cash_test_ns", "mutate": mutate, "ACC": []}
    src = "def f():\n    return mutate(ACC)\n"
    filename = f"<cash-test-{uuid.uuid4().hex}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    exec(compile(src, filename, "exec"), ns)
    fn = c.cache(dynamic_depends_on=resolver)(ns["f"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = [fn() for _ in range(5)]

    assert vals[2:] == [vals[2]] * 3, f"never converged: {vals}"
    assert len(ns["ACC"]) <= 2, f"accumulator kept growing: {ns['ACC']}"
