"""A captured FUNCTION has to reach the cache key.

Round-14 gate finding (WRONG). The strategy-factory shape --

    def make_scorer(weight_fn):
        @cash.cache
        def score(px, lookback):
            return f(px, weight_fn(lookback))
        return score

-- folded nothing at all. Two scorers built with different weightings share a
source and a qualname (``make_scorer.<locals>.score``), so they collided on one
cache key and returned each other's results: `flat` and `ramp` both came back
0.025001250062501867, from a single body execution, with no warning.

Why the existing capture fold missed it: a capture the body PASSES TO A CALL is
marked unsafe and deliberately not folded (watch it rather than fold it blind),
and calling is exactly what you do with a captured function. A call cannot
mutate a function, so that reason does not apply here.

Note this is NOT the documented "closure passed as an argument" limitation.
That one is about an argument, it fails loudly at pickling, and the docs promise
it "does not return a wrong answer, it just never caches". This is the closure
on the decorated function itself, and it did return a wrong answer.
"""
from __future__ import annotations

import pytest

from cash.backends import InMemoryBackend
from cash.core import Cash


@pytest.fixture
def cash_instance():
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    yield c
    c.backend.clear()


def test_two_factory_built_functions_do_not_share_a_cache_entry(cash_instance):
    """The reported shape: different captured lambdas, same source and qualname."""
    ran: list[int] = []

    def make(weight):
        @cash_instance.cache
        def score(n):
            ran.append(n)
            return weight(n)
        return score

    doubler = make(lambda n: n * 2)
    tripler = make(lambda n: n * 3)

    assert doubler(10) == 20
    assert tripler(10) == 30, "the second scorer returned the first one's result"
    assert len(ran) == 2, f"only {len(ran)} body execution(s): the two collided"


def test_a_captured_function_built_by_a_factory_is_distinguished(cash_instance):
    """Source alone is not enough.

    A factory-built helper has the SAME source whatever it was built with, so
    fingerprinting the capture by source only pushes the collision down one
    level -- measured, both arms returned 20. The captured function's own
    captures have to fold too.
    """
    ran: list[int] = []

    def outer(k):
        return lambda n: n * k          # identical source for every k

    def make(weight):
        @cash_instance.cache
        def score(n):
            ran.append(n)
            return weight(n)
        return score

    assert make(outer(2))(10) == 20
    assert make(outer(3))(10) == 30, "collided one level down, in the helper"
    assert len(ran) == 2


def test_an_identically_built_function_still_hits(cash_instance):
    """The other half: distinguishing must not cost the cache.

    Folding a capture is the kind of change that "fixes" a collision by making
    every key unique, which would be a different bug. Rebuilding the same helper
    must still restore.
    """
    ran: list[int] = []

    def outer(k):
        return lambda n: n * k

    def make(weight):
        @cash_instance.cache
        def score(n):
            ran.append(n)
            return weight(n)
        return score

    assert make(outer(2))(10) == 20
    before = len(ran)
    assert make(outer(2))(10) == 20
    assert len(ran) == before, (
        "an identically-built helper missed the cache: the fold is now "
        "over-discriminating"
    )


def test_two_lambdas_on_the_same_source_line_are_distinguished(cash_instance):
    """`inspect.getsource` returns the whole LINE for a lambda.

    So two different lambdas written on one line share their source text and a
    source-only fingerprint collides -- measured, both arms returned "AAA".
    Their code objects differ, which is what separates them.
    """
    def make(weight):
        @cash_instance.cache
        def g():
            return weight()
        return g()

    first, second = make(lambda: "AAA"), make(lambda: "BBB")
    assert (first, second) == ("AAA", "BBB")


def test_the_capture_fingerprint_is_stable_across_processes(tmp_path):
    """A fingerprint must not move between runs, or every restart is a miss.

    The obvious way to digest a code object -- `repr()` -- embeds a memory
    address for nested code objects, which would be stable within a process and
    different in the next one. That failure is invisible in-process, so it has
    to be checked across two.
    """
    import subprocess
    import sys

    script = tmp_path / "fp.py"
    script.write_text(
        "from cash.core import Cash\n"
        "def outer(k):\n"
        "    return lambda n: [x * k for x in range(n)]\n"
        "print(Cash._code_fingerprint(outer(3).__code__))\n",
        encoding="utf-8",
    )
    runs = [
        subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] and runs[0] == runs[1], (
        f"the fingerprint moved between processes: {runs}"
    )


def test_editing_a_captured_helper_invalidates(cash_instance):
    """The point of keying on the capture at all: a changed helper must re-run."""
    ran: list[int] = []

    def make(weight):
        @cash_instance.cache
        def score(n):
            ran.append(n)
            return weight(n)
        return score

    assert make(lambda n: n * 2)(10) == 20
    assert make(lambda n: n * 2 + 1)(10) == 21, "an edited helper served a stale result"
    assert len(ran) == 2
