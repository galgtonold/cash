"""First-run correctness for one-shot loop iterables in the fast-loop path (CAS-121).

The single-unit fast-loop optimisation caches a large loop as one opaque unit by
re-executing it *from source* — which evaluates the loop header a SECOND time
(``process`` already evaluated it once to size the loop).  For a re-iterable
container built by a side-effect-free header that double evaluation is harmless.
For a ONE-SHOT consumable it is not: a stored generator / ``iter(...)`` /
``map`` / ``zip`` is already exhausted, and a side-effecting call such as
``drain()`` (which empties a global) returns an empty source the second time.
Either way the loop body never runs on the FIRST execution and the result is
wrong (a sum comes out 0), diverging from a plain kernel.

The fix evaluates the header exactly once: a self-iterator value or a header
that calls a bare non-builtin name is routed to the per-iteration path (driven
from the single, already-evaluated iterator), while re-iterable containers keep
the byte-identical single-unit fast path.

Companion to CAS-120 (``test_reassign_accumulator_loop_trust.py``), which fixed
the *separate* downstream-read re-drain.  This file covers the *within-first-run*
double evaluation.
"""
import time

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations]


# ---------------------------------------------------------------------------
# 1. Core: a side-effecting consumable header reaches the single-unit fast path
#    (>50 iterations, 3 body statements -> >1s estimated overhead) and must be
#    evaluated EXACTLY ONCE on the first run.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_draining_call_list_first_run_single_eval(nb_runner):
    """``drain()`` returns a 60-element list (has ``__len__`` -> hits the
    single-unit heuristic).  A second evaluation empties ``q`` before the body
    runs, so the double-eval bug yields ``total=0``."""
    nb_runner.create_notebook([
        "q = list(range(60))",
        "def drain():\n"
        "    global q\n"
        "    out, q = q, []\n"
        "    return out",
        "total = 0\n"
        "for item in drain():\n"
        "    a = item + 1\n"
        "    b = a * 2\n"
        "    total = total + b",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # sum((i+1)*2 for i in range(60)) == 3660
    assert "total=3660" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.timeout(90)
def test_draining_call_tuple_first_run_single_eval(nb_runner):
    """Same as above but the consumable is a *tuple* (also re-iterable-by-type,
    also ``__len__`` -> single-unit) produced by a side-effecting call."""
    nb_runner.create_notebook([
        "q = list(range(60))",
        "def drain():\n"
        "    global q\n"
        "    out, q = tuple(q), []\n"
        "    return out",
        "total = 0\n"
        "for item in drain():\n"
        "    a = item + 1\n"
        "    b = a * 2\n"
        "    total = total + b",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=3660" in nb_runner.get_output(4), nb_runner.get_output(4)


# ---------------------------------------------------------------------------
# 2. Consumable-type variants: generator / map / zip / iter([...]).  These lack
#    ``__len__`` so they go through the per-iteration path, but each is a genuine
#    one-shot source (a draining call over a global): a second evaluation of the
#    header would drain the exhausted global to an empty sum. First-run value
#    must equal plain Python.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_generator_header_first_run_single_eval(nb_runner):
    nb_runner.create_notebook([
        "q = list(range(6))",
        "def gdrain():\n    global q\n    d, q = q, []\n    return (i for i in d)",
        "total = 0\nfor v in gdrain():\n    total = total + v",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=15" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.timeout(90)
def test_map_header_first_run_single_eval(nb_runner):
    nb_runner.create_notebook([
        "q = list(range(6))",
        "def mdrain():\n    global q\n    d, q = q, []\n    return map(lambda i: i * 2, d)",
        "total = 0\nfor v in mdrain():\n    total = total + v",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # sum(i*2 for i in range(6)) == 30
    assert "total=30" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.timeout(90)
def test_zip_header_first_run_single_eval(nb_runner):
    nb_runner.create_notebook([
        "q = list(range(6))",
        "def zdrain():\n    global q\n    d, q = q, []\n    return zip(d, d)",
        "total = 0\nfor a, b in zdrain():\n    total = total + a + b",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # sum(i+i for i in range(6)) == 30
    assert "total=30" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.timeout(90)
def test_iter_list_header_first_run_single_eval(nb_runner):
    """``iter([...])`` is a self-iterator (``iter(x) is x``); stored and then
    looped, a re-evaluation would restart from an exhausted iterator."""
    nb_runner.create_notebook([
        "src = iter([1, 2, 3, 4, 5])",
        "total = 0\nfor v in src:\n    total = total + v",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=15" in nb_runner.get_output(3), nb_runner.get_output(3)


# ---------------------------------------------------------------------------
# 3. Re-iterable containers are UNCHANGED: a list / range loop still uses the
#    single-unit fast path, produces the correct result, and still caches on an
#    identical 2nd run (restore is observably faster than the first compute).
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_reiterable_list_still_single_unit_and_caches(nb_runner):
    """A bare-name list header (side-effect-free, re-iterable) stays on the
    single-unit fast path: correct result, and the expensive first run is
    restored from cache on the identical 2nd run."""
    nb_runner.create_notebook([
        "import time\ndata = list(range(60))",
        "acc = 0\n"
        "for x in data:\n"
        # 60 x 50ms = ~3s of real work. The ratio asserted below is
        # overhead / (work + overhead), so what matters is that the work
        # DWARFS the fixed per-run orchestration. At the original 10ms the
        # work was ~0.6s against ~0.4s of contended overhead -- a ratio of
        # ~0.45 against a 0.5 bound, which is why this failed under load.
        # At 50ms the same overhead yields ~0.14, roughly 3.5x of headroom.
        # Still far above _SPLIT_MAX_ITER_SEC, so the loop stays on the
        # single-unit path this test is about.
        "    time.sleep(0.05)\n"
        "    t = x + 1\n"
        "    acc = acc + t",
        "print(f'acc={acc}')",
    ])
    nb_runner.start_kernel()

    t0 = time.time()
    nb_runner.run_all()
    dur1 = time.time() - t0
    # sum(x+1 for x in range(60)) == 1830
    assert "acc=1830" in nb_runner.get_output(3), nb_runner.get_output(3)

    t0 = time.time()
    nb_runner.run_all()
    dur2 = time.time() - t0
    assert "acc=1830" in nb_runner.get_output(3), nb_runner.get_output(3)
    # Restored from the single-unit cache entry: no re-sleeping. Measured on
    # an idle box: run1=3.09s, run2=0.06s, ratio 0.02. With caching disabled
    # the same test measures 1.00. The bound sits in open space between them,
    # and run2 would have to grow 25x -- from 0.06s to 1.5s -- before load
    # could false-fail it.
    assert dur2 < dur1 * 0.5, (
        f"single-unit re-iterable loop did not restore from cache: "
        f"run1={dur1:.2f}s run2={dur2:.2f}s ratio={dur2 / dur1:.2f} "
        f"(expected <0.5; a restore measures ~0.02, a full recompute ~1.00)"
    )


@pytest.mark.timeout(90)
def test_reiterable_range_still_correct_both_runs(nb_runner):
    """``range(...)`` (a pure builtin producer) stays on the fast path and is
    correct + idempotent across two runs."""
    nb_runner.create_notebook([
        "acc = 0\n"
        "for i in range(60):\n"
        "    a = i * 2\n"
        "    b = a + 1\n"
        "    acc = acc + b",
        "print(f'acc={acc}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # sum(i*2+1 for i in range(60)) == 3600
    assert "acc=3600" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    assert "acc=3600" in nb_runner.get_output(2), nb_runner.get_output(2)


# ---------------------------------------------------------------------------
# 4. End-to-end with CAS-120: a reassignment accumulator over a one-shot
#    consumable is correct on the first run (CAS-121: header evaluated once) AND
#    survives a plain downstream re-run (CAS-120: the trusted accumulator is not
#    re-computed, so ``drain()`` is not called a second time and the source is
#    not re-drained).
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_consumable_accumulator_first_run_and_plain_rerun(nb_runner):
    nb_runner.create_notebook([
        "q = list(range(60))",
        "def drain():\n"
        "    global q\n"
        "    out, q = q, []\n"
        "    return out",
        "total = 0\n"
        "for item in drain():\n"
        "    a = item + 1\n"
        "    b = a * 2\n"
        "    total = total + b",
        "print(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=3660" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Plain re-run of the downstream reader, no edits. A plain kernel just reads
    # total=3660; cash must not re-execute the loop (which would call drain()
    # again over the now-empty q and collapse total to 0).
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "total=3660" in out, f"consumable accumulator re-drained on re-run: {out}"
