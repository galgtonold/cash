"""CAS-186: does the NOTEBOOK path share CAS-183's parameter-defaults blind spot?

CAS-183 was a decorator-path bug: a parameter default lives on the function
OBJECT (``__defaults__`` / ``__kwdefaults__``), not in the code object, so the
``@cash.cache`` fingerprint (built from code-object fields) could not see it.
Editing ``n_estimators=300`` -> ``400`` returned the 300-tree model on an instant
HIT.

``notebook/function_tracker.py`` has the same shape: ``get_function_source_hash``
hashes ``inspect.getsource(func)`` (text) with an ``_update_code_object_hash``
bytecode fallback -- neither carries ``__defaults__``. So the ``func_source_hash``
channel that feeds a callable input into a consumer's cache key
(``statement/lineage.py`` line ~110 and ``cache_key._process_input_var``) is
*blind to defaults*, exactly like the decorator path.

**Verdict: NOT vulnerable.** A ``def`` statement's output ``f`` is assigned a
LINEAGE that folds in the free variables of its default expressions -- the analyzer
(``notebook/analysis.py`` ``_handle_function``) descends into ``args.defaults`` in
the enclosing scope, so ``THRESHOLD`` in ``def f(x, t=THRESHOLD)`` is an INPUT of the
``def`` statement. That input lineage flows into ``f``'s output lineage and thence
into every consumer's cache key. The text/bytecode blind spot is real but
backstopped by this second (lineage) channel, which the decorator path did not have.

Each leak test is self-proving: it first does an UNCHANGED re-run and asserts the
consumer RESTORES (a cache HIT -- proving the statement genuinely caches, so a stale
result was on the table), THEN changes the default and asserts the consumer
RECOMPUTES to the fresh value. Had the default change been invisible, the second
re-run would have RESTORED the stale value -- the exact CAS-183 failure.

The oracle is the real kernel (``nb_runner``); the discriminator is a value-based
``@cash:no-cache`` probe that reads the LIVE namespace, so a stale cache HIT on the
consumer surfaces as a wrong printed value rather than being replayed from the
probe's own cached output. Both consumers are forced to cache with
``# @cash:persist`` -- without it the sub-millisecond ``result = f(10)`` never writes
an entry, so it would always recompute and the test could not observe a leak.
"""

import pytest
from conftest import shows_cached, shows_executed

pytestmark = pytest.mark.upstream

# `%cash_badge print` puts the per-cell status (EXECUTED / CACHED / NOT CACHED)
# into the cell output so a cache HIT vs a recompute is directly observable.
SETUP = "import cash\n%cash_on\n%cash_badge print"


def test_plain_literal_default_edit_recomputes_consumer(nb_runner):
    """Shape 1 -- plain literal default ``def f(x, t=100)`` edited to ``t=200``.

    Editing the literal changes the ``def`` cell's SOURCE TEXT, so the statement's
    ``source_hash`` changes, ``f``'s lineage changes, and the consumer's key
    changes. This is the ordinary case CAS-186 predicted the statement-text hash
    would mask -- confirm it does.
    """
    nb_runner.create_notebook([
        SETUP,                                          # 1
        "def f(x, t=100):\n    return x + t",           # 2
        "# @cash:persist\nresult = f(10)",              # 3  (forced to cache)
        "# @cash:no-cache\nprint('R', result)",         # 4  (reads live value)
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert shows_executed(nb_runner.get_output(3)), nb_runner.get_output(3)
    assert "R 110" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Unchanged re-run: the consumer must HIT. This proves the statement genuinely
    # caches, so had the following edit been invisible it would RESTORE the stale
    # 110 -- the CAS-183 failure mode. run_all also rebinds `f` (cell 2 re-executes),
    # so the only thing that can make cell 3 recompute below is a changed key.
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_output(3)
    assert "R 110" in nb_runner.get_output(4), nb_runner.get_output(4)

    # Edit ONLY the default literal.
    nb_runner.set_cell_source(2, "def f(x, t=200):\n    return x + t")
    nb_runner.run_all()
    consumer = nb_runner.get_output(3)
    out = nb_runner.get_output(4)
    assert not shows_cached(consumer), (
        f"consumer served the literal-default result from cache (CAS-186 leak): {consumer!r}"
    )
    assert "R 210" in out, (
        f"consumer served a STALE literal-default result (CAS-186 leak): {out!r}"
    )


def test_enclosing_value_default_change_recomputes_consumer(nb_runner):
    """Shape 2 -- THE CAS-183 twin: default is an enclosing NAME whose VALUE changes
    while the ``def`` cell's source text stays byte-identical.

    ``t=THRESHOLD`` is evaluated at def-time in the enclosing scope, so ``THRESHOLD``
    is NOT in ``f``'s body ``co_names`` (the read-globals channel misses it) and the
    ``def`` text does not change (the func_source_hash channel misses it). If those
    were the only channels, the consumer would HIT stale -- the exact CAS-183
    failure. The masking channel under test: the analyzer counts ``THRESHOLD`` as an
    input of the ``def`` statement, so ``f``'s output lineage folds it in.
    """
    nb_runner.create_notebook([
        SETUP,                                          # 1
        "THRESHOLD = 0.5",                              # 2
        "def f(x, t=THRESHOLD):\n    return x + t",     # 3  (text is invariant)
        "# @cash:persist\nresult = f(10)",              # 4  (forced to cache)
        "# @cash:no-cache\nprint('R', result)",         # 5  (reads live value)
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert shows_executed(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert "R 10.5" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Unchanged re-run: the consumer must HIT (stale result now possible).
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert "R 10.5" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Change the VALUE the default binds, without touching f's source text.
    nb_runner.set_cell_source(2, "THRESHOLD = 100.0")
    nb_runner.run_all()
    consumer = nb_runner.get_output(4)
    out = nb_runner.get_output(5)
    assert not shows_cached(consumer), (
        f"consumer served the enclosing-value default from cache (CAS-186 leak, "
        f"the CAS-183 twin): {consumer!r}"
    )
    assert "R 110.0" in out, (
        f"consumer served a STALE enclosing-value default (CAS-186 leak, the "
        f"CAS-183 twin): {out!r}"
    )


def test_unchanged_default_still_hits(nb_runner):
    """Over-invalidation guard: an UNCHANGED default must still HIT on an isolated
    re-run.

    The mirror of the two leak tests. Whatever carries the default into the key must
    not make an unchanged default miss -- that regression would be worse than the
    bug it guards. Pins the enclosing-value shape (the more fragile one) to a cache
    HIT when nothing changed.
    """
    nb_runner.create_notebook([
        SETUP,                                          # 1
        "THRESHOLD = 0.5",                              # 2
        "def f(x, t=THRESHOLD):\n    return x + t",     # 3
        "# @cash:persist\nresult = f(10)",              # 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    # First run computes and writes the entry (RAM+DISK).
    assert shows_executed(nb_runner.get_output(4)), nb_runner.get_output(4)

    # Re-run the consumer in isolation with nothing changed -> must be a cache HIT.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert shows_cached(out), (
        f"unchanged default failed to HIT -- over-invalidation regression: {out!r}"
    )
