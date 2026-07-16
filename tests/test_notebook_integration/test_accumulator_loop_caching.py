"""Accumulator-loop caching fast path (CAS-145).

A *pure* accumulator loop — ``out = []`` then ``for e in it: out.append(f(e))``
— is byte-identical to the comprehension ``out = [f(e) for e in it]`` yet was
refused caching entirely because the ``append`` reads as an in-place mutation
("In-place mutation on: out"; 1.0x). This suite pins the narrow fast path that
makes that exact shape cacheable by routing the whole loop through the statement
cache as ONE unit, capturing BOTH the accumulator AND the leaked loop variable
as outputs (so the namespace on a hit is identical to running the real loop).

The correctness gates:
  * #1 caches + restores, result byte-identical to a no-cash run;
  * #2 survives a REAL kernel restart;
  * #3 the leaked loop variable ``e`` is preserved on a hit (the loop-var-leak
    guard — a comprehension rewrite would drop it);
  * #4 a loop with a genuine side effect is NOT matched and re-fires every run
    (the wrong-result guard);
  * #5 a pre-seeded / non-empty accumulator is NOT matched;
  * #6 / #7 an upstream edit of the iterable / body function invalidates.

``slow`` lives in its OWN cell (not the ``%cash_on`` setup cell) so cash tracks
its lineage and source hash — an input without a lineage would refuse the loop
for an unrelated reason ("Input variable missing lineage").
"""
import asyncio

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations, pytest.mark.timeout(120)]

# ``print`` badge so the RESTORED status lands in the cell's text output.
SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    "import time"
)
# Slow enough that the loop clears the "too cheap to cache" floor.
SLOW = "def slow(e):\n    time.sleep(0.03)\n    return e * 10"


def _restart(nb_runner):
    """Restart the kernel in place and re-inject the notebook path."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(nb_runner.client.km._async_restart_kernel(now=True))
    loop.run_until_complete(nb_runner.client.kc._async_wait_for_ready(timeout=30))
    nb_runner._inject_notebook_path()


# ---------------------------------------------------------------------------
# #1 Core: the accumulator loop caches, restores on an isolated re-run, and the
#    result is byte-identical to a plain (no-cash) run.
# ---------------------------------------------------------------------------

def test_accumulator_loop_caches(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3, 4, 5]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Byte-identical to a plain kernel: slow(e) == e*10.
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Isolated re-run of the loop cell must land a cache HIT (RESTORED), NOT
    # re-execute the loop. Without the fast path the loop is refused caching and
    # this re-run recomputes (no RESTORED badge) — the fails-without/passes-with
    # gate.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "RESTORED" in out, f"accumulator loop did not cache: {out!r}"

    nb_runner.run_cell(5)
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #2 The cached loop survives a REAL kernel restart (persisted to disk).
# ---------------------------------------------------------------------------

def test_accumulator_loop_survives_restart(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3, 4, 5]",
        # persist directive on the FOR line so it scopes to the loop unit, not
        # the (too-cheap) seed above it.
        "out = []\n"
        "# @cash:persist\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5)

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on
    nb_runner.run_cell(2)  # def slow (re-derives the same source hash)
    nb_runner.run_cell(3)  # items (re-derives the same lineage)

    nb_runner.run_cell(4)
    out4 = nb_runner.get_output(4)
    assert "RESTORED" in out4, f"loop did not restore from disk after restart: {out4!r}"

    nb_runner.run_cell(5)
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #3 Loop-var-leak guard: the leaked loop variable ``e`` must be preserved on a
#    cache hit (equal to the LAST item), NOT dropped. A restart makes ``e``
#    genuinely absent, so the only way a downstream cell can read it back is if
#    the cached loop RESTORED it — a comprehension rewrite would have lost it.
# ---------------------------------------------------------------------------

def test_loop_variable_preserved_on_hit(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [10, 20, 30]",
        "out = []\n"
        "# @cash:persist\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        # no-cache so this always re-reads the live ``e`` rather than replaying.
        "# @cash:no-cache\n"
        "print(f'e={e}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "e=30" in nb_runner.get_output(5), nb_runner.get_output(5)

    _restart(nb_runner)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)

    # After the restart ``e`` does NOT exist. The loop cell hits the disk cache
    # and must restore ``e`` (the last item) alongside ``out``.
    nb_runner.run_cell(4)
    assert "RESTORED" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.run_cell(5)
    out5 = nb_runner.get_output(5)
    assert "e=30" in out5, f"leaked loop variable not restored on cache hit: {out5!r}"


# ---------------------------------------------------------------------------
# #4 Wrong-result guard: a loop body that appends AND performs a side effect is
#    NOT matched (the body is not a single append), so it is still refused and
#    the side effect fires on EVERY run — proven by an append-mode file that
#    grows. The file is read from the TEST side (no reader cell) so cash's
#    file-writer re-execution machinery never double-counts it.
# ---------------------------------------------------------------------------

def test_side_effect_loop_not_cached(nb_runner, tmp_path):
    sink = tmp_path / "sink.txt"
    sink_s = str(sink).replace("\\", "/")
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        f"SINK = r'{sink_s}'\nitems = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))\n"
        "    with open(SINK, 'a') as f:\n"
        "        f.write(str(e) + '\\n')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert sink.read_text().count("\n") == 3, sink.read_text()

    # Re-run the loop cell: the side effect (file append) must fire AGAIN — the
    # loop is not cached, so the file grows to 6 lines. A wrongly-cached loop
    # would skip the write and leave it at 3.
    nb_runner.run_cell(4)
    loop_out = nb_runner.get_output(4)
    assert "RESTORED" not in loop_out, f"side-effect loop was wrongly cached: {loop_out!r}"
    assert sink.read_text().count("\n") == 6, (
        f"side effect was skipped on re-run (loop wrongly cached): "
        f"{sink.read_text()!r}"
    )


# ---------------------------------------------------------------------------
# #5 A pre-seeded / non-empty accumulator is NOT matched (caching a partial
#    accumulator would drop or double its prefix). Falls back to today's
#    behaviour, which still produces the correct value.
# ---------------------------------------------------------------------------

def test_preseeded_accumulator_not_cached(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        # non-empty seed -> not the fresh-empty shape -> not matched.
        "out = [0]\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[0, 10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Isolated re-run: not cached (no RESTORED), and the value stays correct
    # (the seeded prefix is preserved, not doubled).
    nb_runner.run_cell(4)
    assert "RESTORED" not in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(5)
    assert "out=[0, 10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)


def test_prior_cell_seed_not_cached(nb_runner):
    """The seed lives in a PRIOR cell, so the loop's immediately-preceding
    top-level sibling is not ``out = []`` -> not matched."""
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []",  # seed in its own cell
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(6), nb_runner.get_output(6)

    nb_runner.run_cell(5)
    assert "RESTORED" not in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #6 Editing the iterable upstream invalidates the cached loop.
# ---------------------------------------------------------------------------

def test_iterable_edit_invalidates(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Edit the upstream iterable -> the loop must re-run and downstream reflect it.
    nb_runner.set_cell_source(3, "items = [1, 2, 3, 4]")
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "out=[10, 20, 30, 40]" in out, f"iterable edit not reflected: {out!r}"


# ---------------------------------------------------------------------------
# #7 Editing the body function's definition invalidates the cached loop.
# ---------------------------------------------------------------------------

def test_body_function_edit_invalidates(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Redefine slow: e * 100 instead of e * 10. The loop's cache key includes the
    # function source hash, so it must re-run.
    nb_runner.set_cell_source(2, SLOW.replace("return e * 10", "return e * 100"))
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "out=[100, 200, 300]" in out, f"body-function edit not reflected: {out!r}"
