"""CAS-184: a bare alias bind (``b = a``) must never restore from cache.

``b = a`` is an ordinary assignment, so cash used to cache *that statement* and,
on a warm re-run, rebind ``b`` to a DESERIALISED COPY taken before any later
mutation of the shared object. The two names then silently stop being the same
object -- which Python guarantees they are -- and a mutation through ``a`` is
invisible through ``b``.

Three harness constraints these tests depend on. Each one, left out, turns a
test here into one that passes while proving nothing:

1. **>=3 warm re-runs.** The bug does not appear on the first warm re-run; that
   one still re-executes. A one-repetition test reports this shape as working,
   which is how CAS-170 came to assert -- wrongly -- that aliases were "correct
   by construction".
2. **``# @cash:persist`` on the alias bind.** A pointer copy executes in ~0ms,
   under the 10ms ``min_execution_time_to_cache_seconds`` floor, so cash normally
   never writes a cache entry for it and the bug stays LATENT. ``persist``
   bypasses that floor (the only other things that do: a >=10ms measured time --
   which is what a big, slow-to-hash object like a fitted estimator hits in the
   wild, per the original report -- file dependencies, and ``persist_all``).
   Without the directive these tests pass on the buggy code.
3. **``# @cash:no-cache`` on EVERY probe print.** The directive binds to ONE
   statement: it walks back over consecutive COMMENT lines only, so a single
   leading directive covers the first statement and lets the rest replay from
   cache. A partly-cached probe prints stale text next to the live values it
   contradicts (exactly the "cached output vs live control flow" half of the
   ticket). Every probe cell here is built by ``_live`` and asserts ``RESTORED``
   is absent, so a regression in that scoping fails loudly instead of lying.
"""
import asyncio

import pytest

from tests.test_notebook_integration.conftest import NotebookTestRunner

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print"

WARM_REPS = 4  # 1 cold + 3 warm re-runs.

BOX = (
    "class Box:\n"
    "    def __init__(self):\n"
    "        self.tag = None\n"
    "obj = Box()"
)


def _restart(nb_runner):
    """Restart the kernel in place and re-inject the notebook path."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(nb_runner.client.km._async_restart_kernel(now=True))
    loop.run_until_complete(nb_runner.client.kc._async_wait_for_ready(timeout=30))
    nb_runner._inject_notebook_path()


def _live(*lines: str) -> str:
    """A probe cell whose EVERY statement is individually ``@cash:no-cache``."""
    return "\n".join(f"# @cash:no-cache\n{line}" for line in lines)


# ----------------------------------------------------------------------
# The core guarantee: identity survives warm re-runs
# ----------------------------------------------------------------------

def test_alias_identity_holds_across_warm_reruns(nb_runner):
    """THE CAS-184 guard: ``backup = obj`` keeps ``backup is obj`` on every re-run.

    The object is mutated AFTER the alias bind, so a restored pre-mutation copy is
    detectable two ways: identity breaks and the mutation is invisible through
    ``backup``. A plain custom class suffices -- the bug is in the ASSIGNMENT, not
    in sklearn, so nothing here needs a heavy library.
    """
    nb_runner.create_notebook([
        SETUP,                                                          # 1
        BOX,                                                            # 2
        "# @cash:persist\nbackup = obj",                                # 3
        "obj.tag = 'fitted'",                                           # 4
        _live(
            "print('same', backup is obj)",
            "print('backup_tag', backup.tag)",
        ),                                                              # 5
    ])
    nb_runner.start_kernel()

    for rep in range(WARM_REPS):
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "RESTORED" not in out, (
            f"rep {rep}: the probe cell replayed from cache, so it is not reading "
            f"the live namespace -- this probe is invalid, not passing: {out!r}"
        )
        assert "same True" in out, (
            f"rep {rep}: alias identity broken -- `backup = obj` restored a copy "
            f"(CAS-184): {out!r}"
        )
        assert "backup_tag fitted" in out, (
            f"rep {rep}: mutation through `obj` invisible via `backup` -- the alias "
            f"is a stale pre-mutation copy: {out!r}"
        )


def test_alias_mutation_visible_through_both_names(nb_runner):
    """A mutation through EITHER name is visible through the other, every re-run.

    Not redundant with the identity check: identity is the mechanism, shared
    mutation is what a user actually observes. Mutating through the ALIAS
    (``backup.append``) also proves the restore is not handing back a copy in the
    other direction.
    """
    nb_runner.create_notebook([
        SETUP,                                        # 1
        "data = [1]",                                 # 2
        "# @cash:persist\nbackup = data",             # 3
        "data.append(2)",                             # 4
        "backup.append(3)",                           # 5
        _live(
            "print('same', backup is data)",
            "print('data', data)",
            "print('backup', backup)",
        ),                                            # 6
    ])
    nb_runner.start_kernel()

    for rep in range(WARM_REPS):
        nb_runner.run_all()
        out = nb_runner.get_output(6)
        assert "RESTORED" not in out, f"rep {rep}: probe cell replayed: {out!r}"
        assert "same True" in out, f"rep {rep}: alias identity broken: {out!r}"
        # One list, two names: both must show BOTH appends.
        assert "data [1, 2, 3]" in out, f"rep {rep}: {out!r}"
        assert "backup [1, 2, 3]" in out, f"rep {rep}: {out!r}"


def test_alias_output_matches_plain_kernel_ground_truth(tmp_path):
    """Ground truth: cash ON must print exactly what a cash-OFF kernel prints.

    The oracle is a real kernel rather than a hand-written literal: the bar is
    identity with plain Python, so a literal would only encode our opinion of it.
    Guards against a "fix" that makes the assertions above pass along some
    cash-specific path while still disagreeing with the language.
    """
    probe = "print('same', backup is obj)\nprint('backup_tag', backup.tag)"
    cells_off = [BOX, "backup = obj", "obj.tag = 'fitted'", probe]
    cells_on = [SETUP, BOX, "# @cash:persist\nbackup = obj", "obj.tag = 'fitted'",
                _live("print('same', backup is obj)",
                      "print('backup_tag', backup.tag)")]

    def _run(work_dir, cells, with_cash, probe_cell):
        work_dir.mkdir(parents=True, exist_ok=True)
        runner = NotebookTestRunner(work_dir=work_dir, use_pool=False)
        try:
            runner.create_notebook(cells)
            runner.start_kernel(with_cash=with_cash)
            for _ in range(WARM_REPS):
                runner.run_all()
            return runner.get_output(probe_cell)
        finally:
            runner.shutdown()

    def _probe_lines(raw: str) -> str:
        """The probe's printed lines, free of stream framing.

        Applied to BOTH sides. ipykernel buffers stdout and flushes on a timer,
        so two `print` calls arrive as one stream message or two depending on
        machine speed; joining them inserts a blank line in the second case.
        That framing is not what this test is about — comparing raw output made
        the oracle guard fail on slower machines with a spurious blank line,
        which reads as a cash bug and is not one. The badge is cash-only chrome
        and is filtered by the same rule.
        """
        return "\n".join(
            ln for ln in raw.splitlines() if ln.startswith(("same ", "backup_tag "))
        ).strip()

    off = _probe_lines(_run(tmp_path / "off", cells_off, False, 4))
    on_lines = _probe_lines(_run(tmp_path / "on", cells_on, True, 5))

    assert off == "same True\nbackup_tag fitted", (
        f"the cash-off oracle itself is wrong -- test bug: {off!r}"
    )
    assert on_lines == off, (
        "cash ON disagrees with a plain kernel on alias identity.\n"
        f"--- cash OFF ---\n{off!r}\n--- cash ON ---\n{on_lines!r}"
    )


def test_alias_after_restart_reconstructs_source(nb_runner):
    """The regression risk OF the fix: ``b = a`` re-executes, so ``a`` must EXIST.

    Refusing to cache means a warm isolated re-run RUNS ``backup = obj`` instead
    of restoring it. After a real kernel restart ``obj`` is gone from the
    namespace, so this would NameError where a restore previously "worked" --
    unless the upstream simulation re-derives ``obj`` first. It does; this pins
    that, because a NameError would be a strictly worse bug than the one fixed.
    """
    nb_runner.create_notebook([
        SETUP,                                                          # 1
        BOX,                                                            # 2
        "# @cash:persist\nbackup = obj",                                # 3
        _live("print('same', backup is obj)"),                          # 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on only -- `obj` is NOT rebuilt here

    for rep in range(3):
        # Cell 3 alone: `obj` is absent from the namespace on rep 0. The upstream
        # simulation must reconstruct it rather than let `backup = obj` NameError.
        nb_runner.run_cell(3)
        out3 = nb_runner.get_output(3)
        assert "NameError" not in out3, (
            f"rep {rep}: re-executing the alias bind NameError'd -- the upstream "
            f"simulation did not reconstruct its source (CAS-184 regression): {out3!r}"
        )
        nb_runner.run_cell(4)
        out4 = nb_runner.get_output(4)
        assert "same True" in out4, f"rep {rep}: {out4!r}"


def test_alias_bind_is_reported_not_cached(nb_runner):
    """The badge names the refusal, stably, and even against an explicit persist.

    ``# @cash:persist`` asks for caching; identity is not the user's to trade away
    by asking for a perf knob, so a correctness gate outranks it -- the same way
    CAS-144 refuses a matplotlib Figure. This is also the net-neutrality proof:
    ``skip_cache`` gates ``_save_to_cache``, so a refused alias bind never
    serialises the object at all. Refusing to cache a pointer copy cannot cost
    anything -- there is no work to save.
    """
    nb_runner.create_notebook([
        SETUP,
        "data = [1, 2, 3]",
        "# @cash:persist\nbackup = data",
    ])
    nb_runner.start_kernel()

    for rep in range(WARM_REPS):
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "NOT CACHED" in out, f"rep {rep}: alias bind cached: {out!r}"
        assert "Alias assignment" in out, f"rep {rep}: reason missing: {out!r}"
        assert "RESTORED" not in out, f"rep {rep}: alias bind restored: {out!r}"


def test_chained_and_tuple_alias_forms_keep_identity(nb_runner):
    """The multi-target forms alias too: ``b = c = a`` and ``e, f = a, d``.

    Both bind existing objects to new names by pointer copy, so both carry the
    same identity guarantee and the same zero cost as ``b = a``.
    """
    nb_runner.create_notebook([
        SETUP,                                        # 1
        "a = [1]\nd = [9]",                           # 2
        "# @cash:persist\nb = c = a",                 # 3
        "# @cash:persist\ne, f = a, d",               # 4
        "a.append(2)",                                # 5
        _live(
            "print('b', b is a)",
            "print('c', c is a)",
            "print('e', e is a)",
            "print('f', f is d)",
            "print('b_val', b)",
        ),                                            # 6
    ])
    nb_runner.start_kernel()

    for rep in range(WARM_REPS):
        nb_runner.run_all()
        out = nb_runner.get_output(6)
        assert "RESTORED" not in out, f"rep {rep}: probe cell replayed: {out!r}"
        for name in ("b", "c", "e", "f"):
            assert f"{name} True" in out, (
                f"rep {rep}: `{name}` lost identity with its source: {out!r}"
            )
        assert "b_val [1, 2]" in out, f"rep {rep}: alias missed mutation: {out!r}"


# ----------------------------------------------------------------------
# Scope of the refusal: it must stay narrow
# ----------------------------------------------------------------------

def test_computed_rhs_still_caches(nb_runner):
    """The refusal stays NARROW: a CALL on the RHS still caches.

    ``b = a.copy()`` / ``c = list(a)`` can be arbitrarily expensive, so the cost
    half of the CAS-184 argument does not transfer and they keep their cache.
    This is the anchor that stops the rule quietly growing into "never cache an
    assignment whose RHS mentions a variable", which would gut the cache.

    NOTE — the boundary moved under CAS-188. This test originally also asserted
    that ``d = a[0]`` keeps its cache, on the same "can be arbitrarily expensive"
    reasoning. That reasoning is sound for a call but NOT for a literal-key
    subscript, which is an O(1) dereference: measured against a ``%cash_off``
    kernel, ``b = lst[0]`` restored a stale pre-mutation COPY and diverged on the
    first warm re-run. A deref is free to re-run, so refusing it satisfies BOTH
    halves of the CAS-184 argument exactly as ``b = a`` does. Literal-key
    subscripts, attribute chains and ternaries over them are therefore refused
    now; see ``reference_alias_targets``. Computed-key subscripts (``a[i]``,
    ``df[mask]``) remain cached — those can be real filters.
    """
    nb_runner.create_notebook([
        SETUP,                                        # 1
        "a = [1, 2, 3]",                              # 2
        "# @cash:persist\nb = a.copy()",              # 3
        "# @cash:persist\nc = list(a)",               # 4
        "i = 0",                                      # 5
        "# @cash:persist\nd = a[i]",                  # 6  computed key -> filter-shaped
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    for rep in range(3):
        nb_runner.run_cells([3, 4, 6])
        for cell in (3, 4, 6):
            out = nb_runner.get_output(cell)
            assert "Alias assignment" not in out, (
                f"rep {rep}: computed RHS in cell {cell} wrongly refused as an "
                f"alias -- the gate is over-broad: {out!r}"
            )


def test_unpack_from_single_name_is_not_an_alias(nb_runner):
    """``b, c = a`` INDEXES ``a`` (binding ``a[0]``/``a[1]``); it is not aliasing.

    The gate must not confuse an unpack with a pointer copy: ``b`` is ``a[0]``, a
    different object from ``a``, so refusing to cache it would be unjustified by
    either half of the argument.
    """
    nb_runner.create_notebook([
        SETUP,                                        # 1
        "a = ([1], [2])",                             # 2
        "# @cash:persist\nb, c = a",                  # 3
        _live("print('b_is_a', b is a)", "print('b_is_a0', b is a[0])"),  # 4
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out3 = nb_runner.get_output(3)
    assert "Alias assignment" not in out3, (
        f"an unpack was misclassified as an alias bind: {out3!r}"
    )
    out4 = nb_runner.get_output(4)
    assert "b_is_a False" in out4, out4
    assert "b_is_a0 True" in out4, out4
