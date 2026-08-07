"""CAS-261 step 2: a cheap many-iteration loop is learned and then split.

The band, measured on HEAD before this landed (n=124, warm rerun, wall clock
against a cash-off arm):

    body 0.1ms  ->  22ms off vs 215ms on   (9.8x SLOWER)
    body 1ms    -> 137ms off vs 324ms on   (2.4x slower)
    body 2.5ms  -> 320ms off vs 617ms on   (1.9x slower)

Every one re-ran all 124 calls AND paid ~190-300ms of per-iteration
decomposition overhead. Not a missed optimisation -- cash being worse than
not using cash. It falls between two floors: n=124 is under the single-unit
threshold so the loop decomposes, and each call is under
``call_unit._COST_FLOOR_S`` so nothing is stored per call either.

**The split is applied by the SIMULATOR, not the runtime.** The re-execution
planner runs the statements ``upstream/virtual_lineage.py`` modelled, so
splitting that model is what makes a head and a tail actually execute. Three
earlier attempts split in the runtime instead and each produced a silent
stale value; ``test_lineage_check_control_structure.py::
test_for_loop_still_reexecutes_when_inputs_change`` is the case that caught
them and must stay green.

Measured as counted real executions, never wall clock -- a timing assertion
would measure the machine under parallel load rather than the policy.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SETUP = "import cash\n%cash_on\n"
SETUP_OFF = "import cash\n"
UNRELATED = "unrelated = 1\n"

# Raises ONLY the split ceiling, from 6ms to 1s. The body is ~1ms plus ~2.2ms
# decomposition overhead, so against 6ms there is under 2x headroom and a
# kernel descheduled mid-probe measures a cheap loop as an expensive one;
# against 1s there is ~300x and no realistic stall crosses it. Descheduling
# only pushes a measurement UP, so raising the ceiling cannot manufacture a
# split that the policy would refuse on an idle machine.
#
# Deliberately NOT conftest's CASH_TEST_PIN_THRESHOLDS, which also zeroes
# call_cost_floor_seconds. This loop's premise is that per-call caching does
# NOT cover it (each call sits under the floor, so nothing is stored per
# call); zero that floor and every call caches individually, the rerun
# measures 0 real calls, and the test passes green without a split ever
# happening. Safe to leave set: prepare_for_test calls reset_session(), which
# restores config defaults, so the pin cannot leak into the next warm-kernel
# test.
SETUP_PINNED_CEILING = (
    "import cash\n"
    "cash.configure(loop_split_max_iter_seconds=1.0)\n"
    "%cash_on\n"
)

# Under the single-unit threshold (n>=125 for a 1-statement body), so this
# loop decomposes -- the band's shape.
_N = 124
# The runtime measures this many iterations before judging, and it becomes
# the split point. The head re-runs on every warm pass, so warm cost is
# bounded by it.
_K = 5


def _compute_def(counter, ms=1.0):
    """Busy-wait, not sleep: Windows sleep granularity (~1-15ms) would swamp
    a 1ms body and make the split verdict depend on the OS timer."""
    return (
        "import time\n"
        "def _busy(ms):\n"
        "    t = time.perf_counter() + ms / 1000.0\n"
        "    while time.perf_counter() < t:\n"
        "        pass\n"
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        f"    _busy({ms})\n"
        "    return v * 10\n"
    )


def _loop(n=_N):
    # Items start at 1, never 0: compute(0) returns `0 * 10`, which CPython
    # interns to the argument object itself and trips the call unit's
    # arg-identity refusal -- one permanently-uncached call that mimics a
    # caching bug (CAS-263).
    return f"for t in list(range(1, {n + 1})):\n    out.append(compute(t))\nprint('OUT', len(out))"


def _cells(counter, setup=SETUP, n=_N, ms=1.0):
    """SETUP(1) / unrelated(2) / compute def(3) / seed(4) / loop(5)."""
    return [setup, UNRELATED, _compute_def(counter, ms), "out = []\n", _loop(n)]


LOOP_CELL = 5
UNRELATED_CELL = 2
DEF_CELL = 3


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _why(work_dir):
    """What cash decided, read off DISK, for use in a failure message.

    These assertions fail only on a genuinely loaded machine -- not under a
    synthetic CPU load, not when the file runs alone beside a full parallel
    suite, only inside one. That makes each real occurrence expensive: it is
    rare, it is not summonable, and without this it reports a bare count that
    cannot distinguish "was split when it should not have been" from "per-call
    caching never engaged". The split store answers exactly that, and reading a
    JSON file perturbs nothing -- unlike `enable_debug()`, which changes the
    timing that decides the verdict in the first place.
    """
    import json
    store = work_dir / ".cash" / "_loop_split.json"
    if not store.exists():
        entries = list((work_dir / ".cash").glob("*.meta")) if (
            work_dir / ".cash").exists() else []
        return f"no split verdict recorded; {len(entries)} cache entries on disk"
    try:
        return f"split store = {json.dumps(json.loads(store.read_text()))}"
    except (OSError, ValueError) as e:
        return f"split store unreadable: {e}"


def _seed_split_verdict(work_dir, loop_src, k=_K):
    """Write a k-verdict into the store FILE, before the kernel starts.

    Written from the test process, on disk, rather than by executing anything
    in the kernel -- and that is the whole point. A ``store_history=False``
    exec is not a free observation: cash cannot find it among the notebook's
    cells, so it reconstructs state around it. Measured, seeding that way: the
    store held the correct verdict and the hash matched the one the simulator
    wanted, and the loop STILL re-ran 124/124 on both later runs. The probe was
    the thing breaking the split it was there to observe.

    Mirrors ``loop_split.LoopSplitStore``: same filename, same schema version,
    and the same identity function (sha256 of ``ast.unparse`` of the loop node),
    recomputed here rather than imported so a change to either side shows up as
    a failing test instead of a silently agreeing one.
    """
    import ast
    import hashlib
    import json

    node = ast.parse(loop_src).body[0]
    source_hash = hashlib.sha256(ast.unparse(node).encode("utf-8")).hexdigest()
    cache_dir = work_dir / ".cash"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "_loop_split.json").write_text(
        json.dumps({"version": 1, "splits": {source_hash: k}}), encoding="utf-8")
    return source_hash


def test_a_recorded_split_costs_only_its_head(nb_runner, tmp_path):
    """Given a verdict, a settled rerun runs the head and nothing else.

    This is the SPLIT itself -- the simulator half -- and it is the part that
    must never regress. The verdict is seeded rather than learned, because
    learning is a wall-clock measurement and therefore hostage to whatever else
    is on the machine; see
    test_the_runtime_eventually_learns_to_split_a_cheap_loop for that half.

    Splitting the two was not tidiness. As one test it failed ~4 of 7 full
    parallel runs with 124/124: under 40+ processes on 32 cores the kernel was
    descheduled during the 5-iteration probe, measured per_iter crossed
    ``for_handler._SPLIT_MAX_ITER_SEC`` (6ms, against a body of ~1ms plus
    ~2.2ms decomposition overhead -- under 2x headroom), and no verdict was
    ever recorded. Instrumented at the moment of failure, the store file did
    not exist on disk. So the flake was in the LEARNING, while the assertion
    that kept failing was about the SPLIT.
    """
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    # Seeded BEFORE the kernel starts, so the store loads it on first use and
    # no out-of-band execution ever happens (see _seed_split_verdict).
    _seed_split_verdict(tmp_path, _loop())
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N, f"baseline did not run all {_N} items: {cold}"

    nb_runner.run_cell(LOOP_CELL)          # splits; the tail is a cold miss
    after_first_split = _n(counter)

    nb_runner.run_cell(LOOP_CELL)          # the tail now hits
    steady = _n(counter) - after_first_split
    assert steady <= _K, (
        f"a settled rerun re-ran {steady}/{_N} calls with a verdict already "
        f"recorded; expected at most the {_K}-iteration head. The split is "
        "applied by the simulator, so this is a simulator-side regression."
    )
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_a_learned_loop_splits_and_then_costs_only_its_head(nb_runner, tmp_path):
    """Three runs, because learning costs one:

    * **run 1** -- the loop is unknown, so it decomposes and stores
      per-iteration entries. The verdict is recorded at the end.
    * **run 2** -- the verdict now exists, so the loop splits. The TAIL has
      never been stored (run 1 wrote per-iteration entries, not a tail), so
      it is a cold miss and runs. Measured: 123 of 124.
    * **run 3 onward** -- the tail hits; only the head runs.

    That one-run lag is inherent to learning from measurement: nothing can
    know a loop is worth splitting until it has been seen. Asserting on run 2
    would have looked like a failure of the split rather than the cost of
    learning, which is why this test is explicit about all three.

    Before this landed, every rerun measured 124 -- the whole loop, forever.

    Runs with the split ceiling pinned (``SETUP_PINNED_CEILING``). Against the
    real 6ms ceiling this body has under 2x headroom, and a kernel descheduled
    mid-probe measures it as expensive and silently declines; that failed ~2 of
    4 full parallel runs even with four retries stacked on top. The retries are
    gone now -- what they papered over was the machine, not the policy.
    """
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter, setup=SETUP_PINNED_CEILING))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N, f"baseline did not run all {_N} items: {cold}"

    # Run 2: the verdict recorded at the end of run 1 now exists, so the loop
    # splits. The tail has never been stored, so it is a cold miss and runs.
    before = _n(counter)
    nb_runner.run_cell(LOOP_CELL)
    learning = _n(counter) - before
    assert learning > _K, (
        f"run 2 re-ran only {learning}/{_N} calls. Expected the tail to be a "
        f"cold miss here -- run 1 stored per-iteration entries, not a tail. "
        f"Something served this loop that should not have."
    )

    # Run 3: the tail hits, so only the head runs. With the ceiling pinned a
    # decline is no longer a scheduling coin flip, so this is asserted once
    # rather than retried -- if it fails, no verdict was recorded and the loop
    # caches nothing at all, which is slower than running with cash off.
    before = _n(counter)
    nb_runner.run_cell(LOOP_CELL)
    steady = _n(counter) - before
    assert steady <= _K, (
        f"run 3 re-ran {steady}/{_N} calls; expected at most the {_K}-iteration "
        f"head. Run 2 ran {learning}, so the loop decomposed but no split "
        f"verdict survived into run 3."
    )
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_the_split_preserves_the_value(nb_runner, tmp_path):
    """Head + tail must produce exactly what the undivided loop produced,
    including the leaked loop variable."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter) + [
        "print('SUM', sum(out), 'LAST', out[-1], 'T', t, 'LEN', len(out))"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = nb_runner.get_output(6)
    assert expected == (f"SUM {sum(range(1, _N + 1)) * 10} LAST {_N * 10} "
                        f"T {_N} LEN {_N}"), expected

    nb_runner.run_cell(LOOP_CELL)
    nb_runner.run_cell(6)
    assert nb_runner.get_output(6) == expected, (
        f"the split changed the loop's result:\n  cold: {expected}\n"
        f"  warm: {nb_runner.get_output(6)}"
    )


def test_a_dependency_edit_still_recomputes(nb_runner, tmp_path):
    """Negative control. Without it every assertion above would pass just as
    happily if the split were caching too eagerly."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.set_cell_source(DEF_CELL, _compute_def(counter).replace("v * 10", "v * 20"))
    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"dependency edit only re-ran {warm}/{_N} calls"
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_an_expensive_body_is_never_split(nb_runner, tmp_path):
    """The gate protecting CAS-259's incremental reuse.

    A loop whose calls clear the cost floor is served by per-call caching,
    which gives INCREMENTAL reuse -- append one item, re-run one call. A
    split's tail is all-or-nothing: its key covers the whole remaining
    iterable, so an append re-runs it entirely. Splitting an expensive-bodied
    loop would silently trade a better mechanism for a worse one.

    Measured via an append: incremental reuse costs 1 real call; a split
    would cost the whole tail.
    """
    counter = tmp_path / "calls.log"
    n = 60
    nb_runner.create_notebook(_cells(counter, n=n, ms=20.0))   # 20ms >> 3ms floor
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == n

    nb_runner.set_cell_source(LOOP_CELL, _loop(n + 1))
    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == 1, (
        f"append re-ran {warm}/{n + 1} calls, expected 1. An expensive-bodied "
        "loop must keep per-call incremental reuse (CAS-259) rather than "
        f"being split into an all-or-nothing tail. [{_why(tmp_path)}]"
    )


def test_oracle_no_caching(nb_runner, tmp_path):
    """Cash off: the same shape re-runs everything. Pins the harness, so a
    zero above means reuse rather than a broken counter."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter, setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"oracle only re-ran {warm}/{_N}; the harness is not measuring real work"
