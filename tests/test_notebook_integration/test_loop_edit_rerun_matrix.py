"""Loop shape x edit-kind matrix: measures INCREMENTAL REUSE and INVALIDATION
PRECISION, not restore.

CAS-259, CAS-261 and CAS-262 all survived a fully green 832-file suite for the
same structural reason: every existing test of a loop shape asserted only an
*unchanged* rerun costs zero real calls. That proves restore works. It says
nothing about whether appending one item recomputes only that item, whether
reordering leaves an unrelated cell alone, or whether an edit to a cell the
loop does not read leaves the loop alone. This module adds that missing
dimension: four loop shapes x five kinds of edit, every cell measured as
counted real executions of the loop body's callee, not inferred from printed
values, badges or timings.

Instrument, not endorsement
----------------------------
``compute()`` appends one byte to an external counter file every time it
actually runs. That file is the ONLY thing any assertion here reads to decide
"did real work happen." This is deliberately the same technique CAS-259's own
regression guard uses (``test_accumulator_single_statement_append_incremental.
py``), and it is a side effect the callee performs purely to serve as this
suite's instrument -- CAS-260 is the record that a callee's own side effects
are not, in general, something cash's caching is obliged to preserve on a
cache hit. Using one here to COUNT executions is not an endorsement of that
gap; it is the least-invasive way to get a ground truth number that survives
being read by a value-blind cache.

Design note discovered while building this: ``compute()`` must NOT read a
plain module-global free variable (e.g. ``return v * factor`` with `factor`
defined in another cell). Measured directly: an otherwise-identical shape
went from 0 real calls on an unchanged rerun to 3 (every call, every time),
regardless of which edit was applied -- call-level caching (CAS-243) simply
never cached calls to a callee with that shape of free-variable read, in
either direction (store or restore). That is a real, reproducible behaviour
(compute() closing over `factor` vs. baking the multiplier directly into the
function body is the one-line difference), but it is a DIFFERENT gap from the
three this module targets and would have confounded every case if left in
place -- it deserves its own investigation, not a footnote inside this one.
The negative-control "edit a real dependency" case therefore redefines
``compute()`` itself (the function the loop actually calls), which every
existing convention in this suite already relies on for exactly this
purpose, rather than introducing a free-variable read.

Shapes (seed placement matters -- see "hoisting" below)
---------------------------------------------------------
A - single-statement accumulator: ``out = []`` / ``out.append(compute(t))``
B - two-statement body: ``v = compute(t)`` / ``out.append(v)``
C - accumulator fold: ``s = 0`` / ``s += compute(t)``
D - subscript store: ``d = {}`` / ``d[t] = compute(t)``

Edits, per shape
-----------------
1. unchanged rerun         -- expect 0 real calls
2. append an item          -- expect 1 real call (only the new item)
3. reorder the items       -- expect 0 real calls
4. edit an unrelated cell  -- expect 0 real calls (the loop does not read it)
5. edit a real dependency  -- expect full recompute (negative control)

Size variant (shape A only)
-----------------------------
CAS-259 (fixed on this branch, commit 61d920d) and CAS-261 (still open) sit on
opposite sides of one boundary: ``for_handler._should_execute_loop_as_single_
unit`` treats a loop as one big cacheable unit once ``n * body_stmts * 0.008s
>= 1.0s`` (n >= 125 for a 1-statement body); below that it decomposes, and
``call_unit._COST_FLOOR_S`` (10ms) only *stores* a call that individually took
that long. A loop can be under BOTH floors at once -- decomposed (so no
whole-unit caching) yet each call too cheap to store individually (so no
per-call caching either) -- while its aggregate cost is well worth caching.
CAS-261's own measured reproduction is n=124 iterations at 5ms/call (0.62s
total): 0 warm calls before CAS-259 touched anything, 124 warm calls after.

The task brief suggested "large (150 items, cheap body)" for this variant.
Measured directly: n=150 at 5ms/call clears the single-unit threshold
(n >= 125) and 61d920d's fix (restoring ``force_outputs`` for that branch)
already covers its unchanged-rerun case cleanly (0 warm calls) -- it is NOT
CAS-261's gap. This module instead uses **n=100** at 5ms/call: solidly inside
the (50, 125) sub-floor band on BOTH the baseline count and after an append
(100 -> 101 stays inside the band; 124 -> 125 would have crossed the
single-unit threshold mid-test and confounded the append measurement with the
boundary itself). The deviation from the brief's literal 150 is deliberate
and is exactly what this module is for: measure the real boundary, not the
suggested one. (n=150's own measured behaviour -- clean on unchanged rerun,
but a full recompute on append/reorder because the single-unit's cache key
still keys on the whole iterable, and a CLEAN pass on the unrelated-edit case
-- is reported in the task report rather than pinned here, to keep this
module's shape list matching the brief's four-shapes-plus-one-size-variant
structure; see the report for the numbers and why they matter for CAS-262.)

CAS-262 confound, found while calibrating this module
---------------------------------------------------------
The existing xfailed probe for CAS-262 (``test_zzprobe_codeleads.py::
test_unrelated_upstream_edit_reruns_loop``) uses a bare-expression loop body
(``results.append(i * i)``, no function call). Measured directly here: with a
CALLABLE body, call-level caching independently absorbs the loop's real work
even in configurations where the *statement* still gets marked stale --
several adjacent-seed and hoisted-seed configurations that were expected to
plausibly reproduce CAS-262 (by shape, per the ticket's own attribution
table) instead measured 0 real calls on an unrelated-edit case. The ONE
configuration in this module where an unrelated edit costs real recomputation
is shape A's large/uncached-size variant -- and there it is indistinguishable
from CAS-261: the SAME configuration already fails to cache a plain unchanged
rerun, so "the unrelated edit re-ran everything" and "nothing was ever cached
in the first place" cannot be told apart by a real-call count. This module's
unrelated-edit case is therefore attributed to CAS-261 everywhere it fails,
never solely to CAS-262 -- see the report for the full reasoning and for why
that narrows, rather than contradicts, CAS-262's stated scope.

xfail policy
------------
A failure that matches CAS-261's or CAS-262's own described mechanism is
marked ``xfail(strict=True)`` naming the ticket, with the measured numbers in
the reason, so a fix shows up loudly as an unexpected XPASS. A failure that
does not match either ticket is left as a plain, failing assertion -- not
xfailed, not silenced, not "fixed" here. Per the task brief: a matrix of
tests that cannot fail is worse than no matrix, and an unattributed red test
is the flag that sends the next investigation to the right place instead of
requiring another hand audit.

``get_output`` / ``run_cell`` on ``nb_runner`` are 1-based cell indices.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above call_unit._COST_FLOOR_S (10ms) so shapes A-small/B/C/D exercise
# real per-call caching rather than the "too cheap to store" floor.
_SLEEP_SMALL = 0.3
# Below call_unit._COST_FLOOR_S; n=100 is below the single-unit threshold
# (n>=125 for a 1-stmt body) -- this is CAS-261's uncached band.
_SLEEP_LARGE = 0.005
_N_LARGE = 100

SETUP = "import cash\n%cash_on\nimport time"
SETUP_OFF = "import cash\nimport time"

UNRELATED = "unrelated = 1"
UNRELATED_EDITED = "unrelated = 1\nz = 1"

SMALL_BASE = "[1, 2, 3]"
SMALL_APPEND = "[1, 2, 3, 4]"
SMALL_REORDER = "[3, 1, 2]"

LARGE_BASE = f"list(range({_N_LARGE}))"
LARGE_APPEND = f"list(range({_N_LARGE + 1}))"
LARGE_REORDER = f"[1, 0] + list(range(2, {_N_LARGE}))"


def _n(path):
    """Real compute() calls so far: one byte per call, read from disk."""
    return len(path.read_bytes()) if path.exists() else 0


def _compute_def(counter, sleep, mult=10):
    return (
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        f"    time.sleep({sleep})\n"
        f"    return v * {mult}"
    )


# ---------------------------------------------------------------------------
# Shape bodies. A/B/C/D bodies each end in their own print so the loop cell
# is self-contained; hoisted shapes keep the seed in an EARLIER cell (the
# whole-cell-caching-confounder trap: seed + loop in one cell makes the trap,
# the fix, and an undirected control all measure identically).
# ---------------------------------------------------------------------------

def _a_body(items_expr):
    return f"for t in {items_expr}:\n    out.append(compute(t))\nprint('OUT', out)"


def _b_body(items_expr):
    return f"for t in {items_expr}:\n    v = compute(t)\n    out.append(v)\nprint('OUT', out)"


def _c_body(items_expr):
    return f"for t in {items_expr}:\n    s += compute(t)\nprint('S', s)"


def _d_body(items_expr):
    return f"for t in {items_expr}:\n    d[t] = compute(t)\nprint('D', sorted(d.items()))"


def _hoisted_cells(counter, seed_src, loop_src, setup=SETUP):
    """SETUP(1) / UNRELATED(2) / compute def(3) / seed(4) / loop(5).

    Seed is hoisted into its own cell, one earlier than the loop -- NOT the
    loop's immediately-preceding sibling in the same cell.
    """
    return [setup, UNRELATED, _compute_def(counter, _SLEEP_SMALL), seed_src, loop_src]


LOOP_CELL_HOISTED = 5
UNRELATED_CELL = 2
COMPUTE_DEF_CELL = 3


def _adjacent_cell(counter, items_expr, setup=SETUP):
    """SETUP(1) / UNRELATED(2) / compute def(3) / seed+loop(4).

    Shape A's large variant deliberately keeps the seed ADJACENT (same cell,
    immediately preceding the `for`) -- that adjacency is part of what CAS-259
    and CAS-261 are about, not a confounder to avoid here.
    """
    return [setup, UNRELATED, _compute_def(counter, _SLEEP_LARGE),
            f"out = []\n{_a_body(items_expr)}"]


LOOP_CELL_ADJACENT = 4


# ===========================================================================
# Shape A, small (n=3, 0.3s/call, HOISTED seed)
# ===========================================================================

def test_shape_a_small_unchanged_rerun(nb_runner, tmp_path):
    """Mutation that would break this: deleting call_unit's cache-hit
    short-circuit (the branch that serves a stored result instead of
    invoking the wrapped callee) makes this jump from 0 to 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3, f"baseline did not run all 3 items: {cold}"
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)

    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_append(nb_runner, tmp_path):
    """Mutation: in call_unit.py, make the discriminator ignore the call's
    argument value (CAS-259's own mutation) -- this must jump from 1 to 3+."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _a_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1 (only the new item)"
    assert "OUT [10, 20, 30, 40]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_reorder(nb_runner, tmp_path):
    """Mutation: strip call-level caching's order-independence (key the call
    by loop position instead of argument value) -- this must jump to 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _a_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0"
    assert "OUT [30, 10, 20]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_unrelated_edit(nb_runner, tmp_path):
    """Mutation: in the upstream lineage simulation, make ANY upstream cell
    edit mark every downstream loop stale regardless of dependency -- if this
    is already broken (CAS-262) that mutation is invisible; measured here as
    a clean pass (0 real calls), though the seed cell's OWN statement is
    still needlessly (harmlessly) re-executed -- see the report."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, (
        f"unrelated upstream edit re-ran {warm} calls, expected 0 "
        f"(raw: {nb_runner.get_raw_output(LOOP_CELL_HOISTED)[:400]!r})"
    )
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_dependency_edit(nb_runner, tmp_path):
    """Negative control: redefines compute() itself (a genuine dependency of
    every call in the loop). Mutation: make the lineage tracker ignore a
    callee's own source-hash when building the call's cache key -- this must
    then WRONGLY stay at 0 instead of recomputing all 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3 (full recompute)"
    assert "OUT [20, 40, 60]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_oracle_no_caching(nb_runner, tmp_path):
    """Cash OFF, identical shape: proves the 0/1/0 results above are cash's
    doing, not some property of the shape (an idempotent for-loop reruns
    everything with no caching at all, every single time)."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3, f"oracle baseline did not run all 3 items: {cold}"

    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3 (no caching)"


# ===========================================================================
# Shape A, LARGE (n=100, 5ms/call, ADJACENT seed) -- CAS-261's uncached band
# ===========================================================================

@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-261: n=100 at 5ms/call is below BOTH the single-unit threshold "
        "(n>=125) and call_unit's 10ms store floor. Measured on HEAD "
        "(a465608): 100/100 real calls on a plain unchanged rerun -- nothing "
        "is ever stored, so the 'restored' rerun is actually a full "
        "recompute. strict=True: must XPASS loudly once CAS-261 is fixed."
    ),
)
def test_shape_a_large_unchanged_rerun(nb_runner, tmp_path):
    """Mutation: raising call_unit._COST_FLOOR_S above 5ms would make this
    fail even harder (more of the small-shape tests would join it);
    lowering it below 5ms, or aligning the two floors per CAS-261's own
    recommended fix, would flip this to a genuine pass."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE, f"baseline did not run all {_N_LARGE} items: {cold}"

    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0 (measured: {warm}/{_N_LARGE})"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-261: same uncached band -- nothing was ever stored for the 100 "
        "baseline items, so appending one more cannot reuse any of them "
        "either. Measured on HEAD (a465608): 101/101 real calls (full "
        "recompute), not the 1 real reuse would cost."
    ),
)
def test_shape_a_large_append(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above -- see
    that test's mutation note."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(LOOP_CELL_ADJACENT, f"out = []\n{_a_body(LARGE_APPEND)}")
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1 (only the new item)"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-261: same uncached band -- nothing to reuse regardless of "
        "order. Measured on HEAD (a465608): 100/100 real calls (full "
        "recompute) instead of the 0 a truly order-independent call cache "
        "would cost."
    ),
)
def test_shape_a_large_reorder(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(LOOP_CELL_ADJACENT, f"out = []\n{_a_body(LARGE_REORDER)}")
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-261, NOT independent evidence of CAS-262: this exact "
        "configuration already fails to cache a plain unchanged rerun (see "
        "test_shape_a_large_unchanged_rerun, also 100/100), so a full "
        "recompute here cannot be told apart from 'nothing was ever cached "
        "in the first place.' Measured on HEAD (a465608): 100/100 real "
        "calls. See the module docstring and task report for why this is "
        "attributed to CAS-261 rather than CAS-262."
    ),
)
def test_shape_a_large_unrelated_edit(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0 (measured: {warm}/{_N_LARGE})"


def test_shape_a_large_dependency_edit(nb_runner, tmp_path):
    """Negative control -- must pass regardless of CAS-261: a real dependency
    change has to recompute even in the uncached band."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_LARGE, mult=20))
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == _N_LARGE, f"real dependency edit only re-ran {warm} calls, expected {_N_LARGE}"


def test_shape_a_large_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE, setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == _N_LARGE, f"oracle only re-ran {warm} calls on unchanged rerun; expected {_N_LARGE}"


# ===========================================================================
# Shape B: two-statement body (v = compute(t); out.append(v)), HOISTED seed
# ===========================================================================

def test_shape_b_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_b_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _b_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "OUT [10, 20, 30, 40]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_b_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _b_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0"


def test_shape_b_unrelated_edit(nb_runner, tmp_path):
    """CAS-262's own ticket cites the TWO-statement spelling as already
    broken on `main`, independent of this branch -- but that ticket's own
    repro has no function call in the body. Measured here (with a call):
    clean pass."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_b_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"


def test_shape_b_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"


# ===========================================================================
# Shape C: accumulator fold (s += compute(t)), HOISTED seed
# ===========================================================================

def test_shape_c_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_c_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _c_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "S 100" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _c_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0"
    assert "S 60" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_unrelated_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_c_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"
    assert "S 120" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"


# ===========================================================================
# Shape D: subscript store (d[t] = compute(t)), HOISTED seed
# ===========================================================================

def test_shape_d_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_d_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _d_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "[(1, 10), (2, 20), (3, 30), (4, 40)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _d_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0"
    assert "[(1, 10), (2, 20), (3, 30)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_unrelated_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_d_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"
    assert "[(1, 20), (2, 40), (3, 60)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"
