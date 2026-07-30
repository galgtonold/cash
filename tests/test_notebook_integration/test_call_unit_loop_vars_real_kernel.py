"""Real-kernel proof that `loop_vars` reaches an intercepted call (CAS-243).

The wiring under test: ``ForLoopHandler._process_one_iteration`` pushes the
current iteration's loop vars onto ``StatementProcessor``'s stack
(``loop_vars_scope``), and ``CallUnit._build_key`` reads them back through
``CallCache``'s ``loop_vars_provider`` at the moment an intercepted call is
actually invoked. Every other test of this feature either drives
``call_cache_key`` directly (proves the discrimination logic, not that the
values ever arrive) or runs in-process with the real IPython shell mocked out
(``tests/test_notebook/test_call_unit_loop_vars_wiring.py``). This file is
the one arm that goes through an actual Jupyter kernel end to end -- the
closest thing to what a user's notebook does.

The motivating shape: ``fetch_next(conn)`` inside a loop, where ``conn`` is a
bare ``Name`` whose lineage never moves (it is the same object all three
iterations) and there is no computed argument expression for ``arg_digests``
to hash. Without ``loop_vars``, all three iterations mint the identical key,
and the second and third iterations serve the first iteration's value --
wrong on the very first run, no pre-existing cache required.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above CallUnit's cost floor (10ms), or the call is never stored at all and
# every assertion below would hold whether or not this feature works.
_SLEEP = 0.15


def _defs(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "counter = {'n': 0}\n"
        "conn = 'db-connection'\n"
        "def fetch_next(conn):\n"
        "    counter['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return counter['n']\n"
    )


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


# `results[t] = fetch_next(conn)` -- a subscript-assignment body, not
# `out.append(fetch_next(conn))`. The append shape is a bare `Expr(Call)`
# body and can be routed through the accumulator single-unit fast path
# instead of real per-iteration decomposition; the assignment shape forces
# the per-iteration path that pushes/pops `loop_vars`.
_LOOP = (
    "results = {}\n"
    "# @cash:cache-calls\n"
    "for t in [1, 2, 3]:\n"
    "    results[t] = fetch_next(conn)\n"
    "print('RESULTS', sorted(results.items()))\n"
)


def test_hidden_state_call_is_correct_on_the_first_run(nb_runner, tmp_path):
    """No pre-existing cache: each iteration must still get its own value.

    This is the CAS-243 bug reproduced live: ``conn``'s lineage is constant
    across all three iterations and ``fetch_next`` takes no other argument,
    so pre-``loop_vars`` all three iterations keyed identically and
    iterations 2/3 served iteration 1's cached ``1`` -- ``RESULTS
    [(1, 1), (2, 1), (3, 1)]`` instead of the correct
    ``[(1, 1), (2, 2), (3, 3)]``.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_defs(log), _LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "RESULTS [(1, 1), (2, 2), (3, 3)]" in output, output
    assert _n(log) == 3, (
        f"expected exactly 3 real fetch_next() executions on the first run, got {_n(log)}"
    )


def test_hidden_state_call_hits_cache_on_rerun(nb_runner, tmp_path):
    """Re-running the loop cell must replay the same three values from cache
    -- zero further real calls -- proving the per-iteration keys are stable
    across runs, not just distinct within one.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_defs(log), _LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3

    nb_runner.run_cell(2)
    output = nb_runner.get_output(2)
    assert "RESULTS [(1, 1), (2, 2), (3, 3)]" in output, output
    assert _n(log) == 3, "compute() re-ran on an unchanged rerun of the loop cell"


# --------------------------------------------------------- sampled-equal loop vars
#
# `call_cache_key` hashes `loop_vars` VALUES with `compute_hash_full`, not the
# sampling `compute_hash`. `object_hashing._hash_collection` reduces any
# list/tuple over 200 elements to its first 5 + last 5 elements; two 300-
# element tuples that agree on both ends but differ in the middle are
# `compute_hash`-equal while being genuinely different values. `for_handler.py`
# already learned this exact lesson for the loop variable's own LINEAGE, two
# lines above where it builds `iteration_context`
# (`_process_one_iteration`): "a sampled hash keyed two iterations over
# arrays that agreed in the sample onto ONE entry - wrong result on the first
# run." Before this fix, `loop_vars` repeated that mistake -- newly live
# (rather than merely present) the moment `loop_vars` stopped being `{}` in
# production.

# Above `_hash_collection`'s 200-element sampling threshold (object_hashing.py).
_SAMPLED_TAIL = (10, 11, 12, 13, 14)
_SAMPLED_HEAD = (1, 2, 3, 4, 5)


def _long_tuple(fill):
    return _SAMPLED_HEAD + (fill,) * 290 + _SAMPLED_TAIL


def _sampling_defs(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "counter = {'n': 0}\n"
        "conn = 'db-connection'\n"
        "def fetch_next(conn):\n"
        "    counter['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return counter['n']\n"
        f"A = {_long_tuple(0)!r}\n"
        f"B = {_long_tuple(1)!r}\n"
    )


# Seed cell separate from the loop cell (like `test_cache_calls_directive.py`'s
# append tests) so `results = []` is NOT the loop's immediately-preceding
# sibling statement within the SAME cell -- that would make the loop eligible
# for `cacheable_accumulator_loop`'s narrow single-unit fast path
# (`control_structures/processor.py`), which routes the whole loop through
# ONE cache entry and never reaches per-iteration decomposition (and so never
# reaches `loop_vars` at all). Only `t` is a loop target here -- deliberately
# no `enumerate()`/index variable, which would itself discriminate the key
# (small ints hash trivially, no sampling involved) and mask the exact bug
# under test.
_SAMPLING_SEED = "results = []\n"
_SAMPLING_LOOP = (
    "# @cash:cache-calls\n"
    "for t in [A, B]:\n"
    "    results.append(fetch_next(conn))\n"
    "print('OUT', results)\n"
)


def test_sampled_equal_loop_var_values_still_get_distinct_keys(nb_runner, tmp_path):
    """The bug: `A` and `B` are `compute_hash`-equal (sampled) but genuinely
    different (`compute_hash_full`-different). `fetch_next(conn)` has no
    other discriminator (`conn` is constant, no computed argument), so if
    `loop_vars` hashed with the sampling `compute_hash`, iteration 2 (loop
    var `B`) would collide with iteration 1's key (loop var `A`) and be
    served iteration 1's cached ``1`` -- ``OUT [1, 1]`` instead of the
    correct ``OUT [1, 2]``, wrong on the first run with no pre-existing
    cache.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_sampling_defs(log), _SAMPLING_SEED, _SAMPLING_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(3)
    assert "OUT [1, 2]" in output, output
    assert _n(log) == 2, (
        f"expected exactly 2 real fetch_next() executions, got {_n(log)}"
    )


def test_sampled_equal_loop_var_values_match_the_no_cash_oracle(nb_runner, tmp_path):
    """Independent confirmation that ``[1, 2]`` is the correct answer, not an
    assumption -- the identical code, run with cash off entirely.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_sampling_defs(log), _SAMPLING_SEED, _SAMPLING_LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "OUT [1, 2]" in nb_runner.get_output(3)


# --------------------------------------------------------- reused loop-target names
#
# A SECOND cost-vs-correctness bug found in review, distinct from the
# sampling one above: the fix for THAT bug looked the loop var's digest up in
# `variable_lineage`, a flat dict keyed only by name and never popped.
# `for_handler.py` writes it once per iteration, so a nested (or sequential
# sibling) loop reusing an outer loop's target name overwrites the entry --
# and nothing restores it once the inner/sibling loop finishes. A call in the
# OUTER loop's body, positioned after the reuse, read the INNER loop's last
# value instead of the outer loop's current one. First-run wrongness, no
# pre-existing cache required -- the shipped fix instead carries the digest
# through `StatementProcessor`'s `loop_vars_scope` push/pop stack, which has
# the scope discipline `variable_lineage` lacks.


def _reused_name_defs(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "counter = {'n': 0}\n"
        "conn = 'db-connection'\n"
        "def fetch_next(conn):\n"
        "    counter['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return counter['n']\n"
    )


# The outer loop's body is TWO statements (the inner `for`, then the
# append) -- never eligible for `cacheable_accumulator_loop`'s single-unit
# fast path (which requires the body to be EXACTLY one bare `acc.append(...)`
# statement), so this always reaches the real per-iteration path regardless
# of whether the seed sits in the same cell.
_NESTED_REUSE_LOOP = (
    "outer_results = []\n"
    "# @cash:cache-calls\n"
    "for t in ['A', 'B']:\n"
    "    for t in [1, 2]:\n"
    "        pass\n"
    "    outer_results.append(fetch_next(conn))\n"
    "print('OUT', outer_results)\n"
)


def test_nested_loop_reusing_the_target_name_gets_correct_values(nb_runner, tmp_path):
    """``for t in ['A','B']: for t in [1,2]: pass; outer_results.append(fetch_next(conn))``.

    The inner loop's body is a no-op (``pass``); it exists purely to reuse
    ``t`` and get popped again before the call runs.

    Both outer iterations bind ``t`` fresh at the top of THEIR OWN
    ``_process_one_iteration`` (outer ``t='A'`` then, later, ``t='B'``), but
    by the time the call executes, the inner loop has ALREADY run to
    completion and overwritten ``variable_lineage['t']`` with its own last
    value (``hash(2)``) in the pre-fix (``variable_lineage``-sourced)
    design. Both outer iterations therefore looked that entry up AFTER the
    inner loop clobbered it, both got the identical ``hash(2)`` digest for
    ``t``, and both calls collapsed onto ONE key -- serving outer iteration
    1's cached value (``1``) to outer iteration 2 as well: ``[1, 1]``.
    ``fetch_next``'s counter genuinely advances only once. The fix (a
    push/pop stack instead of a flat dict) restores the outer's own digest
    the moment the inner loop's pushes are popped, giving the correct
    ``[1, 2]``.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _NESTED_REUSE_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "OUT [1, 2]" in output, output
    assert _n(log) == 2, (
        f"expected exactly 2 real fetch_next() executions, got {_n(log)}"
    )


def test_nested_loop_reusing_the_target_name_matches_the_no_cash_oracle(nb_runner, tmp_path):
    """Independent confirmation that ``[1, 2]`` is correct -- cash off entirely."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _NESTED_REUSE_LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "OUT [1, 2]" in nb_runner.get_output(2)


# Sibling variant: the reused name comes from TWO SEQUENTIAL inner loops (not
# one nested inside the other), with the call after the second -- same
# staleness class, different timing. Under the pre-fix design,
# `variable_lineage['t']` would be left holding the SECOND sibling loop's
# last value (`hash(20)`) rather than the outer loop's current one.
_SIBLING_REUSE_LOOP = (
    "sibling_results = []\n"
    "# @cash:cache-calls\n"
    "for t in ['A', 'B']:\n"
    "    for t in [1, 2]:\n"
    "        pass\n"
    "    for t in [10, 20]:\n"
    "        pass\n"
    "    sibling_results.append(fetch_next(conn))\n"
    "print('OUT', sibling_results)\n"
)


def test_sibling_loops_reusing_the_target_name_get_correct_values(nb_runner, tmp_path):
    """Two SEQUENTIAL inner loops both reusing ``t``, call after the second."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _SIBLING_REUSE_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "OUT [1, 2]" in output, output
    assert _n(log) == 2, (
        f"expected exactly 2 real fetch_next() executions, got {_n(log)}"
    )


def test_sibling_loops_reusing_the_target_name_match_the_no_cash_oracle(nb_runner, tmp_path):
    """Independent confirmation that ``[1, 2]`` is correct -- cash off entirely."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _SIBLING_REUSE_LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "OUT [1, 2]" in nb_runner.get_output(2)


# --------------------------------------------------------- sampled _cash_lineage_hash
#
# A THIRD bug found in review, distinct from the two above: `for_handler.py`
# prefers a loop variable's own `val._cash_lineage_hash` when present, over
# computing `compute_hash_full(val)` fresh. That attribute is not always a
# full-content hash -- `update_mutated_variable_lineages`
# (`control_structures/helpers.py`) derives a MUTATED accumulator's lineage
# from the SAMPLING `compute_hash` and `LineageStore.record` stamps that
# result onto the object as `_cash_lineage_hash`. Two DataFrames that agree
# on shape, dtypes, and `head(5)` (what `compute_hash` samples for a
# DataFrame) but differ further down hash EQUAL under that sampled lineage
# while being genuinely different values -- so if this loop variable is later
# bound to each of them in turn, preferring the attribute for the CALL KEY
# collapses iteration 2 onto iteration 1's cached value. Same lesson as the
# sampling bug two sections up, one layer further out: it arrives through an
# attribute instead of being passed directly, but a sampled hash is still
# never sound as a key discriminator.
#
# The fix: `for_handler.py` computes `loop_var_digests[name]` from
# `compute_hash_full(val)` directly, never through `_cash_lineage_hash`.
# `variable_lineage[name]` keeps preferring the attribute -- that write wants
# PROVENANCE (did this binding come from the same upstream computation?),
# which is what `_cash_lineage_hash` is FOR; the call key wants CONTENT
# (are two iterations' bindings the same value?), which only a full hash
# answers soundly. Conflating the two consumers is what let a sampled hash
# back in.


def _sampled_lineage_defs(log):
    return (
        "import time, pathlib\n"
        "import pandas as pd\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "counter = {'n': 0}\n"
        "conn = 'db-connection'\n"
        "def fetch_next(conn):\n"
        "    counter['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return counter['n']\n"
        "df_a = pd.DataFrame({'x': range(300)})\n"
        "df_b = pd.DataFrame({'x': range(300)})\n"
    )


# Mutates BOTH df_a and df_b inside the SAME for-loop (one iteration), via
# plain subscript assignment (`df['x'] = ...`) -- the standard, easily-
# detected in-place-mutation shape (mirrors `results[k] = v` elsewhere in
# this file). Both mutations sharing ONE loop's source text and iterable
# matters: `update_mutated_variable_lineages` combines `loop_code_hash`
# (from the loop's own text) with each variable's own sampled `value_hash`,
# so both DataFrames must go through the IDENTICAL loop code for their
# resulting `_cash_lineage_hash` values to collide when their sampled
# content also agrees. `_vals_b` differs from `_vals_a` ONLY at index 10 --
# well past `head(5)`, so shape/dtypes/`head(5)` (what `compute_hash`
# samples for a DataFrame) stay identical between the two.
_MUTATE_BOTH_DATAFRAMES = (
    "for _ in range(1):\n"
    "    df_a['x'] = list(range(300))\n"
    "    _vals_b = list(range(300))\n"
    "    _vals_b[10] = -999999\n"
    "    df_b['x'] = _vals_b\n"
)

# Two-statement body (`n = len(df)` then the append), never eligible for
# `cacheable_accumulator_loop`'s single-unit fast path.
_SAMPLED_LINEAGE_LOOP = (
    "SL2 = []\n"
    "# @cash:cache-calls\n"
    "for df in [df_a, df_b]:\n"
    "    n = len(df)\n"
    "    SL2.append(fetch_next(conn))\n"
    "print('SL2', SL2, 'runs', counter['n'])\n"
)


def test_sampled_cash_lineage_hash_on_loop_var_still_gets_distinct_keys(nb_runner, tmp_path):
    """``df_a``/``df_b`` are ``compute_hash``-equal (sampled -- same shape,
    dtypes, ``head(5)``) but ``compute_hash_full``-different (row 10) AFTER
    both are mutated inside one shared for-loop, which is what stamps a
    matching, sampled-derived ``_cash_lineage_hash`` onto each. ``fetch_next``
    has no other discriminator (``conn`` is constant), so if the loop-vars
    leg preferred that attribute, iteration 2 (``df_b``) would collide with
    iteration 1's key (``df_a``) and be served iteration 1's cached ``1`` --
    ``SL2 [1, 1]`` instead of the correct ``SL2 [1, 2]``, wrong on the first
    run with no pre-existing cache.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _sampled_lineage_defs(log), _MUTATE_BOTH_DATAFRAMES, _SAMPLED_LINEAGE_LOOP,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(3)
    assert "SL2 [1, 2]" in output, output
    assert _n(log) == 2, (
        f"expected exactly 2 real fetch_next() executions, got {_n(log)}"
    )


def test_sampled_cash_lineage_hash_on_loop_var_matches_the_no_cash_oracle(nb_runner, tmp_path):
    """Independent confirmation that ``[1, 2]`` is correct -- cash off entirely."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _sampled_lineage_defs(log), _MUTATE_BOTH_DATAFRAMES, _SAMPLED_LINEAGE_LOOP,
    ])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "SL2 [1, 2]" in nb_runner.get_output(3)


# --------------------------------------------------------- reused name, call INSIDE the reuse (CAS-257 defect 1)
#
# A THIRD variant of the reused-name shape, distinct from both sections
# above: there, the call sits AFTER the inner loop that reuses the outer
# target name -- by the time it runs, the inner loop's own pushes onto
# `_call_unit_loop_vars`/`_call_unit_loop_var_digests` have already been
# popped, so only the outer scope is active and its digest resolves cleanly.
# Here the call sits INSIDE the inner loop, so BOTH scopes are
# simultaneously active on the stack at the moment the key is built.
# `current_loop_vars()` returns only the stack's TOP (the inner scope's own,
# already-pre-merged-with-parent dict -- see that method's docstring), and
# `current_loop_var_digests()` merges the whole digest stack by BARE name,
# innermost winning -- both means the reused name 'q' resolves to only the
# INNER scope's value/digest; the outer iteration has no slot in the key at
# all. Two different outer iterations that both see the same inner sequence
# are therefore indistinguishable: pre-fix, this collapsed 4 genuinely
# distinct (outer, inner) pairs onto 2 keys.
_NESTED_REUSE_INSIDE_LOOP = (
    "acc_inside = []\n"
    "# @cash:cache-calls\n"
    "for q in ['p', 'r']:\n"
    "    for q in [7, 8]:\n"
    "        acc_inside.append(fetch_next(conn))\n"
    "print('INSIDE', acc_inside)\n"
)


def test_call_inside_a_name_reusing_inner_loop_gets_correct_values(nb_runner, tmp_path):
    """``for q in ['p','r']: for q in [7,8]: acc_inside.append(fetch_next(conn))``.

    Pre-fix: both outer iterations see the identical inner cycle (7 then 8),
    and with the reused name 'q' resolving to only the inner scope's
    value/digest, both outer passes mint the SAME two keys -- iteration 3
    (outer='r', inner=7) hits iteration 1's entry (outer='p', inner=7), and
    iteration 4 hits iteration 2's. ``acc_inside`` reads ``[1, 2, 1, 2]``
    with only 2 real ``fetch_next()`` calls, instead of 4 genuinely distinct
    calls giving ``[1, 2, 3, 4]``.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _NESTED_REUSE_INSIDE_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "INSIDE [1, 2, 3, 4]" in output, output
    assert _n(log) == 4, (
        f"expected exactly 4 real fetch_next() executions, got {_n(log)}"
    )


def test_call_inside_a_name_reusing_inner_loop_matches_the_no_cash_oracle(nb_runner, tmp_path):
    """Independent confirmation that ``[1, 2, 3, 4]`` is correct -- cash off entirely."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _NESTED_REUSE_INSIDE_LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "INSIDE [1, 2, 3, 4]" in nb_runner.get_output(2)


def test_call_inside_a_name_reusing_inner_loop_reuses_cache_on_rerun(nb_runner, tmp_path):
    """Re-running the identical loop cell must replay the same four values
    from cache -- zero further real calls. Establishes that the depth-keyed
    slots this fix introduces are stable ACROSS RUNS, not merely distinct
    within one: depth is the stack length at the moment of each push, which
    is fixed by this statement's lexical loop nesting and identical on every
    execution of it, so the same (depth, name) pair is produced -- and looked
    up in the cache -- on the rerun as on the first run.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_reused_name_defs(log), _NESTED_REUSE_INSIDE_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 4

    nb_runner.run_cell(2)
    output = nb_runner.get_output(2)
    assert "INSIDE [1, 2, 3, 4]" in output, output
    assert _n(log) == 4, "fetch_next() re-ran on an unchanged rerun of the loop cell"
