"""Adversarial probes: exotic assignment & syntax channels.

Each probe attacks one DISTINCT mechanism by which a binding/mutation can be
expressed in Python syntax that cash's AST analysis might mis-model:

1.  chained assignment producing a fresh shared object (a = b = [..]), later
    cell mutates through one alias -> is the pair restored consistently on an
    isolated re-run?
2.  NESTED 1:1 literal tuple unpack alias ((p,(q,)) = (x,(y,))) -> alias map
    depth (flat case is fixed).
3.  augmented assignment on a SLICE target (lst[1:3] += [..]) -> in-place
    mutation via subscript-slice channel, isolated re-run idempotence.
4.  nested subscript self-reference (d['a']['b'] += 1) -> idempotence.
5.  exec("q = 42") producing a variable invisible to AST analysis -> edit
    invalidation of the downstream reader.
6.  globals()['gv'] = 5 producing a variable invisible to AST analysis ->
    edit invalidation of the downstream reader.
7.  f-string with a mutating side effect f"{stack.pop()}" -> idempotence.
8.  semicolon-joined multi-statement line with self-modifying ops ->
    statement splitting + reset.
9.  match statement capture bindings as cell outputs -> downstream edit
    invalidation.
10. annotated assignment (x: int = 5) + PEP 695 `type` alias statement ->
    edit invalidation.
11. same variable assigned twice self-referentially in one cell -> idempotence.
12. immediately-invoked lambda mutating an upstream list -> idempotence
    (object-protocol reset family, lambda channel).
13. `global` statement inside a function defined AND called in the same cell
    -> idempotence (regression variant of the CAS-68..80 family).
14. very long cell (60 sequential statements) -> statement-splitting
    integrity, cache stability on unchanged re-run, edit propagation.
15. backslash line continuations inside an expression -> edit invalidation.

Known limitations deliberately NOT re-filed: multi-target swap a,b = b,a
(CAS-43), walrus-as-receiver / attribute / container-element / ternary alias
(CAS-61), exhausted generators (CAS-50).
"""

import pytest

pytestmark = [pytest.mark.timeout(90)]


def _rerun(nb_runner, setup, cell, expect):
    """run_all, assert, then isolated re-run of cell 2 must be idempotent."""
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- 1. chained assignment creating a fresh shared object ------------------

def test_chained_assign_fresh_object_pair_rerun(nb_runner):
    # a = b = [..] creates ONE object bound to two names in the producer cell.
    # Cell 2 mutates through a. On isolated re-run BOTH names must be restored
    # to the shared base (same values AND still the same object), else the
    # printed line changes across re-runs.
    _rerun(
        nb_runner,
        "a = b = [1, 2, 3]",
        "a.append(99)\nprint(a, b, a is b)",
        "[1, 2, 3, 99] [1, 2, 3, 99] True",
    )


# --- 2. nested 1:1 literal tuple unpack alias -------------------------------

def test_nested_tuple_unpack_alias_rerun(nb_runner):
    # Flat (y,) = (x,) alias is fixed; probe one nesting level deeper.
    _rerun(
        nb_runner,
        "x = [1, 2]\ny = [3, 4]",
        "(p, (q,)) = (x, (y,))\nq.append(9)\nprint(y)",
        "[3, 4, 9]",
    )


# --- 3. augmented assignment on a slice target -------------------------------

def test_augassign_slice_target_rerun(nb_runner):
    # lst[1:3] += [99]  ==>  [1, 2, 3, 99, 4, 5]; re-run must not grow again.
    _rerun(
        nb_runner,
        "lst = [1, 2, 3, 4, 5]",
        "lst[1:3] += [99]\nprint(lst)",
        "[1, 2, 3, 99, 4, 5]",
    )


# --- 4. nested subscript self-reference --------------------------------------

def test_nested_subscript_selfref_rerun(nb_runner):
    _rerun(
        nb_runner,
        "cfg = {'m': {'count': 1}}",
        "cfg['m']['count'] += 1\nprint(cfg['m']['count'])",
        "2",
    )


# --- 5. exec() creating a variable invisible to AST analysis -----------------

def test_exec_created_var_edit_invalidation(nb_runner):
    nb_runner.create_notebook([
        'exec("q = 42")',
        "print(f'q={q}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "q=42" in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.set_cell_source(1, 'exec("q = 100")')
    nb_runner.run_all()
    assert "q=100" in nb_runner.get_output(2), (
        f"stale exec-produced value: {nb_runner.get_output(2)!r}"
    )


# --- 6. globals()['gv'] assignment invisible to AST analysis -----------------

def test_globals_subscript_assign_edit_invalidation(nb_runner):
    nb_runner.create_notebook([
        "globals()['gv'] = 5",
        "print(f'gv={gv}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "gv=5" in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.set_cell_source(1, "globals()['gv'] = 50")
    nb_runner.run_all()
    assert "gv=50" in nb_runner.get_output(2), (
        f"stale globals()-assigned value: {nb_runner.get_output(2)!r}"
    )


# --- 7. f-string side effect --------------------------------------------------

def test_fstring_mutating_side_effect_rerun(nb_runner):
    _rerun(
        nb_runner,
        "stack = [1, 2, 3]",
        "msg = f'top={stack.pop()}'\nprint(msg)\nprint(stack)",
        "top=3",
    )
    # also assert the container itself did not shrink twice
    assert "[1, 2]" in nb_runner.get_output(2), (
        f"stack not reset: {nb_runner.get_output(2)!r}"
    )


# --- 8. semicolon-joined multi-statement line --------------------------------

def test_semicolon_multistmt_selfmod_rerun(nb_runner):
    _rerun(
        nb_runner,
        "items = [1, 2, 3]",
        "v = items.pop(); items.append(v * 10); print(items)",
        "[1, 2, 30]",
    )


# --- 9. match statement capture bindings --------------------------------------

def test_match_capture_bindings_edit_invalidation(nb_runner):
    nb_runner.create_notebook([
        "pt = (1, 2)",
        "match pt:\n    case (mx, my):\n        ms = mx + my\nprint(f'ms={ms}')",
        "print(f'double={ms * 2}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ms=3" in nb_runner.get_output(2)
    assert "double=6" in nb_runner.get_output(3)
    nb_runner.set_cell_source(1, "pt = (10, 20)")
    nb_runner.run_all()
    assert "ms=30" in nb_runner.get_output(2), (
        f"match-capture cell stale: {nb_runner.get_output(2)!r}"
    )
    assert "double=60" in nb_runner.get_output(3), (
        f"downstream of match-capture stale: {nb_runner.get_output(3)!r}"
    )


# --- 10. annotated assignment + PEP 695 type alias ---------------------------

def test_annassign_and_type_alias_edit_invalidation(nb_runner):
    nb_runner.create_notebook([
        "x: int = 5",
        "type Vec = list[int]\ny: int = x * 2\nprint(f'y={y}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y=10" in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.set_cell_source(1, "x: int = 7")
    nb_runner.run_all()
    assert "y=14" in nb_runner.get_output(2), (
        f"annotated-assign downstream stale: {nb_runner.get_output(2)!r}"
    )


# --- 11. same var assigned twice self-referentially in one cell --------------

def test_same_var_twice_selfref_rerun(nb_runner):
    _rerun(
        nb_runner,
        "w = 5",
        "w = w + 1\nw = w + 1\nprint(w)",
        "7",
    )


# --- 12. immediately-invoked lambda mutating an upstream list ----------------

def test_lambda_iife_mutation_rerun(nb_runner):
    _rerun(
        nb_runner,
        "bag = [1, 2]",
        "(lambda: bag.append(3))()\nprint(bag)",
        "[1, 2, 3]",
    )


# --- 13. global statement in a function defined+called in one cell -----------

def test_global_stmt_func_same_cell_rerun(nb_runner):
    _rerun(
        nb_runner,
        "counter = 0",
        "def bump():\n    global counter\n    counter += 1\nbump()\nprint(counter)",
        "1",
    )


# --- 14. very long cell: statement-splitting integrity -----------------------

def test_long_cell_statement_splitting_integrity(nb_runner):
    # 60 sequential dependent statements in ONE cell.
    stmts = ["v0 = base + 0"]
    stmts += [f"v{i} = v{i - 1} + 1" for i in range(1, 60)]
    stmts.append("print(f'v59={v59}')")
    big_cell = "\n".join(stmts)
    nb_runner.create_notebook(["base = 1", big_cell])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "v59=60" in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"

    # unchanged re-run: value identical and no CELL_CHANGED
    nb_runner.run_all()
    assert "v59=60" in nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(2)
    assert "[CELL_CHANGED]" not in raw, f"long cell falsely flagged changed: {raw[:400]}"

    # edit upstream: whole chain must recompute
    nb_runner.set_cell_source(1, "base = 100")
    nb_runner.run_all()
    assert "v59=159" in nb_runner.get_output(2), (
        f"long-cell chain stale after edit: {nb_runner.get_output(2)!r}"
    )


# --- 15. backslash line continuation ------------------------------------------

def test_backslash_continuation_edit_invalidation(nb_runner):
    nb_runner.create_notebook([
        "p = 2\nq = 3",
        "total = p + \\\n    q\nprint(f'total={total}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=5" in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.set_cell_source(1, "p = 2\nq = 30")
    nb_runner.run_all()
    assert "total=32" in nb_runner.get_output(2), (
        f"continuation-line cell stale: {nb_runner.get_output(2)!r}"
    )
