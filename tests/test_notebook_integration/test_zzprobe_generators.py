"""Adversarial probes: generators, iterators & lazy objects crossing cells.

Known limitation NOT re-filed: CAS-50 (exhausted generator/stateful iterator on
plain ISOLATED re-run of the consumer cell). These probes attack adjacent
mechanisms instead:

 1. test_genexpr_cross_cell_second_run_all      — gen in A, consumed in B, run_all twice
 2. test_genexpr_unrelated_edit_run_all         — edit unrelated cell C, run_all: does the
                                                  skip logic resurrect an exhausted gen?
 3. test_map_filter_cross_cell_second_run_all   — unpicklable lazy builtins (lambda inside)
 4. test_zip_enumerate_chain_second_run_all     — picklable C iterators crossing cells
 5. test_cycle_cross_cell_second_run_all        — infinite picklable iterator (advanced state)
 6. test_tee_cross_cell_second_run_all          — itertools.tee pair consumed in two cells
 7. test_send_protocol_cross_cell_second_run_all— primed coroutine advanced via send()
 8. test_gen_close_cross_cell_second_run_all    — gen closed in consumer cell
 9. test_dict_view_alias_upstream_mutation_edit — d.keys() view aliasing across restore + edit
10. test_reversed_later_mutated_list_second_run_all — reversed() over a later-appended list
11. test_file_handle_iteration_second_run_all   — open file handle iterated in a later cell
12. test_iterator_restart_persist_graceful      — kernel restart + persist with unpicklable gen
13. test_genfunc_downstream_instance_edit       — gen FUNCTION edited; instance made in a
                                                  separate cell from consumption
14. test_half_consumed_iterator_isolated_rerun_persist — half in B, rest in C, isolated
                                                  re-run of C under persist (CAS-50-adjacent:
                                                  does the statement cache rescue it?)

Oracle: cash promises run_all() twice -> identical outputs (plain Python agrees
here because the producer cell re-executes and recreates the iterator), and
edits invalidate downstream. Assertions are on printed values via get_output().
"""

import pytest

pytestmark = [pytest.mark.timeout(90)]


# ---------------------------------------------------------------------------
# run_all twice: producer cell must not be skipped while the consumer re-runs
# on an exhausted/advanced iterator.
# ---------------------------------------------------------------------------

def test_genexpr_cross_cell_second_run_all(nb_runner):
    """Generator expression created in A, consumed in B. run_all twice must
    print the same values (A re-runs -> fresh generator)."""
    nb_runner.create_notebook([
        "g = (i * i for i in range(4))",
        "vals = list(g)\nprint(vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[0, 1, 4, 9]" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    assert "[0, 1, 4, 9]" in nb_runner.get_output(2), (
        f"second run_all must reproduce the first: {nb_runner.get_output(2)!r}"
    )


def test_genexpr_unrelated_edit_run_all(nb_runner):
    """Gen in A, consumed in B, unrelated cell C edited, then run_all.
    Skipping A (unchanged) while re-executing B would consume an exhausted
    generator -> []."""
    nb_runner.create_notebook([
        "g = (i * i for i in range(4))",
        "vals = list(g)\nprint(vals)",
        "z = 1\nprint(z)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[0, 1, 4, 9]" in nb_runner.get_output(2)
    nb_runner.set_cell_source(3, "z = 2\nprint(z)")
    nb_runner.run_all()
    assert "[0, 1, 4, 9]" in nb_runner.get_output(2), (
        f"after unrelated edit, run_all must keep B's value: {nb_runner.get_output(2)!r}"
    )
    assert "2" in nb_runner.get_output(3)


def test_map_filter_cross_cell_second_run_all(nb_runner):
    """map/filter objects with lambdas (unpicklable) crossing cells."""
    nb_runner.create_notebook([
        "m = map(str, filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))",
        "out = list(m)\nprint(out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "['2', '4']" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    assert "['2', '4']" in nb_runner.get_output(2), (
        f"second run_all: {nb_runner.get_output(2)!r}"
    )


def test_zip_enumerate_chain_second_run_all(nb_runner):
    """zip/enumerate/chain (picklable C iterators) crossing cells; a stale
    RESTORE of an advanced iterator would change the printed lists."""
    nb_runner.create_notebook([
        "import itertools\n"
        "pairs = zip([1, 2, 3], 'abc')\n"
        "en = enumerate('xy')\n"
        "ch = itertools.chain([1, 2], [3])",
        "print(list(pairs))\nprint(list(en))\nprint(list(ch))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out1 = nb_runner.get_output(2)
    assert "[(1, 'a'), (2, 'b'), (3, 'c')]" in out1, out1
    assert "[(0, 'x'), (1, 'y')]" in out1, out1
    assert "[1, 2, 3]" in out1, out1
    nb_runner.run_all()
    out2 = nb_runner.get_output(2)
    assert "[(1, 'a'), (2, 'b'), (3, 'c')]" in out2, out2
    assert "[(0, 'x'), (1, 'y')]" in out2, out2
    assert "[1, 2, 3]" in out2, out2


def test_cycle_cross_cell_second_run_all(nb_runner):
    """itertools.cycle is infinite AND picklable-with-state: restoring the
    advanced cycle (or skipping the producer) shifts the printed window."""
    nb_runner.create_notebook([
        "import itertools\ncyc = itertools.cycle([1, 2, 3])",
        "vals = [next(cyc) for _ in range(4)]\nprint(vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[1, 2, 3, 1]" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    assert "[1, 2, 3, 1]" in nb_runner.get_output(2), (
        f"advanced cycle leaked into second run: {nb_runner.get_output(2)!r}"
    )


def test_tee_cross_cell_second_run_all(nb_runner):
    """tee pair created in A (multi-target), each branch consumed in its own
    cell. Both branches must replay identically on the second run_all."""
    nb_runner.create_notebook([
        "import itertools\nsrc = iter([1, 2, 3])\nta, tb = itertools.tee(src)",
        "a_vals = list(ta)\nprint(a_vals)",
        "b_vals = list(tb)\nprint(b_vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[1, 2, 3]" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "[1, 2, 3]" in nb_runner.get_output(3), nb_runner.get_output(3)
    nb_runner.run_all()
    assert "[1, 2, 3]" in nb_runner.get_output(2), (
        f"tee branch a on second run: {nb_runner.get_output(2)!r}"
    )
    assert "[1, 2, 3]" in nb_runner.get_output(3), (
        f"tee branch b on second run: {nb_runner.get_output(3)!r}"
    )


def test_send_protocol_cross_cell_second_run_all(nb_runner):
    """Primed coroutine-style generator; a skipped producer cell leaves an
    advanced coroutine, so send(5) would print 10 instead of 5."""
    nb_runner.create_notebook([
        "def acc():\n"
        "    total = 0\n"
        "    while True:\n"
        "        x = yield total\n"
        "        total += x",
        "a = acc()\nnext(a)",
        "r = a.send(5)\nprint(r)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip().endswith("5"), nb_runner.get_output(3)
    nb_runner.run_all()
    out = nb_runner.get_output(3).strip()
    assert out.endswith("5") and not out.endswith("15") and "10" not in out, (
        f"send() on a stale advanced coroutine: {out!r}"
    )


def test_gen_close_cross_cell_second_run_all(nb_runner):
    """Consumer closes the generator. Second run_all with a skipped producer
    would call next() on a closed generator -> StopIteration error output."""
    nb_runner.create_notebook([
        "g = (i for i in range(5))",
        "first = next(g)\ng.close()\nprint(first)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "0" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "0" in out and "StopIteration" not in out, (
        f"closed generator leaked into second run: {out!r}"
    )


# ---------------------------------------------------------------------------
# View aliasing + edit invalidation
# ---------------------------------------------------------------------------

def test_dict_view_alias_upstream_mutation_edit(nb_runner):
    """ks = d.keys() aliases d live. After editing the mutating cell, the
    downstream print of the VIEW must reflect the new mutation. If cash
    restores d as a NEW object (breaking the alias) or serves ks stale, the
    view shows the old key set."""
    nb_runner.create_notebook([
        "d = {'a': 1}",
        "ks = d.keys()",
        "d['b'] = 2",
        "print(sorted(ks))\nprint(len(ks))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "['a', 'b']" in out1, out1
    nb_runner.set_cell_source(3, "d['b'] = 2\nd['c'] = 3")
    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert "['a', 'b', 'c']" in out2 and "3" in out2, (
        f"view must reflect the edited mutation: {out2!r}"
    )


def test_reversed_later_mutated_list_second_run_all(nb_runner):
    """reversed() snapshots the start index at creation; appending afterwards
    does not extend it -> [3, 2, 1] (plain semantics). Second run_all must
    reproduce that, not [4, 3, 2, 1] (view recreated after the mutation) nor
    an empty list (exhausted iterator)."""
    nb_runner.create_notebook([
        "lst = [1, 2, 3]",
        "r = reversed(lst)",
        "lst.append(4)",
        "print(list(r))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[3, 2, 1]" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_all()
    assert "[3, 2, 1]" in nb_runner.get_output(4), (
        f"second run_all diverged: {nb_runner.get_output(4)!r}"
    )


# ---------------------------------------------------------------------------
# Real resources / persistence
# ---------------------------------------------------------------------------

def test_file_handle_iteration_second_run_all(nb_runner):
    """Open file handle created in A, iterated in B. Skipping A on the second
    run_all leaves fh at EOF -> B prints []."""
    nb_runner.create_notebook([
        "with open('probe_data.txt', 'w') as f:\n"
        "    f.write('a\\nb\\nc\\n')\n"
        "fh = open('probe_data.txt')",
        "lines = [l.strip() for l in fh]\nprint(lines)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "['a', 'b', 'c']" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_all()
    assert "['a', 'b', 'c']" in nb_runner.get_output(2), (
        f"EOF handle leaked into second run: {nb_runner.get_output(2)!r}"
    )


def test_iterator_restart_persist_graceful(nb_runner):
    """Kernel restart under persist with an unpicklable generator variable.
    The gen cannot be persisted -- restart must recompute (or restore the
    downstream list) and print correct values with no traceback."""
    nb_runner.create_notebook([
        "import time\ng = (i * i for i in range(4))\ntime.sleep(0.02)",
        "vals = list(g)\nprint(vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "[0, 1, 4, 9]" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    raw1 = nb_runner.get_raw_output(1)
    raw2 = nb_runner.get_raw_output(2)
    assert "[0, 1, 4, 9]" in out, f"post-restart value wrong: {out!r}"
    for raw in (raw1, raw2):
        assert "Traceback" not in raw and "StopIteration" not in raw, (
            f"post-restart error output: {raw[:400]!r}"
        )


# ---------------------------------------------------------------------------
# Generator functions & partial consumption
# ---------------------------------------------------------------------------

def test_genfunc_downstream_instance_edit(nb_runner):
    """Generator FUNCTION defined in A, instance created in B, consumed in C.
    Editing the function body must propagate through the instance cell to the
    consumer."""
    nb_runner.create_notebook([
        "def gen(n):\n    for i in range(n):\n        yield i * 2",
        "g = gen(3)",
        "vals = list(g)\nprint(vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[0, 2, 4]" in nb_runner.get_output(3), nb_runner.get_output(3)
    nb_runner.set_cell_source(
        1, "def gen(n):\n    for i in range(n):\n        yield i * 3"
    )
    nb_runner.run_all()
    assert "[0, 3, 6]" in nb_runner.get_output(3), (
        f"edited generator function did not propagate: {nb_runner.get_output(3)!r}"
    )


def test_half_consumed_iterator_isolated_rerun_persist(nb_runner):
    """Half of the iterator consumed in B, rest in C; ISOLATED re-run of C
    under persist. CAS-50-adjacent: with the statement cached (persist on),
    can cash serve rest=[3, 4, 5] instead of re-executing on the exhausted
    iterator?"""
    nb_runner.create_notebook([
        "it = iter(range(6))",
        "first3 = [next(it) for _ in range(3)]\nprint(first3)",
        "rest = list(it)\nprint(rest)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "[0, 1, 2]" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "[3, 4, 5]" in nb_runner.get_output(3), nb_runner.get_output(3)
    nb_runner.run_cell(3)
    assert "[3, 4, 5]" in nb_runner.get_output(3), (
        f"isolated re-run of the tail consumer: {nb_runner.get_output(3)!r}"
    )
