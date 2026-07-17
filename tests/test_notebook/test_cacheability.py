"""Tests for cacheability.py — the merged pure-AST analysis module.

All assertions target :class:`StatementAnalysis` fields or
``skip_reasons()``.  No direct visitor or detector API remains; this
file replaces ``test_mutation_detector.py`` and ``test_side_effects.py``.
"""
from __future__ import annotations

import ast

import pytest

from cash.notebook.cacheability import (
    KNOWN_PURE_METHODS,
    MutationInfo,
    SideEffectInfo,
    StatementAnalysis,
    alias_mutation_sources,
    aliased_sources,
    analyze_statement,
    bare_alias_targets,
    crossref_reassigned_vars,
    subscript_view_bindings,
    function_arg_mutations,
    function_global_mutations,
    stateful_self_functions,
    stateful_closure_vars,
    partial_arg_mutations,
    mutating_partials,
    reduce_free_mutations,
    object_protocol_mutations,
    params_mutated_in_function,
    standalone_call_arg_targets,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
    assigned_method_call_receivers,
    selfref_inplace_write_vars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyze(code: str) -> StatementAnalysis:
    return analyze_statement(code, None)


def _analyze_with_tree(code: str) -> StatementAnalysis:
    tree = ast.parse(code)
    return analyze_statement(code, tree)


# ---------------------------------------------------------------------------
# Mutation detection — all_mutated_vars
# ---------------------------------------------------------------------------

class TestAllMutatedVars:
    """Behaviour previously tested through MutationDetector.detect_mutations /
    get_mutated_variables."""

    def test_list_append(self):
        a = _analyze("lst.append(42)")
        assert 'lst' in a.all_mutated_vars

    def test_list_extend(self):
        a = _analyze("data.extend([1, 2, 3])")
        assert 'data' in a.all_mutated_vars

    def test_list_sort(self):
        a = _analyze("items.sort(key=lambda x: x.name)")
        assert 'items' in a.all_mutated_vars

    def test_list_insert(self):
        a = _analyze("lst.insert(0, 'first')")
        assert 'lst' in a.all_mutated_vars

    def test_dict_update(self):
        a = _analyze("config.update({'key': 'value'})")
        assert 'config' in a.all_mutated_vars

    def test_dict_setitem(self):
        a = _analyze("d['key'] = value")
        assert 'd' in a.all_mutated_vars

    def test_dict_pop(self):
        a = _analyze("d.pop('key')")
        assert 'd' in a.all_mutated_vars

    def test_set_add(self):
        a = _analyze("seen.add(item)")
        assert 'seen' in a.all_mutated_vars

    def test_augmented_add(self):
        a = _analyze("counter += 1")
        assert 'counter' in a.all_mutated_vars

    def test_augmented_multiply(self):
        a = _analyze("arr *= 2")
        assert 'arr' in a.all_mutated_vars

    def test_pandas_fillna_inplace(self):
        a = _analyze("df.fillna(0, inplace=True)")
        assert 'df' in a.all_mutated_vars

    def test_pandas_dropna_inplace(self):
        a = _analyze("df.dropna(inplace=True)")
        assert 'df' in a.all_mutated_vars

    def test_pandas_sort_inplace(self):
        a = _analyze("df.sort_values('col', inplace=True)")
        assert 'df' in a.all_mutated_vars

    def test_pandas_no_inplace(self):
        """Without inplace=True, it's not a mutation."""
        a = _analyze("df.fillna(0)")
        assert 'df' not in a.all_mutated_vars

    def test_pandas_inplace_false(self):
        """Explicit inplace=False is not a mutation."""
        a = _analyze("df.fillna(0, inplace=False)")
        assert 'df' not in a.all_mutated_vars

    def test_numpy_out_kwarg(self):
        """numpy ufunc out= writes its target in place: out=a mutates a."""
        a = _analyze("np.add(a, 10, out=a)")
        assert 'a' in a.all_mutated_vars

    def test_numpy_out_kwarg_distinct_target(self):
        """out= names the mutated array even when it is not also an input."""
        a = _analyze("np.multiply(x, y, out=result)")
        assert 'result' in a.all_mutated_vars

    def test_numpy_out_kwarg_tuple(self):
        """Multi-output ufuncs take a tuple: out=(a, b) mutates both."""
        a = _analyze("np.divmod(x, y, out=(q, r))")
        assert {'q', 'r'} <= a.all_mutated_vars

    def test_numpy_out_kwarg_subscript(self):
        """out=arr[1:] mutates the base array arr."""
        a = _analyze("np.add(arr, 1, out=arr[1:])")
        assert 'arr' in a.all_mutated_vars

    def test_out_kwarg_captured_result(self):
        """out= mutation is detected even when the call's result is assigned."""
        a = _analyze("b = np.add(a, 10, out=a)")
        assert 'a' in a.all_mutated_vars

    def test_method_mutation_in_list_comprehension(self):
        # CAS-67: a known-mutating method inside a comprehension element.
        a = _analyze("r = [base.append(0) for _ in range(2)]")
        assert 'base' in a.all_mutated_vars

    def test_method_mutation_in_generator_expr(self):
        a = _analyze("r = list(base.append(0) for _ in range(2))")
        assert 'base' in a.all_mutated_vars

    def test_method_mutation_in_dict_comprehension(self):
        a = _analyze("r = {i: base.append(0) for i in range(2)}")
        assert 'base' in a.all_mutated_vars

    def test_method_mutation_in_fstring(self):
        a = _analyze("s = f'{base.append(0)}'")
        assert 'base' in a.all_mutated_vars

    def test_method_mutation_captured_result(self):
        # r = d.pop(k) still mutates d (append/pop/... always mutate).
        a = _analyze("r = d.pop('k')")
        assert 'd' in a.all_mutated_vars

    def test_pure_method_in_comprehension_not_flagged(self):
        # A non-mutating method (copy/head) inside a comprehension is not a mutation.
        a = _analyze("r = [base.copy() for _ in range(2)]")
        assert 'base' not in a.all_mutated_vars

    def test_subscript_assign(self):
        a = _analyze("arr[0] = 100")
        assert 'arr' in a.all_mutated_vars

    def test_nested_subscript(self):
        a = _analyze("matrix[0][1] = 42")
        assert 'matrix' in a.all_mutated_vars

    def test_attribute_assign(self):
        a = _analyze("obj.name = 'new_name'")
        assert 'obj' in a.all_mutated_vars

    def test_del_subscript(self):
        a = _analyze("del d['key']")
        assert 'd' in a.all_mutated_vars

    def test_tuple_unpack_subscript_target(self):
        # CAS-56: subscript writes nested in a tuple target mutate their bases.
        a = _analyze("df['a'], df['b'] = df['b'], df['a']")
        assert 'df' in a.all_mutated_vars

    def test_tuple_unpack_mixed_bases(self):
        a = _analyze("d['x'], lst[0] = 1, 2")
        assert {'d', 'lst'} <= a.all_mutated_vars

    def test_list_target_attribute(self):
        a = _analyze("[obj.x, obj.y] = (1, 2)")
        assert 'obj' in a.all_mutated_vars

    def test_tuple_unpack_plain_names_no_mutation(self):
        a = _analyze("x, y = 1, 2")
        assert len(a.all_mutated_vars) == 0

    def test_multiple_mutations(self):
        code = "lst.append(1)\nlst.append(2)\nd['key'] = 'val'\ncounter += 1"
        a = _analyze(code)
        assert {'lst', 'd', 'counter'} <= a.all_mutated_vars

    def test_pure_assignment(self):
        a = _analyze("x = 42")
        assert len(a.all_mutated_vars) == 0

    def test_function_call_no_mutation(self):
        a = _analyze("result = sorted(lst)")
        assert len(a.all_mutated_vars) == 0

    def test_method_call_with_assignment(self):
        """Result captured — original isn't mutated."""
        a = _analyze("new_df = df.fillna(0)")
        assert 'df' not in a.all_mutated_vars

    def test_syntax_error_returns_empty(self):
        a = _analyze("def :")
        assert len(a.all_mutated_vars) == 0

    def test_pre_parsed_tree(self):
        code = "lst.append(42)"
        a = _analyze_with_tree(code)
        assert 'lst' in a.all_mutated_vars


# ---------------------------------------------------------------------------
# Standalone method-mutation receivers (lineage-bump + skip-cache trigger)
# ---------------------------------------------------------------------------

class TestStandaloneMethodMutationReceivers:
    """``standalone_method_mutation_receivers`` returns the base variable of a
    top-level bare-``Expr`` method call that mutates its receiver in place
    (``lst.append(x)``, ``df.dropna(inplace=True)``).  These receivers are
    routed into the statement's *output* set so their lineage is bumped — the
    runtime and the upstream simulation both consume this helper so the bump
    stays in sync.  Pure standalone calls (``df.head()``) and assignments
    (``r = lst.append(x)``) are excluded.
    """

    @staticmethod
    def _receivers(code: str) -> frozenset[str]:
        return standalone_method_mutation_receivers(ast.parse(code))

    def test_list_append(self):
        assert self._receivers("lst.append(42)") == {'lst'}

    def test_set_add(self):
        assert self._receivers("box.add(10)") == {'box'}

    def test_attribute_receiver_append(self):
        assert self._receivers("box.items.append(10)") == {'box'}

    def test_dict_update(self):
        assert self._receivers("config.update({'k': 'v'})") == {'config'}

    def test_pandas_inplace_true(self):
        assert self._receivers("df.dropna(inplace=True)") == {'df'}

    def test_pure_method_excluded(self):
        """A non-mutating standalone call must NOT bump its receiver."""
        assert self._receivers("df.head()") == frozenset()

    def test_pandas_inplace_false_excluded(self):
        assert self._receivers("df.fillna(0, inplace=False)") == frozenset()

    def test_assignment_excluded(self):
        """Captured result — not a bare standalone mutation statement."""
        assert self._receivers("r = lst.append(42)") == frozenset()

    def test_loop_body_mutation_excluded(self):
        """Mutations inside a loop are handled by the loop-mutation path, not here."""
        assert self._receivers("for x in xs:\n    lst.append(x)") == frozenset()

    def test_function_body_mutation_excluded(self):
        code = "def f():\n    lst.append(1)"
        assert self._receivers(code) == frozenset()

    def test_multiple_receivers(self):
        assert self._receivers("a.append(1)\nb.add(2)") == {'a', 'b'}

    def test_numpy_out_kwarg(self):
        """numpy out= writes its target in place -> bump the out target ``a``.
        (``np`` also appears here because ``add`` is in MUTATING_METHODS via
        ``set.add``; that pre-existing artifact is harmless — the runtime/sim
        loops skip module receivers.)"""
        assert 'a' in self._receivers("np.add(a, 10, out=a)")

    def test_numpy_out_kwarg_tuple(self):
        assert self._receivers("np.divmod(x, y, out=(q, r))") == {'q', 'r'}

    def test_numpy_out_kwarg_distinct_target(self):
        assert self._receivers("np.multiply(x, y, out=result)") == {'result'}

    def test_none_tree(self):
        assert standalone_method_mutation_receivers(None) == frozenset()


class TestStandaloneMethodCallReceivers:
    """``standalone_method_call_receivers`` returns ``(base, method)`` for EVERY
    top-level bare-``Expr`` method call — the broad candidate set the runtime
    classifies (mutating / pure / observe) for the precise broad extension.
    Unlike the narrow helper it does not filter by method name.
    """

    @staticmethod
    def _calls(code: str) -> frozenset:
        return standalone_method_call_receivers(ast.parse(code))

    def test_pure_method_included_as_candidate(self):
        # narrow helper drops df.head(); the broad candidate set keeps it
        assert self._calls("df.head()") == {('df', 'head')}

    def test_mutating_method_included(self):
        assert self._calls("box.add(10)") == {('box', 'add')}

    def test_attribute_receiver(self):
        assert self._calls("box.items.append(10)") == {('box', 'append')}

    def test_custom_method(self):
        assert self._calls("bus.on(handler)") == {('bus', 'on')}

    def test_chained_call_uses_outer_method_and_root_base(self):
        assert self._calls("df.groupby('a').sum()") == {('df', 'sum')}

    def test_assignment_excluded(self):
        assert self._calls("r = df.head()") == frozenset()

    def test_bare_name_call_excluded(self):
        assert self._calls("print(x)") == frozenset()

    def test_loop_and_function_bodies_excluded(self):
        assert self._calls("for i in xs:\n    a.append(i)") == frozenset()
        assert self._calls("def f():\n    a.append(1)") == frozenset()

    def test_multiple(self):
        assert self._calls("a.foo()\nb.bar(1)") == {('a', 'foo'), ('b', 'bar')}

    def test_numpy_out_kwarg_candidate(self):
        """out= target is a candidate (method label 'out='); np.add receiver too."""
        assert self._calls("np.add(a, 10, out=a)") == {('np', 'add'), ('a', 'out=')}

    def test_none_tree(self):
        assert standalone_method_call_receivers(None) == frozenset()


class TestAssignedMethodCallReceivers:
    """``assigned_method_call_receivers`` returns ``(base, method)`` for method
    calls on the RHS of a top-level assignment — the captured-return set the
    runtime routes as a draw when the receiver is identity-coupled (CAS-199).
    The dual of ``standalone_method_call_receivers``: assignments here, bare
    ``Expr`` there.
    """

    @staticmethod
    def _calls(code: str) -> frozenset:
        return assigned_method_call_receivers(ast.parse(code))

    def test_tuple_unpack_captured(self):
        assert self._calls("counts, bins, patches = ax.hist(data)") == {('ax', 'hist')}

    def test_single_target_captured(self):
        assert self._calls("h = ax.hist(data, bins=11)") == {('ax', 'hist')}

    def test_annotated_assignment_captured(self):
        assert self._calls("h: object = ax.hist(data)") == {('ax', 'hist')}

    def test_bare_annotation_has_no_value(self):
        assert self._calls("h: object") == frozenset()

    def test_nested_in_larger_rhs_expression(self):
        # whole RHS is walked, so a draw nested in an expression is caught
        assert self._calls("n = int((ax.hist(data)[0] > 0).sum())") == {('ax', 'hist')}

    def test_pure_capture_included_as_candidate(self):
        # the helper is receiver-agnostic; the identity-coupled gate (applied by
        # the runtime/sim, not here) is what keeps df.mean() cacheable
        assert self._calls("m = df.mean()") == {('df', 'mean')}

    def test_bare_expr_excluded(self):
        assert self._calls("ax.hist(data)") == frozenset()

    def test_augassign_excluded(self):
        assert self._calls("total += box.value()") == frozenset()

    def test_loop_and_function_bodies_excluded(self):
        assert self._calls("for i in xs:\n    r = a.pop(i)") == frozenset()
        assert self._calls("def f():\n    r = a.pop()") == frozenset()

    def test_none_tree(self):
        assert assigned_method_call_receivers(None) == frozenset()


class TestKnownPureMethods:
    """The conservative known-pure inspection set lets the runtime skip
    content-observation for read-only methods on big objects."""

    def test_common_pandas_inspection_pure(self):
        for m in ('head', 'tail', 'describe', 'info', 'sample', 'value_counts'):
            assert m in KNOWN_PURE_METHODS

    def test_mutating_methods_not_in_pure_set(self):
        for m in ('append', 'add', 'update', 'pop', 'sort'):
            assert m not in KNOWN_PURE_METHODS


# ---------------------------------------------------------------------------
# Top-level mutation detection
# ---------------------------------------------------------------------------

class TestTopLevelMutatedVars:
    """Behaviour previously in MutationDetector.get_top_level_mutated_variables."""

    def test_top_level_append_detected(self):
        a = _analyze("lst.append(1)")
        assert 'lst' in a.top_level_mutated_vars

    def test_mutation_inside_function_not_top_level(self):
        code = "def foo():\n    lst.append(1)"
        a = _analyze(code)
        # all_mutated_vars walks the full tree
        assert 'lst' in a.all_mutated_vars
        # top_level_mutated_vars skips function/class bodies
        assert 'lst' not in a.top_level_mutated_vars

    def test_mutation_inside_class_not_top_level(self):
        code = "class Foo:\n    def method(self):\n        self.val = 1"
        a = _analyze(code)
        assert 'self' not in a.top_level_mutated_vars

    def test_mixed_top_level_and_nested(self):
        code = "d['x'] = 1\ndef foo():\n    lst.append(2)"
        a = _analyze(code)
        assert 'd' in a.top_level_mutated_vars
        assert 'lst' not in a.top_level_mutated_vars
        assert 'lst' in a.all_mutated_vars


# ---------------------------------------------------------------------------
# Side-effect detection
# ---------------------------------------------------------------------------

class TestSideEffects:
    """Behaviour previously in SideEffectDetector."""

    def test_open_write_mode(self):
        a = _analyze("with open('out.txt', 'w') as f: f.write('x')")
        kinds = [e.kind for e in a.side_effects]
        assert 'file_write' in kinds

    def test_open_append_mode(self):
        a = _analyze("open('log.txt', 'a')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_open_read_mode_not_detected(self):
        a = _analyze("f = open('data.csv', 'r')")
        assert not any(e.kind == 'file_write' for e in a.side_effects)

    def test_pathlib_write_text(self):
        # CAS-83: Path.write_text is a file write; without this the write-only
        # cell is cacheable and a cache hit skips creating the file.
        a = _analyze("from pathlib import Path\nPath('out.json').write_text('x')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_pathlib_write_bytes(self):
        a = _analyze("p.write_bytes(b'\\x00')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_pathlib_read_text_not_detected(self):
        a = _analyze("s = p.read_text()")
        assert not any(e.kind == 'file_write' for e in a.side_effects)

    def test_statement_writes_files_helper(self):
        # CAS-81/82: the sim/planner seam for scheduling stale writers.
        from cash.notebook.cacheability import statement_writes_files
        assert statement_writes_files("df.to_csv('out.csv', index=False)")
        assert statement_writes_files("with open('f.txt', 'w') as f:\n    f.write('x')")
        assert statement_writes_files("pickle.dump(obj, fh)")
        assert not statement_writes_files("df = pd.read_csv('in.csv')")
        assert not statement_writes_files("x = 1 + 2")
        assert not statement_writes_files("with open('f.txt') as f:\n    body = f.read()")

    def test_os_remove(self):
        a = _analyze("import os; os.remove('file.txt')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_os_makedirs(self):
        a = _analyze("import os; os.makedirs('new_dir')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_shutil_copy(self):
        a = _analyze("import shutil; shutil.copy('a', 'b')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_pandas_to_csv(self):
        a = _analyze("df.to_csv('output.csv')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_pandas_to_parquet(self):
        a = _analyze("df.to_parquet('data.parquet')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_json_dump(self):
        a = _analyze("import json; json.dump(data, f)")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_pickle_dump(self):
        a = _analyze("import pickle; pickle.dump(obj, f)")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_savefig(self):
        a = _analyze("plt.savefig('plot.png')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_numpy_save(self):
        a = _analyze("np.save('arr.npy', data)")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_write_method(self):
        a = _analyze("f.write('content')")
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_subprocess_run(self):
        a = _analyze("import subprocess; subprocess.run(['ls'])")
        assert any(e.kind == 'system' for e in a.side_effects)

    def test_subprocess_popen(self):
        a = _analyze("subprocess.Popen(['cmd'])")
        assert any(e.kind == 'system' for e in a.side_effects)

    def test_os_system(self):
        a = _analyze("os.system('rm -f file')")
        assert any(e.kind == 'system' for e in a.side_effects)

    def test_requests_post(self):
        a = _analyze("requests.post('https://api.example.com', data=payload)")
        assert any(e.kind == 'network' for e in a.side_effects)

    def test_requests_delete(self):
        a = _analyze("requests.delete(url)")
        assert any(e.kind == 'network' for e in a.side_effects)

    def test_no_side_effects_for_pure_code(self):
        a = _analyze("x = 1 + 2")
        assert len(a.side_effects) == 0

    def test_syntax_error_returns_empty(self):
        a = _analyze("def (:")
        assert len(a.side_effects) == 0

    def test_pre_parsed_tree(self):
        code = "df.to_csv('output.csv')"
        tree = ast.parse(code)
        a = analyze_statement(code, tree)
        assert any(e.kind == 'file_write' for e in a.side_effects)

    def test_line_numbers_captured(self):
        code = "df.to_csv('out.csv')\nx = 1"
        a = _analyze(code)
        assert a.side_effects[0].line == 1

    def test_multiple_effects_detected(self):
        code = "os.remove('a')\ndf.to_csv('b')"
        a = _analyze(code)
        assert len(a.side_effects) >= 2


# ---------------------------------------------------------------------------
# statement_written_paths — output-path extraction for write-provenance (CAS-153)
# ---------------------------------------------------------------------------

class TestStatementWrittenPaths:
    """The write-freshness short-circuit's output-path extractor.

    Correctness contract: resolve a literal / namespace-bound path, and return
    ``None`` (conservative) for anything computed or unrecognised so a writer is
    never silently skipped when its target cannot be pinned down.
    """

    def _paths(self, code, namespace=None):
        from cash.notebook.cacheability import statement_written_paths
        return statement_written_paths(code, None, namespace)

    def test_to_csv_string_literal(self):
        assert self._paths("df.to_csv('out.csv', index=False)") == {'out.csv'}

    def test_to_parquet_pickle_json_feather(self):
        assert self._paths("df.to_parquet('a.parquet')") == {'a.parquet'}
        assert self._paths("df.to_pickle('a.pkl')") == {'a.pkl'}
        assert self._paths("df.to_json('a.json')") == {'a.json'}
        assert self._paths("df.to_feather('a.feather')") == {'a.feather'}

    def test_open_write_and_append_modes(self):
        assert self._paths("with open('log.txt', 'w') as f:\n    f.write('x')") == {'log.txt'}
        assert self._paths("with open('log.txt', 'a') as f:\n    f.write('x')") == {'log.txt'}

    def test_open_read_mode_returns_none(self):
        # A read is not a write target — nothing to vouch for.
        assert self._paths("with open('log.txt') as f:\n    body = f.read()") is None

    def test_savefig(self):
        assert self._paths("plt.savefig('plot.png')") == {'plot.png'}

    def test_numpy_save_first_arg(self):
        assert self._paths("np.save('arr.npy', data)") == {'arr.npy'}

    def test_bare_save_method_is_conservative(self):
        # An ambiguous ``.save`` (PIL first-arg vs torch second-arg) is not
        # resolved: it is path-bearing-but-unresolvable -> None.
        assert self._paths("img.save('pic.png')") is None

    def test_pathlib_write_text_inline_constructor(self):
        assert self._paths("Path('out.json').write_text('x')") == {'out.json'}
        assert self._paths("Path('out.bin').write_bytes(b'x')") == {'out.bin'}

    def test_pathlib_write_text_non_inline_receiver_conservative(self):
        # ``p.write_text`` with the path not on an inline Path(...) -> None.
        assert self._paths("p.write_text('x')") is None

    def test_json_dump_into_open(self):
        assert self._paths("json.dump(obj, open('d.json', 'w'))") == {'d.json'}

    def test_pickle_dump_into_open(self):
        assert self._paths("pickle.dump(obj, open('d.pkl', 'wb'))") == {'d.pkl'}

    def test_name_bound_to_str_in_namespace(self):
        assert self._paths("df.to_csv(OUT)", {'OUT': '/tmp/x.csv'}) == {'/tmp/x.csv'}

    def test_name_bound_to_pathlike_in_namespace(self):
        import pathlib
        got = self._paths("df.to_csv(OUT)", {'OUT': pathlib.PurePosixPath('/tmp/x.csv')})
        assert got == {'/tmp/x.csv'}

    def test_fstring_path_returns_none(self):
        # Computed path -> conservative None (must NOT silently skip the writer).
        assert self._paths("df.to_csv(f'{base}/out.csv')") is None

    def test_unknown_name_returns_none(self):
        assert self._paths("df.to_csv(OUT)", {}) is None

    def test_os_path_join_returns_none(self):
        assert self._paths("df.to_csv(os.path.join(d, 'out.csv'))") is None

    def test_to_sql_is_not_a_file_path(self):
        # First arg is a table name, not a filesystem path.
        assert self._paths("df.to_sql('mytable', conn)") is None

    def test_os_remove_not_path_bearing(self):
        # A delete is a write side effect but not a recognised output-producer.
        assert self._paths("os.remove('gone.txt')") is None

    def test_non_writer_returns_none(self):
        assert self._paths("x = 1 + 2") is None
        assert self._paths("df = pd.read_csv('in.csv')") is None


# ---------------------------------------------------------------------------
# called_names
# ---------------------------------------------------------------------------

class TestCalledNames:
    """Bare-name function-call targets for stateful-call detection."""

    def test_simple_call(self):
        a = _analyze("foo()")
        assert 'foo' in a.called_names

    def test_multiple_calls(self):
        a = _analyze("foo()\nbar(x)")
        assert {'foo', 'bar'} <= a.called_names

    def test_method_call_not_included(self):
        """obj.method() — func is an Attribute, not a Name."""
        a = _analyze("obj.method()")
        assert 'method' not in a.called_names
        assert 'obj' not in a.called_names

    def test_nested_call(self):
        a = _analyze("result = foo(bar(x))")
        assert 'foo' in a.called_names
        assert 'bar' in a.called_names


# ---------------------------------------------------------------------------
# skip_reasons
# ---------------------------------------------------------------------------

class TestSkipReasons:
    """skip_reasons(outputs) produces the right human-readable strings."""

    def test_pure_code_no_reasons(self):
        a = _analyze("x = 1 + 2")
        assert a.skip_reasons(set()) == []

    def test_mutation_not_in_outputs_generates_reason(self):
        a = _analyze("lst.append(1)")
        reasons = a.skip_reasons(set())
        assert any('lst' in r for r in reasons)

    def test_mutation_on_output_does_not_generate_reason(self):
        """If 'lst' is an output (e.g. lst = []; lst.append(1)), no skip reason."""
        a = _analyze("lst.append(1)")
        reasons = a.skip_reasons({'lst'})
        assert reasons == []

    def test_side_effect_generates_reason(self):
        a = _analyze("df.to_csv('out.csv')")
        reasons = a.skip_reasons(set())
        assert any('Side effect' in r for r in reasons)
        assert any('file_write' in r for r in reasons)

    def test_both_mutation_and_side_effect(self):
        code = "lst.append(1)\ndf.to_csv('out.csv')"
        a = _analyze(code)
        reasons = a.skip_reasons(set())
        assert any('lst' in r for r in reasons)
        assert any('Side effect' in r for r in reasons)


class TestAccumulatorHint:
    """CAS-145 part b: an accumulator mutation (``out.append(f(e))``) that blocks
    caching also emits a guidance hint pointing at the comprehension form. The
    hint is advisory — it never changes the caching decision, and it is scoped to
    the accumulator methods (append/extend/add/update), NOT every in-place
    mutation (a subscript store / pop / inplace-kwarg call must not get it)."""

    @staticmethod
    def _tips(reasons):
        return [r for r in reasons if r.startswith('tip:')]

    def test_append_accumulator_gets_tip(self):
        # The canonical case: `out = []` then `for e in it: out.append(slow(e))`.
        # Here `out` is mutated (append) but is not an output -> refused + hint.
        a = _analyze("for e in it:\n    out.append(slow(e))")
        reasons = a.skip_reasons(set())
        tips = self._tips(reasons)
        assert len(tips) == 1
        assert 'out = [f(e) for e in it]' in tips[0]

    def test_bare_append_gets_tip(self):
        reasons = _analyze("out.append(slow(e))").skip_reasons(set())
        assert self._tips(reasons)

    def test_extend_add_update_get_tip(self):
        for code in ("acc.extend(xs)", "seen.add(x)", "d.update(other)"):
            reasons = _analyze(code).skip_reasons(set())
            assert self._tips(reasons), code

    def test_subscript_store_does_not_get_tip(self):
        # df['x'] = 1 is an in-place mutation but has no comprehension rewrite.
        a = _analyze("df['x'] = 1")
        reasons = a.skip_reasons(set())
        assert any('In-place mutation on: df' in r for r in reasons)
        assert self._tips(reasons) == []

    def test_attribute_store_does_not_get_tip(self):
        reasons = _analyze("obj.attr = 1").skip_reasons(set())
        assert self._tips(reasons) == []

    def test_non_accumulator_mutating_method_does_not_get_tip(self):
        # pop/sort mutate in place but are NOT accumulators -> reason but no tip.
        for code in ("lst.pop()", "lst.sort()", "del d['k']"):
            reasons = _analyze(code).skip_reasons(set())
            assert reasons, code
            assert self._tips(reasons) == [], code

    def test_inplace_kwarg_mutation_does_not_get_tip(self):
        reasons = _analyze("df.dropna(inplace=True)").skip_reasons(set())
        assert any('In-place mutation on: df' in r for r in reasons)
        assert self._tips(reasons) == []

    def test_tip_is_advisory_decision_unchanged(self):
        # The hint does not change the decision: the statement is still refused
        # (non-empty reasons) and the mutated-var set is unchanged.
        a = _analyze("for e in it:\n    out.append(slow(e))")
        with_tip = a.skip_reasons(set())
        assert any('In-place mutation on: out' in r for r in with_tip)
        assert a.top_level_mutated_vars == {'out'}
        assert 'out' in a.accumulator_mutated_vars
        # When `out` IS an output, nothing fires at all (tip included).
        assert a.skip_reasons({'out'}) == []


# ---------------------------------------------------------------------------
# Immutability of StatementAnalysis
# ---------------------------------------------------------------------------

class TestImmutability:
    """StatementAnalysis is frozen — fields are immutable."""

    def test_frozen(self):
        a = _analyze("x = 1")
        with pytest.raises(Exception):
            a.all_mutated_vars = frozenset({'x'})  # type: ignore[misc]

    def test_fields_are_frozenset_and_tuple(self):
        a = _analyze("lst.append(1)\ndf.to_csv('out.csv')")
        assert isinstance(a.top_level_mutated_vars, frozenset)
        assert isinstance(a.all_mutated_vars, frozenset)
        assert isinstance(a.side_effects, tuple)
        assert isinstance(a.called_names, frozenset)


class TestSelfrefInplaceWriteVars:
    """``selfref_inplace_write_vars`` detects NON-IDEMPOTENT self-referential
    in-place subscript/attribute writes (the written target is also read) so the
    receiver is reset on isolated re-run (CAS-54). New-target writes read from
    OTHER keys are excluded so they keep their per-statement cache (CAS-42)."""

    @staticmethod
    def _vars(code):
        return selfref_inplace_write_vars(ast.parse(code))

    def test_column_scale_self_ref(self):
        assert self._vars("df['a'] = df['a'] * 2") == {'df'}

    def test_column_augmented(self):
        assert self._vars("df['a'] += 100") == {'df'}

    def test_iloc_scalar_self_ref(self):
        assert self._vars("df.iloc[2, 0] = df.iloc[2, 0] + 1000") == {'df'}

    def test_iloc_augmented(self):
        assert self._vars("df.iloc[2, 0] += 5") == {'df'}

    def test_column_method_self_ref(self):
        assert self._vars("df['a'] = df['a'].fillna(0)") == {'df'}

    def test_attribute_self_ref(self):
        assert self._vars("obj.total = obj.total + 1") == {'obj'}

    def test_new_column_from_other_excluded(self):
        # CAS-42: writes a NEW column read from a DIFFERENT column -> idempotent.
        assert self._vars("df['b'] = df['a'] + 1") == frozenset()

    def test_groupby_transform_excluded(self):
        assert self._vars(
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.mean())"
        ) == frozenset()

    def test_constant_assign_excluded(self):
        assert self._vars("df['c'] = 5") == frozenset()

    def test_name_reassign_excluded(self):
        # plain Name reassignment is handled by the reassigned-names path, not here
        assert self._vars("x = x + 1") == frozenset()

    def test_cross_object_excluded(self):
        assert self._vars("df['a'] = other['a'] * 2") == frozenset()

    def test_loc_masked_self_ref_same_column(self):
        # CAS-55: masked .loc write reads the SAME column spelled differently
        # (df['a']) than the target (df.loc[mask, 'a']) -> still self-referential.
        assert self._vars("df.loc[df['a'] >= 50, 'a'] = df['a'] * 2") == {'df'}

    def test_loc_masked_self_ref_via_loc_read(self):
        assert self._vars("df.loc[mask, 'a'] = df.loc[mask, 'a'] + 1") == {'df'}

    def test_loc_masked_list_columns_self_ref(self):
        assert self._vars("df.loc[mask, ['a', 'b']] = df[['a', 'b']] * 2") == {'df'}

    def test_loc_masked_new_column_excluded(self):
        # CAS-42: masked write to a DIFFERENT column read from another -> idempotent.
        assert self._vars("df.loc[df['a'] >= 50, 'b'] = df['a'] * 2") == frozenset()

    def test_subset_assign_other_column_excluded(self):
        assert self._vars("df.loc[mask, 'b'] = df['a'] + df['c']") == frozenset()

    def test_loc_row_grow_len(self):
        # CAS-74: df.loc[len(df)] = .. appends a row (size-dependent index).
        assert self._vars("df.loc[len(df)] = 99") == {'df'}

    def test_loc_row_grow_shape(self):
        assert self._vars("df.loc[df.shape[0]] = 99") == {'df'}

    def test_loc_masked_reading_frame_excluded(self):
        # a masked write whose key reads the frame (not len/shape) is idempotent.
        assert self._vars("df.loc[df['a'] > 0, 'b'] = 5") == frozenset()

    def test_iloc_positional_other_excluded(self):
        # positional target is unknown-column; RHS reads a named column -> no
        # provable overlap, and not an exact-target match -> excluded.
        assert self._vars("df.iloc[0, 1] = df['a'].sum()") == frozenset()

    def test_tuple_unpack_column_swap(self):
        # CAS-56: df['a'], df['b'] = df['b'], df['a'] -- statement reads & writes
        # overlapping columns of df -> non-idempotent -> flag df.
        assert self._vars("df['a'], df['b'] = df['b'], df['a']") == {'df'}

    def test_tuple_unpack_self_scale(self):
        assert self._vars("df['a'], df['b'] = df['a'] * 2, df['b'] * 2") == {'df'}

    def test_tuple_unpack_new_columns_excluded(self):
        # CAS-42: new columns c,d derived from existing a,b -> idempotent.
        assert self._vars("df['c'], df['d'] = df['a'], df['b']") == frozenset()

    def test_tuple_unpack_cross_object_excluded(self):
        assert self._vars("df['a'], df['b'] = other['x'], other['y']") == frozenset()

    def test_del_subscript(self):
        # CAS-56: del df['b'] removes a column in place -> non-idempotent
        # (second del KeyErrors) -> df must reset on isolated re-run.
        assert self._vars("del df['b']") == {'df'}

    def test_del_attribute(self):
        assert self._vars("del obj.cache") == {'obj'}

    def test_del_plain_name_excluded(self):
        # deleting a plain name is a namespace op, handled elsewhere -> not here.
        assert self._vars("del x") == frozenset()

    def test_conditional_self_mutation(self):
        # CAS-57: a self-mutation nested in an if-body executes at module level.
        assert self._vars("if cond:\n    df['a'] = df['a'] * 2") == {'df'}

    def test_for_loop_self_mutation(self):
        assert self._vars("for c in cols:\n    df[c] = df[c] * 2") == {'df'}

    def test_for_loop_augmented_self_mutation(self):
        assert self._vars("for c in cols:\n    df[c] += 1") == {'df'}

    def test_with_block_self_mutation(self):
        assert self._vars("with ctx:\n    df['a'] = df['a'] * 2") == {'df'}

    def test_try_body_del(self):
        assert self._vars("try:\n    del df['b']\nexcept KeyError:\n    pass") == {'df'}

    def test_nested_control_self_mutation(self):
        assert self._vars("for c in cols:\n    if c:\n        df[c] = df[c] * 2") == {'df'}

    def test_function_body_not_scanned(self):
        # a mutation inside a def runs only when called -> not a module-level write.
        assert self._vars("def f():\n    df['a'] = df['a'] * 2") == frozenset()

    def test_loop_building_from_source_excluded(self):
        # CAS-42: loop builds df columns from a DIFFERENT object -> idempotent.
        assert self._vars("for c in cols:\n    df[c] = source[c] * 2") == frozenset()

    def test_conditional_new_column_excluded(self):
        # CAS-42: new column derived in a conditional -> idempotent.
        assert self._vars("if cond:\n    df['b'] = df['a'] + 1") == frozenset()

    def test_none_tree(self):
        assert selfref_inplace_write_vars(None) == frozenset()


class TestParamsMutatedInFunction:
    """``params_mutated_in_function`` reports which PARAMETERS a function body
    mutates in place (CAS-58, interprocedural arg-mutation detection)."""

    @staticmethod
    def _params(code):
        fdef = ast.parse(code).body[0]
        return params_mutated_in_function(fdef)

    def test_list_append_param(self):
        assert self._params("def f(x):\n    x.append(1)") == {'x'}

    def test_subscript_assign_param(self):
        assert self._params("def f(d):\n    d['k'] = 1") == {'d'}

    def test_augmented_subscript_param(self):
        assert self._params("def f(d):\n    d['n'] += 1") == {'d'}

    def test_attribute_assign_param(self):
        assert self._params("def f(o):\n    o.count = 5") == {'o'}

    def test_del_subscript_param(self):
        assert self._params("def f(d):\n    del d['k']") == {'d'}

    def test_only_mutated_param_reported(self):
        assert self._params("def f(a, b):\n    a.append(1)\n    return b") == {'a'}

    def test_reassignment_is_not_mutation(self):
        # rebinding the param to a new local object is not a caller mutation.
        assert self._params("def f(x):\n    x = x + [1]\n    return x") == frozenset()

    def test_pure_function_no_params(self):
        assert self._params("def f(x):\n    return x * 2") == frozenset()

    def test_mutation_of_local_not_param(self):
        assert self._params("def f(x):\n    tmp = []\n    tmp.append(x)") == frozenset()

    def test_conditional_mutation_param(self):
        assert self._params("def f(d, c):\n    if c:\n        d['k'] = 1") == {'d'}

    def test_keyword_only_param(self):
        assert self._params("def f(*, d):\n    d.append(1)") == {'d'}


class TestStandaloneCallArgTargets:
    """``standalone_call_arg_targets`` extracts top-level bare-Expr Name-calls and
    their positional/keyword variable arguments."""

    @staticmethod
    def _targets(code):
        return standalone_call_arg_targets(ast.parse(code))

    def test_single_positional(self):
        assert self._targets("f(data)") == frozenset({('f', ('data',), ())})

    def test_multiple_positional(self):
        assert self._targets("f(a, b)") == frozenset({('f', ('a', 'b'), ())})

    def test_keyword_arg(self):
        assert self._targets("f(x=data)") == frozenset({('f', (), (('x', 'data'),))})

    def test_non_name_arg_is_none(self):
        assert self._targets("f(a, [1, 2])") == frozenset({('f', ('a', None), ())})

    def test_starred_arg_is_none(self):
        assert self._targets("f(*args)") == frozenset({('f', (None,), ())})

    def test_method_call_excluded(self):
        # obj.method(x) is handled by the method-receiver path, not here.
        assert self._targets("obj.method(data)") == frozenset()

    def test_captured_result_excluded(self):
        # r = f(data) is not a bare Expr -> excluded (pure calls capture results).
        assert self._targets("r = f(data)") == frozenset()


class TestFunctionArgMutations:
    """``function_arg_mutations`` maps a body's mutated params back to the call's
    argument variables, via a source resolver."""

    SRCS = {
        'append_one': "def append_one(x):\n    x.append(99)",
        'bump': "def bump(d):\n    d['n'] += 1",
        'pure': "def pure(x):\n    return x * 2",
        'two': "def two(a, b):\n    a.append(1)\n    return b",
        'kw': "def kw(*, target):\n    target.append(1)",
    }

    def _muts(self, code):
        return function_arg_mutations(ast.parse(code), self.SRCS.get)

    def test_positional_mutation(self):
        assert self._muts("append_one(data)") == {'data'}

    def test_dict_mutation(self):
        assert self._muts("bump(cfg)") == {'cfg'}

    def test_pure_call_excluded(self):
        assert self._muts("pure(x)") == frozenset()

    def test_only_mutated_position(self):
        # two(a, b) mutates only the first param -> only the first arg.
        assert self._muts("two(rows, keep)") == {'rows'}

    def test_keyword_mapping(self):
        assert self._muts("kw(target=mylist)") == {'mylist'}

    def test_unknown_function_excluded(self):
        assert self._muts("mystery(data)") == frozenset()

    def test_captured_result_excluded(self):
        assert self._muts("r = append_one(data)") == frozenset()


class TestInterproceduralArgMutations:
    """``function_arg_mutations`` follows nested calls (CAS-61): a param mutated
    only through a further resolvable call is detected too."""

    SRCS = {
        'inner': "def inner(z):\n    z.append(9)",
        'outer': "def outer(y):\n    inner(y)",
        'mid': "def mid(b):\n    inner(b)",
        'deep': "def deep(a):\n    mid(a)",
        'inner_pure': "def inner_pure(z):\n    return z * 2",
        'pure_outer': "def pure_outer(y):\n    return inner_pure(y)",
        'recurse': "def recurse(x):\n    recurse(x)",
    }

    def _muts(self, code):
        return function_arg_mutations(ast.parse(code), self.SRCS.get)

    def test_depth2(self):
        assert self._muts("outer(data)") == {'data'}

    def test_depth3(self):
        assert self._muts("deep(data)") == {'data'}

    def test_pure_chain_excluded(self):
        assert self._muts("pure_outer(data)") == frozenset()

    def test_self_recursion_terminates(self):
        # recurse(x) calls itself; the seen-guard prevents infinite recursion and
        # there is no actual mutation, so nothing is flagged.
        assert self._muts("recurse(data)") == frozenset()


class TestAliasMutationSources:
    """``alias_mutation_sources`` maps an in-place mutation through a bare
    ``y = x`` alias back to the upstream source ``x`` (CAS-60)."""

    def _src(self, code):
        return alias_mutation_sources(ast.parse(code))

    def test_method_mutation(self):
        assert self._src("y = x\ny.append(99)") == {'x'}

    def test_subscript_aug_assign(self):
        assert self._src("y = x\ny[0] += 5") == {'x'}

    def test_alias_chain_resolves_to_root(self):
        assert self._src("y = x\nz = y\nz.append(1)") == {'x'}

    def test_two_aliases_same_source(self):
        assert self._src("a = src\nb = src\nb.add(1)") == {'src'}

    def test_copy_is_not_alias(self):
        assert self._src("y = x.copy()\ny.append(1)") == frozenset()

    def test_slice_is_not_alias(self):
        assert self._src("y = x[:]\ny.append(1)") == frozenset()

    def test_no_mutation_excluded(self):
        assert self._src("y = x\nprint(y)") == frozenset()

    def test_direct_mutation_no_alias(self):
        assert self._src("x.append(1)") == frozenset()

    def test_self_assign_ignored(self):
        assert self._src("x = x\nx.append(1)") == frozenset()

    def test_chained_assignment(self):
        assert self._src("a = b = x\na.append(1)") == {'x'}

    def test_tuple_unpack_pair(self):
        assert self._src("(y,) = (x,)\ny.append(1)") == {'x'}

    def test_unpack_only_name_elements(self):
        # literal list element is not a Name -> not an alias
        assert self._src("a, b = [1], src\nb.append(1)") == {'src'}

    def test_non_literal_unpack_excluded(self):
        # RHS is a call, not a literal tuple -> no element aliasing
        assert self._src("a, b = compute()\na.append(1)") == frozenset()


class TestFunctionGlobalMutations:
    """``function_global_mutations`` attributes a called function's free/global
    mutations back to the global (CAS-68 A)."""

    SRCS = {
        'bump': "def bump():\n    global g\n    g += 1",
        'add': "def add():\n    items.append(1)",
        'put': "def put():\n    store['k'] = store.get('k', 0) + 1",
        'pure': "def pure():\n    return 42",
        'local': "def local():\n    acc = []\n    acc.append(1)\n    return acc",
        'arg': "def arg(x):\n    x.append(1)",
    }

    def _f(self, code):
        return function_global_mutations(ast.parse(code), self.SRCS.get)

    def test_global_augassign(self):
        assert self._f("bump()") == {'g'}

    def test_free_var_append(self):
        assert self._f("add()") == {'items'}

    def test_free_var_subscript(self):
        assert self._f("put()") == {'store'}

    def test_pure_excluded(self):
        assert self._f("pure()") == frozenset()

    def test_local_excluded(self):
        assert self._f("local()") == frozenset()

    def test_param_mutation_excluded(self):
        # a param mutation is CAS-58's job, not a global
        assert self._f("arg(d)") == frozenset()


class TestStatefulSelfFunctions:
    """``stateful_self_functions`` flags a called function that mutates state on
    its own object — mutable default arg or function attribute (CAS-68 B)."""

    SRCS = {
        'collect': "def collect(x, acc=[]):\n    acc.append(x)\n    return acc",
        'tally': "def tally(k, acc={}):\n    acc[k] = acc.get(k, 0) + 1\n    return acc",
        'tick': "def tick():\n    tick.count = getattr(tick, 'count', 0) + 1\n    return tick.count",
        'pure': "def pure(x):\n    return x + 1",
        'none_default': "def g(x, cache=None):\n    return x",
        'fresh_default': "def h(x, acc=[]):\n    return acc + [x]",
        'memo': "@functools.lru_cache(maxsize=None)\ndef memo(x):\n    seen.append(x)\n    return x",
        'pure_memo': "@lru_cache\ndef pure_memo(x):\n    return x * x",
    }

    def _f(self, code):
        return stateful_self_functions(ast.parse(code), self.SRCS.get)

    def test_lru_cache_memoizer(self):
        # a functools memoizer carries a persistent cache -> stateful (CAS-80)
        assert self._f("memo(1)") == {'memo'}

    def test_bare_lru_cache_memoizer(self):
        assert self._f("r = pure_memo(2)") == {'pure_memo'}

    def test_mutable_default_list_captured(self):
        assert self._f("r = collect(1)") == {'collect'}

    def test_mutable_default_dict_captured(self):
        assert self._f("r = tally('a')") == {'tally'}

    def test_function_attribute(self):
        assert self._f("print(tick())") == {'tick'}

    def test_pure_excluded(self):
        assert self._f("r = pure(1)") == frozenset()

    def test_none_default_excluded(self):
        assert self._f("r = g(1)") == frozenset()

    def test_default_read_not_mutated_excluded(self):
        # acc is a mutable default but only READ (acc + [x]), not mutated in place
        assert self._f("r = h(1)") == frozenset()


class TestStatefulClosureVars:
    """``stateful_closure_vars`` flags a called closure variable whose factory
    returns an inner function mutating factory-local state (CAS-68 B closure)."""

    FACTORIES = {
        'c': "def make_counter():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n        return n\n    return inc",
        'adder': "def make_list():\n    data = []\n    def add(x):\n        data.append(x)\n    return add",
        'p': "def make_pure():\n    def f(x):\n        return x + 1\n    return f",
    }

    def _resolve(self, name):
        src = self.FACTORIES.get(name)
        return ast.parse(src).body[0] if src else None

    def _f(self, code):
        return stateful_closure_vars(ast.parse(code), self._resolve)

    def test_nonlocal_counter(self):
        assert self._f("print(c())") == {'c'}

    def test_captured_list(self):
        assert self._f("adder(1)") == {'adder'}

    def test_pure_closure_excluded(self):
        assert self._f("r = p(1)") == frozenset()

    def test_unknown_var_excluded(self):
        assert self._f("unknown()") == frozenset()


class TestFunctoolsHiddenMutations:
    """partial / reduce hidden-mutation detectors (CAS-72)."""

    SRCS = {
        'push': "def push(lst, v):\n    lst.append(v)",
        'tick': "def tick(step):\n    counter[0] += step",
        'combine': "def combine(a, b):\n    log.append(b)\n    return a + b",
        'mul': "def mul(a, b):\n    return a * b",
    }
    PARTIALS = {
        'p': ('push', ['shared']),
        't': ('tick', [None]),          # partial(tick, 1) -> literal, no bound Name
        'double': ('mul', [None]),      # partial(mul, 2) -> pure
    }

    def _pam(self, code):
        return partial_arg_mutations(ast.parse(code), self.PARTIALS.get, self.SRCS.get)

    def _mp(self, code):
        return mutating_partials(ast.parse(code), self.PARTIALS.get, self.SRCS.get)

    def _rfm(self, code):
        return reduce_free_mutations(ast.parse(code), self.SRCS.get)

    def test_partial_bound_arg_mutated(self):
        assert self._pam("p('a')") == {'shared'}

    def test_partial_free_var_mutated(self):
        assert self._pam("t()") == {'counter'}

    def test_partial_pure_not_flagged(self):
        assert self._pam("r = double(5)") == frozenset()

    def test_mutating_partial_bound_arg(self):
        # p binds a mutated arg -> must re-bind
        assert self._mp("p('a')") == {'p'}

    def test_mutating_partial_free_var_excluded(self):
        # t mutates only a free var (global) -> no re-bind needed
        assert self._mp("t()") == frozenset()

    def test_reduce_side_effect(self):
        assert self._rfm("total = reduce(combine, [1, 2, 3], 0)") == {'log'}

    def test_reduce_pure_not_flagged(self):
        assert self._rfm("total = reduce(mul, [1, 2, 3], 1)") == frozenset()


class TestObjectProtocolMutations:
    """``object_protocol_mutations`` attributes a hidden mutation reached through
    the object protocol — a ``with`` statement, a custom dunder, a decorated
    call, a constructor, or an instance / class method — to the correct reset
    channel (free var / receiver / class def), CAS-69/70/71/73."""

    CLASSES = {
        'Counter': ("class Counter:\n    def __init__(self):\n        self.n = 0\n"
                    "    def __enter__(self):\n        self.n += 1\n        return self\n"
                    "    def __exit__(self, *a):\n        return False"),
        'Store': ("class Store:\n    def __setitem__(self, k, v):\n        log.append((k, v))\n"
                  "    def __delitem__(self, k):\n        log.append(k)"),
        'Accum': ("class Accum:\n    def __init__(self):\n        self.calls = []\n"
                  "    def __call__(self, x):\n        self.calls.append(x)\n        return x"),
        'Registry': ("class Registry:\n    log = []\n    @classmethod\n"
                     "    def record(cls):\n        cls.log.append('r')"),
        'Reg': ("class Reg:\n    registry = []\n    def __init__(self):\n"
                "        Reg.registry.append(id(self))"),
        'Shared': ("class Shared:\n    data = []\n    def add(self, x):\n"
                   "        self.data.append(x)"),
        'Stack': ("class Stack:\n    def __init__(self):\n        self.data = []\n"
                  "    def push(self, x):\n        self.data.append(x)\n        return self.data"),
        'Point': ("class Point:\n    def __init__(self, x, y):\n        self.x = x\n"
                  "        self.y = y\n    def norm(self):\n        return self.x + self.y"),
        'Doubler': "class Doubler:\n    def __call__(self, x):\n        return x * 2",
        'Box': ("class Box:\n    def __iadd__(self, x):\n        log.append(x)\n"
                "        return self"),
        'Scaler': ("class Scaler:\n    seen = []\n    def __imul__(self, x):\n"
                   "        Scaler.seen.append(x)\n        return self"),
        'Money': ("class Money:\n    def __init__(self, c):\n        self.c = c\n"
                  "    def __iadd__(self, x):\n        return Money(self.c + x)"),
    }
    FUNCS = {
        'track': ("@contextlib.contextmanager\ndef track():\n    log.append('e')\n"
                  "    yield\n    log.append('x')"),
        'logged': ("def logged(f):\n    def wrap(*a, **k):\n        calls.append('x')\n"
                   "        return f(*a, **k)\n    return wrap"),
        'work': "@logged\ndef work():\n    return 42",
        'trace': ("def trace(f):\n    def wrap(*a, **k):\n        return f(*a, **k)\n    return wrap"),
        'square': "@trace\ndef square(x):\n    return x * x",
    }
    # var -> constructing name (for instance_class + reassignment-decorator factory)
    FACTORIES = {'cm': 'Counter', 's': 'Store', 'a': 'Accum', 'r': 'Reg',
                 'x': 'Shared', 'y': 'Shared', 'st': 'Stack', 'p': 'Point',
                 'd': 'Doubler', 'bx': 'Box', 'sc': 'Scaler', 'mo': 'Money'}

    def _instance_class(self, var):
        cls = self.FACTORIES.get(var)
        return cls if cls in self.CLASSES else None

    def _var_factory(self, name):
        src = self.FUNCS.get(self.FACTORIES.get(name))
        if not src:
            return None
        node = ast.parse(src).body[0]
        return node if isinstance(node, ast.FunctionDef) else None

    def _f(self, code):
        return object_protocol_mutations(
            ast.parse(code), self.CLASSES.get, self._instance_class,
            self.FUNCS.get, self._var_factory,
        )

    # --- free-var channel ----------------------------------------------------
    def test_contextmanager_generator_free_var(self):
        assert self._f("with track():\n    pass").free_vars == {'log'}

    def test_setitem_free_var(self):
        assert self._f("s['k'] = 1").free_vars == {'log'}

    def test_delitem_free_var(self):
        assert self._f("del s['a']").free_vars == {'log'}

    def test_decorator_wrapper_free_var(self):
        assert self._f("work()").free_vars == {'calls'}

    # --- receiver channel ----------------------------------------------------
    def test_with_enter_mutates_self(self):
        assert self._f("with cm:\n    pass").receivers == {'cm'}

    def test_call_mutates_self(self):
        assert self._f("r2 = a('z')").receivers == {'a'}

    def test_instance_method_mutates_self(self):
        assert self._f("out = st.push('x')").receivers == {'st'}

    # --- class-def channel ---------------------------------------------------
    def test_classmethod_class_var(self):
        assert self._f("Registry.record()").class_defs == {'Registry'}

    def test_constructor_class_var(self):
        assert self._f("obj = Reg()").class_defs == {'Reg'}

    def test_instance_method_class_var(self):
        # Shared.data is class-level (no __init__), so add() mutates the class var
        assert self._f("x.add('v')").class_defs == {'Shared'}
        assert self._f("x.add('v')").receivers == frozenset()

    # --- pure guards (nothing flagged) ---------------------------------------
    def test_pure_construction_not_flagged(self):
        r = self._f("p = Point(3, 4)")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_pure_call_not_flagged(self):
        r = self._f("out = d(5)")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_pure_decorator_not_flagged(self):
        assert self._f("square(6)").free_vars == frozenset()

    def test_builtin_subscript_not_flagged(self):
        # a plain name that resolves to no notebook class is never flagged
        r = self._f("unknown['k'] = 1")
        assert not (r.free_vars or r.receivers or r.class_defs)

    # --- in-place operator dunders (CAS-78) ----------------------------------
    def test_iadd_free_var(self):
        assert self._f("bx += 1").free_vars == {'log'}

    def test_imul_class_var(self):
        assert self._f("sc *= 2").class_defs == {'Scaler'}

    def test_pure_iadd_not_flagged(self):
        r = self._f("mo += 5")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_int_augassign_not_flagged(self):
        # plain int += on a non-instance name resolves to no class
        r = self._f("i += 1")
        assert not (r.free_vars or r.receivers or r.class_defs)


class TestObjectProtocolInheritance:
    """``object_protocol_mutations`` follows base classes (CAS-76) and dataclass
    ``__post_init__`` (CAS-79): an inherited method / __init__ / __enter__ or a
    base-owned class variable is attributed to the OWNING class."""

    CLASSES = {
        'Base': ("class Base:\n    registry = []\n    def __init__(self):\n"
                 "        Base.registry.append(1)"),
        'Sub': "class Sub(Base):\n    pass",
        'SBase': ("class SBase:\n    shared = []\n    def add(self, v):\n"
                  "        self.shared.append(v)"),
        'SSub': "class SSub(SBase):\n    pass",
        'SupBase': ("class SupBase:\n    seen = []\n    def __init__(self):\n"
                    "        SupBase.seen.append(1)"),
        'SupSub': ("class SupSub(SupBase):\n    def __init__(self):\n"
                   "        super().__init__()"),
        'PBase': ("class PBase:\n    def __init__(self, n):\n        self.n = n\n"
                  "    def doubled(self):\n        return self.n * 2"),
        'PSub': "class PSub(PBase):\n    pass",
        'DReg': ("@dataclass\nclass DReg:\n    name: str\n    def __post_init__(self):\n"
                 "        log.append(self.name)"),
        'DNode': ("@dataclass\nclass DNode:\n    val: int\n    registry: ClassVar[list] = []\n"
                  "    def __post_init__(self):\n        DNode.registry.append(self.val)"),
    }
    FACTORIES = {'sub': 'Sub', 'a': 'SSub', 'b': 'SSub', 'sup': 'SupSub', 'ps': 'PSub'}

    def _instance_class(self, var):
        cls = self.FACTORIES.get(var)
        return cls if cls in self.CLASSES else None

    def _f(self, code):
        return object_protocol_mutations(
            ast.parse(code), self.CLASSES.get, self._instance_class,
            lambda n: None, lambda n: None,
        )

    def test_inherited_init_owner_is_base(self):
        assert self._f("sub = Sub()").class_defs == {'Base', 'Sub'}

    def test_super_init_follows_base(self):
        assert self._f("sup = SupSub()").class_defs == {'SupBase', 'SupSub'}

    def test_inherited_method_shared_attr(self):
        # add() (on SBase) mutates self.shared, a base class var — reset both
        assert self._f("a.add('v')").class_defs == {'SBase', 'SSub'}

    def test_pure_inheritance_not_flagged(self):
        r = self._f("ps = PSub(4)")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_dataclass_post_init_free_var(self):
        assert self._f("r = DReg('a')").free_vars == {'log'}

    def test_dataclass_post_init_class_var(self):
        assert self._f("n = DNode(7)").class_defs == {'DNode'}


class TestObjectProtocolDescriptor:
    """``object_protocol_mutations`` follows the descriptor protocol (CAS-77): an
    attribute assign / load dispatching to a ``@property`` setter/getter or a
    data-descriptor ``__set__`` / ``__get__``."""

    CLASSES = {
        'Cfg': ("class Cfg:\n    @property\n    def x(self):\n        return self._x\n"
                "    @x.setter\n    def x(self, v):\n        log.append(v)\n        self._x = v"),
        'Rec': ("class Rec:\n    def __init__(self):\n        self.history = []\n"
                "    @property\n    def val(self):\n        return self._v\n"
                "    @val.setter\n    def val(self, v):\n        self.history.append(v)\n"
                "        self._v = v"),
        'Meter': ("class Meter:\n    def __init__(self):\n        self.reads = []\n"
                  "    @property\n    def now(self):\n        self.reads.append(1)\n"
                  "        return len(self.reads)"),
        'Circle': ("class Circle:\n    def __init__(self, r):\n        self._r = r\n"
                   "    @property\n    def area(self):\n        return 3 * self._r * self._r"),
        'Tracked': ("class Tracked:\n    def __set__(self, obj, v):\n        log.append(v)\n"
                    "    def __get__(self, obj, owner=None):\n        log.append(1)\n"
                    "        return None"),
        'Model': "class Model:\n    field = Tracked()",
        'Bag': "class Bag:\n    def __init__(self):\n        self.n = 0",
    }
    FACTORIES = {'c': 'Cfg', 'r': 'Rec', 'm': 'Meter', 'ci': 'Circle',
                 'mdl': 'Model', 'bag': 'Bag'}

    def _instance_class(self, var):
        cls = self.FACTORIES.get(var)
        return cls if cls in self.CLASSES else None

    def _f(self, code):
        return object_protocol_mutations(
            ast.parse(code), self.CLASSES.get, self._instance_class,
            lambda n: None, lambda n: None,
        )

    def test_property_setter_free_var(self):
        assert self._f("c.x = 5").free_vars == {'log'}

    def test_property_setter_self_list(self):
        assert self._f("r.val = 7").receivers == {'r'}

    def test_property_getter_side_effect_self(self):
        assert self._f("v = m.now").receivers == {'m'}

    def test_descriptor_set_free_var(self):
        assert self._f("mdl.field = 3").free_vars == {'log'}

    def test_descriptor_get_free_var(self):
        # __set__ AND __get__ both append to log; a bare load hits __get__
        r = self._f("v = mdl.field")
        assert 'log' in r.free_vars

    def test_pure_property_not_flagged(self):
        r = self._f("a = ci.area")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_plain_attribute_assign_not_flagged(self):
        r = self._f("bag.n = 5")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_plain_attribute_load_not_flagged(self):
        r = self._f("y = bag.n")
        assert not (r.free_vars or r.receivers or r.class_defs)


class TestObjectProtocolExotic:
    """``object_protocol_mutations`` handles the indirect channels (CAS-80):
    ``next(it)``, ``ExitStack.enter_context``, a context-manager factory, and a
    class-based decorator."""

    CLASSES = {
        'CM': ("class CM:\n    def __enter__(self):\n        log.append(1)\n"
               "        return self\n    def __exit__(self, *a):\n        return False"),
        'Mgr': ("class Mgr:\n    def __enter__(self):\n        hits.append(1)\n"
                "        return len(hits)\n    def __exit__(self, *a):\n        return False"),
        'It': ("class It:\n    def __iter__(self):\n        return self\n"
               "    def __next__(self):\n        cursor[0] += 1\n        return 1"),
        'Counter': ("class Counter:\n    def __init__(self, f):\n        self.f = f\n"
                    "        self.n = 0\n    def __call__(self, *a, **k):\n"
                    "        self.n += 1\n        return self.f(*a, **k)"),
        'Quiet': ("class Quiet:\n    def __enter__(self):\n        return 42\n"
                  "    def __exit__(self, *a):\n        return False"),
    }
    FUNCS = {'cm': "def cm():\n    return Mgr()",
             'make': "def make():\n    return Quiet()"}
    FACTORIES = {'cmv': 'CM', 'it': 'It'}
    DECORATED = {'task': 'Counter'}

    def _ic(self, var):
        cls = self.FACTORIES.get(var)
        return cls if cls in self.CLASSES else None

    def _dc(self, var):
        cls = self.DECORATED.get(var)
        return cls if cls in self.CLASSES else None

    def _f(self, code):
        return object_protocol_mutations(
            ast.parse(code), self.CLASSES.get, self._ic, self.FUNCS.get,
            lambda n: None, decorated_class=self._dc,
        )

    def test_next_free_var(self):
        assert self._f("v = next(it)").free_vars == {'cursor'}

    def test_enter_context_free_var(self):
        assert self._f("stack.enter_context(cmv)").free_vars == {'log'}

    def test_with_factory_free_var(self):
        assert self._f("with cm() as x:\n    pass").free_vars == {'hits'}

    def test_class_based_decorator_stateful(self):
        assert self._f("task()").class_defs == {'task'}

    def test_pure_factory_cm_not_flagged(self):
        r = self._f("with make() as v:\n    pass")
        assert not (r.free_vars or r.receivers or r.class_defs)

    def test_aliased_decorator_via_factory(self):
        # resolve_var_factory follows h -> g -> counting (alias handled upstream);
        # here it directly returns counting's def, exercising the wrapper analysis
        counting = ("def counting(f):\n    def wrap(*a, **k):\n        log.append(1)\n"
                    "        return f(*a, **k)\n    return wrap")

        def vf(name):
            return ast.parse(counting).body[0] if name == 'h' else None

        r = object_protocol_mutations(
            ast.parse("h()"), lambda n: None, lambda n: None, lambda n: None, vf,
        )
        assert r.free_vars == {'log'}


class TestSubscriptViewBindings:
    """``subscript_view_bindings`` maps ``alias = base[...]`` bindings (CAS-74)."""

    def _f(self, code):
        return subscript_view_bindings(ast.parse(code))

    def test_slice_binding(self):
        assert self._f("v = arr[1:]") == {'v': 'arr'}

    def test_full_slice_binding(self):
        assert self._f("v = arr[:]") == {'v': 'arr'}

    def test_step_slice_binding(self):
        assert self._f("v = arr[::2]") == {'v': 'arr'}

    def test_plain_name_not_a_view(self):
        assert self._f("v = arr") == {}

    def test_copy_call_not_a_view(self):
        assert self._f("v = arr.copy()") == {}

    def test_binding_in_if_body(self):
        assert self._f("if c:\n    v = arr[1:]") == {'v': 'arr'}


class TestCrossrefReassignedVars:
    """``crossref_reassigned_vars`` flags swap/rotate/temp-swap names but not
    single-statement self-accumulation (CAS-65)."""

    def _f(self, code):
        return crossref_reassigned_vars(ast.parse(code))

    def test_tuple_swap(self):
        assert self._f("a, b = b, a") == {'a', 'b'}

    def test_three_way_rotate(self):
        assert self._f("a, b, c = c, a, b") == {'a', 'b', 'c'}

    def test_temp_swap(self):
        assert self._f("tmp = a\na = b\nb = tmp") == {'a', 'b'}

    def test_partial_swap(self):
        assert self._f("a, b = b, c") == {'b'}

    def test_read_before_write_from_other(self):
        # b read then reassigned from a DIFFERENT value -> swap-like.
        assert self._f("y = b\nb = k") == {'b'}

    def test_read_before_write_selfref_excluded(self):
        # total is read then reassigned FROM ITSELF -> accumulator, not a swap.
        assert self._f("y = total\ntotal = total + k") == frozenset()

    def test_self_accumulate_excluded(self):
        assert self._f("x = x + 1") == frozenset()

    def test_accumulator_excluded(self):
        assert self._f("total = total + k") == frozenset()

    def test_df_mutate_then_selfref_transform_excluded(self):
        # df mutated then reassigned from itself -> handled by reassign-reset, not a swap.
        assert self._f("df['rm'] = df['p'].rolling(3).mean()\ndf = df.sort_values('d')") == frozenset()

    def test_fresh_unpack_excluded(self):
        assert self._f("a, b = 1, 2") == frozenset()

    def test_list_unpack_excluded(self):
        assert self._f("x, y = data") == frozenset()

    def test_swap_in_if_body(self):
        assert self._f("if cond:\n    a, b = b, a") == {'a', 'b'}


class TestAliasedSources:
    """``aliased_sources`` resolves a set of names through the alias map back to
    their upstream roots (used to extend selfref / method-receiver sets)."""

    def _f(self, code, names):
        return aliased_sources(ast.parse(code), names)

    def test_selfref_df_alias(self):
        # df2 is the selfref receiver; resolves to df
        assert self._f("df2 = df\ndf2['a'] = df2['a'] * 2", {'df2'}) == {'df'}

    def test_method_df_alias(self):
        assert self._f("df2 = df\ndf2.fillna(0, inplace=True)", {'df2'}) == {'df'}

    def test_non_alias_name_contributes_nothing(self):
        assert self._f("df['a'] = df['a'] * 2", {'df'}) == frozenset()

    def test_empty_names(self):
        assert self._f("y = x", set()) == frozenset()


class TestBareAliasTargets:
    """``bare_alias_targets`` finds bindings that are pure pointer copies of a
    bare ``Name`` — the statements that must never cache, because a restore hands
    back a copy where Python guarantees identity (CAS-184)."""

    def _f(self, code):
        return bare_alias_targets(ast.parse(code))

    # --- the shapes that ARE aliases ---

    def test_simple_alias(self):
        assert self._f("b = a") == {'b'}

    def test_chained_alias(self):
        # ``b = c = a``: BOTH names bind a's object.
        assert self._f("b = c = a") == {'b', 'c'}

    def test_tuple_alias_1to1(self):
        # The RHS tuple is built then unpacked element-wise: two pointer copies.
        assert self._f("b, c = a, d") == {'b', 'c'}

    def test_list_display_alias(self):
        assert self._f("[b, c] = [a, d]") == {'b', 'c'}

    # --- the shapes that are NOT aliases ---

    def test_unpack_from_single_name_is_not_alias(self):
        # ``b, c = a`` INDEXES a: b is a[0], a different object from a.
        assert self._f("b, c = a") == frozenset()

    def test_self_bind_excluded(self):
        assert self._f("x = x") == frozenset()

    def test_copy_is_not_alias(self):
        assert self._f("b = a.copy()") == frozenset()

    def test_slice_is_not_alias(self):
        assert self._f("b = a[:]") == frozenset()

    def test_attribute_rhs_is_not_alias(self):
        # ``b = a.attr`` can alias a live mutable, but it is not a bare Name and
        # may be expensive, so the cost half of the argument does not transfer.
        assert self._f("b = a.attr") == frozenset()

    def test_subscript_rhs_is_not_alias(self):
        assert self._f("b = a[0]") == frozenset()

    def test_call_rhs_is_not_alias(self):
        assert self._f("b = f(a)") == frozenset()

    def test_literal_rhs_is_not_alias(self):
        assert self._f("b = 5") == frozenset()

    def test_tuple_with_computed_element_is_not_alias(self):
        # ``f()`` is real work worth caching, so the statement keeps its cache.
        assert self._f("b, c = a, f()") == frozenset()

    def test_tuple_arity_mismatch_is_not_alias(self):
        # A starred unpack is not a 1:1 pointer copy.
        assert self._f("b, *c = a, d, e") == frozenset()

    def test_subscript_target_is_not_alias(self):
        assert self._f("d['k'] = a") == frozenset()

    def test_attribute_target_is_not_alias(self):
        assert self._f("o.attr = a") == frozenset()

    def test_none_tree(self):
        assert bare_alias_targets(None) == frozenset()


class TestAliasSkipReason:
    """The alias finding must reach the cacheability verdict via ``skip_reasons``
    — that is the seam ``decide_cacheability`` consults."""

    def test_alias_statement_reports_reason(self):
        analysis = analyze_statement("backup = model", None)
        assert analysis.alias_targets == {'backup'}
        reasons = analysis.skip_reasons({'backup'})
        assert any('Alias assignment' in r for r in reasons), reasons

    def test_non_alias_statement_has_no_alias_reason(self):
        analysis = analyze_statement("backup = model.copy()", None)
        assert analysis.alias_targets == frozenset()
        assert analysis.skip_reasons({'backup'}) == []

    def test_alias_reason_fires_even_when_target_not_in_outputs(self):
        # The refusal is a property of the statement's SHAPE, not of the caller's
        # output set, so it must not depend on how outputs were computed.
        analysis = analyze_statement("b = a", None)
        assert any('Alias assignment' in r for r in analysis.skip_reasons(set()))
