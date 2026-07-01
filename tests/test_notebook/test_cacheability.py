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
    crossref_reassigned_vars,
    function_arg_mutations,
    params_mutated_in_function,
    standalone_call_arg_targets,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
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
