"""Tests for the MutationDetector (Phase 3.4)."""

from cash.notebook.mutation_detector import MutationDetector


class TestMutationDetector:
    """Tests for detecting in-place mutations in code."""

    # --- List mutations ---
    
    def test_list_append(self):
        code = "lst.append(42)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'lst'
        assert mutations[0].method == 'append'
        assert mutations[0].kind == 'method_call'

    def test_list_extend(self):
        code = "data.extend([1, 2, 3])"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'data'
        assert mutations[0].method == 'extend'

    def test_list_sort(self):
        code = "items.sort(key=lambda x: x.name)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'sort'

    def test_list_insert(self):
        code = "lst.insert(0, 'first')"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'insert'

    # --- Dict mutations ---
    
    def test_dict_update(self):
        code = "config.update({'key': 'value'})"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'config'
        assert mutations[0].method == 'update'

    def test_dict_setitem(self):
        code = "d['key'] = value"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'd'
        assert mutations[0].kind == 'subscript_assign'

    def test_dict_pop(self):
        code = "d.pop('key')"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'pop'

    # --- Set mutations ---
    
    def test_set_add(self):
        code = "seen.add(item)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'add'

    # --- Augmented assignments ---
    
    def test_augmented_add(self):
        code = "counter += 1"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'counter'
        assert mutations[0].kind == 'augmented_assign'

    def test_augmented_multiply(self):
        code = "arr *= 2"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'arr'
        assert mutations[0].kind == 'augmented_assign'

    # --- Pandas inplace ---
    
    def test_pandas_fillna_inplace(self):
        code = "df.fillna(0, inplace=True)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'df'
        assert mutations[0].method == 'fillna'
        assert mutations[0].kind == 'inplace_kwarg'

    def test_pandas_dropna_inplace(self):
        code = "df.dropna(inplace=True)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'dropna'
        assert mutations[0].kind == 'inplace_kwarg'

    def test_pandas_sort_inplace(self):
        code = "df.sort_values('col', inplace=True)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].method == 'sort_values'

    def test_pandas_no_inplace(self):
        """Without inplace=True, it's not a mutation."""
        code = "df.fillna(0)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 0

    def test_pandas_inplace_false(self):
        """Explicit inplace=False is not a mutation."""
        code = "df.fillna(0, inplace=False)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 0

    # --- Subscript and attribute assignments ---
    
    def test_subscript_assign(self):
        code = "arr[0] = 100"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'arr'
        assert mutations[0].kind == 'subscript_assign'

    def test_nested_subscript(self):
        code = "matrix[0][1] = 42"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'matrix'

    def test_attribute_assign(self):
        code = "obj.name = 'new_name'"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'obj'
        assert mutations[0].kind == 'attribute_assign'

    # --- Delete ---
    
    def test_del_subscript(self):
        code = "del d['key']"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 1
        assert mutations[0].variable == 'd'
        assert mutations[0].kind == 'subscript_delete'

    # --- Multiple mutations ---
    
    def test_multiple_mutations(self):
        code = """
lst.append(1)
lst.append(2)
d['key'] = 'val'
counter += 1
"""
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 4
        vars_mutated = {m.variable for m in mutations}
        assert 'lst' in vars_mutated
        assert 'd' in vars_mutated
        assert 'counter' in vars_mutated

    # --- No mutations ---
    
    def test_pure_assignment(self):
        code = "x = 42"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 0

    def test_function_call_no_mutation(self):
        code = "result = sorted(lst)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 0

    def test_method_call_with_assignment(self):
        """When result is captured, the original isn't mutated."""
        code = "new_df = df.fillna(0)"
        mutations = MutationDetector.detect_mutations(code)
        assert len(mutations) == 0

    # --- Convenience methods ---
    
    def test_get_mutated_variables(self):
        code = """
lst.append(1)
d['key'] = 'val'
x = 42
"""
        mutated = MutationDetector.get_mutated_variables(code)
        assert mutated == {'lst', 'd'}

    def test_has_mutations_true(self):
        assert MutationDetector.has_mutations("lst.append(1)")

    def test_has_mutations_false(self):
        assert not MutationDetector.has_mutations("x = 42")

    def test_syntax_error(self):
        """Should gracefully handle syntax errors."""
        mutations = MutationDetector.detect_mutations("def :")
        assert mutations == []
