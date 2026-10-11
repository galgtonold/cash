"""``len(df)`` and ``print(df)`` do not hash the frame before and after.

A bare call hands its name arguments to a callee that might change them, so
each is fingerprinted by content before and after the statement. For a frame
of 1.7 million parsed rows that is a hash of every value, twice: 14 s for
``len(df)``. The unshadowed builtins ``len``, ``print`` and a few more read a
pandas, numpy or builtin value and change nothing.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.namespace_effects import bare_call_arguments

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _candidates(code: str, **ns) -> frozenset[str]:
    return bare_call_arguments(ast.parse(code), ns)


@pytest.mark.parametrize("code", ["len(df)", "print(df)", "print('rows', df)", "repr(df)", "type(df)", "str(df)"])
def test_a_reading_builtin_on_a_frame_is_no_candidate(code):
    assert _candidates(code, df=pd.DataFrame({"a": [1]})) == frozenset()


def test_arrays_lists_and_dicts_are_read_without_a_fingerprint_too():
    ns = {"arr": np.arange(3), "items": [1, 2], "mapping": {"a": 1}}
    assert _candidates("print(arr, items, mapping)", **ns) == frozenset()


def test_another_callee_still_gets_its_arguments_fingerprinted():
    df = pd.DataFrame({"a": [1]})
    assert _candidates("clean(df)", df=df, clean=lambda x: x) == frozenset({"df"})


def test_a_notebooks_own_type_is_not_trusted_to_have_a_pure_len():
    class Thing:
        def __len__(self):
            return 0

    assert _candidates("len(thing)", thing=Thing()) == frozenset({"thing"})


def test_a_frame_beside_an_object_of_a_notebooks_own_class_is_still_a_candidate():
    class Thing:
        pass

    assert _candidates("print(df, thing)", df=pd.DataFrame({"a": [1]}), thing=Thing()) == frozenset({"df", "thing"})


def test_a_shadowed_builtin_is_not_trusted():
    df = pd.DataFrame({"a": [1]})
    assert _candidates("len(df)", df=df, len=lambda x: x) == frozenset({"df"})


def test_print_to_a_file_may_write_so_its_handle_is_a_candidate(tmp_path):
    with open(tmp_path / "f.txt", "w", encoding="utf-8") as handle:
        assert _candidates("print(df, file=handle)", df=pd.DataFrame({"a": [1]}), handle=handle) == frozenset(
            {"df", "handle"}
        )


def test_a_read_only_library_function_is_no_candidate():
    """``np.quantile(deltas, 0.95)`` over 1.7 million values took 38.7 s
    against 0.4 s plain: the list was pickled before and after."""
    ns = {"np": np, "deltas": [np.timedelta64(1, "s")] * 3, "df": pd.DataFrame({"a": [1]})}
    assert _candidates("np.quantile(deltas, 0.95)", **ns) == frozenset()
    assert _candidates("np.mean(deltas)", **ns) == frozenset()


def test_a_function_of_the_same_name_is_not_trusted():
    fake = type("np", (), {"quantile": staticmethod(lambda x, q: x)})
    assert _candidates("np.quantile(deltas, 0.95)", np=fake, deltas=[1, 2]) == frozenset({"deltas"})


def test_a_read_only_library_function_on_a_notebooks_own_type_is_a_candidate():
    class Thing:
        def __array__(self, dtype=None, copy=None):
            return np.zeros(2)

    assert _candidates("np.mean(thing)", np=np, thing=Thing()) == frozenset({"thing"})
