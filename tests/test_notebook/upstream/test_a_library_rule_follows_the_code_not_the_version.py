"""A rule that rests on a library's pickling code follows the code.

Two rules let an output be stored although it keeps a slice of an anchored
array: scikit-learn's ``BaseEstimator`` and numpy's ``MaskedArray`` pickle
in a way cash read. They were tied to one exact release each (``==`` on
the version string), so on every other release, the next patch release
included, the statement ran every time and nothing said why. They now hold
for the text of the functions that were read (`cash.notebook.code_read`):
another version number changes nothing, an edited function turns the rule
off, and ``%cash_status`` says which.
"""

from __future__ import annotations

import pytest

from cash.notebook import anchors

np = pytest.importorskip("numpy")
# A rule looks only at modules the process has loaded.
pytest.importorskip("numpy.ma")


@pytest.fixture
def code_read():
    from cash.notebook import code_read

    return code_read


def _needs(code_read, rule: str) -> None:
    if not code_read.is_read(rule):
        pytest.skip(f"the installed library's code for {rule} is not one cash read: {code_read.status()[rule]}")


def _masked_beside_a_slice():
    X = np.arange(24.0).reshape(6, 4)
    return {"m": np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])}, {"X_train": X[:4]}


def test_another_version_number_over_the_same_code_keeps_the_masked_array_rule(code_read, monkeypatch):
    _needs(code_read, "masked_array_pickling")
    monkeypatch.setattr(np, "__version__", "99.0.0")
    value, anchored = _masked_beside_a_slice()
    assert anchors._masked_array() is np.ma.MaskedArray
    assert anchors.array_memory_held_apart(value, anchored) is None
    assert code_read.status()["masked_array_pickling"].startswith("on (")
    assert "installed: 99.0.0" in code_read.status()["masked_array_pickling"]


def test_an_edited_pickling_function_turns_the_masked_array_rule_off(code_read, monkeypatch):
    _needs(code_read, "masked_array_pickling")

    def __reduce__(self):  # another text than the one read
        return (np.ma.core._mareconstruct, (self.__class__, self._baseclass, (0,), "b"), self.__getstate__())

    monkeypatch.setattr(np.ma.MaskedArray, "__reduce__", __reduce__)
    value, anchored = _masked_beside_a_slice()
    assert anchors._masked_array() is None
    assert anchors.array_memory_held_apart(value, anchored) == "X_train"
    line = code_read.status()["masked_array_pickling"]
    assert line.startswith("off: numpy ") and "runs every time" in line
    monkeypatch.undo()
    # Judged again when the function is the one read again.
    assert anchors._masked_array() is np.ma.MaskedArray


def test_another_version_number_over_the_same_code_keeps_the_estimator_rule(code_read, monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    base = pytest.importorskip("sklearn.base").BaseEstimator
    _needs(code_read, "estimator_pickling")
    monkeypatch.setattr(sklearn, "__version__", "99.0.0")
    assert anchors._sklearn_pickling() == {
        "__getstate__": vars(base)["__getstate__"],
        "__setstate__": vars(base)["__setstate__"],
    }


def test_an_edited_pickling_function_turns_the_estimator_rule_off(code_read, monkeypatch):
    base = pytest.importorskip("sklearn.base").BaseEstimator
    _needs(code_read, "estimator_pickling")

    def __getstate__(self):
        return dict(self.__dict__, extra=1)

    monkeypatch.setattr(base, "__getstate__", __getstate__)
    assert anchors._sklearn_pickling() == {}
    assert code_read.status()["estimator_pickling"].startswith("off: sklearn ")


def _first(value):
    total = value + 1
    return total


class _Holder:
    def first(value):
        total = value + 1
        return total


def _other(value):
    total = value + 2
    return total


def test_the_digest_is_of_the_text_whatever_its_indentation(code_read):
    one = code_read.source_digest([_first])
    assert one is not None and len(one) == 64
    assert one != code_read.source_digest([_other])
    assert one != code_read.source_digest([_first, _other])
    # The same lines one level deeper, under another name: the name is in
    # the text, so it differs; the indentation alone would not.
    method = code_read.source_digest([_Holder.first])
    assert method != one
    assert method == code_read.source_digest([vars(_Holder)["first"]])


def test_a_function_with_no_text_has_no_digest_and_its_rule_is_off(code_read, monkeypatch):
    assert code_read.source_digest([len]) is None
    assert code_read.source_digest([None]) is None
    rule = code_read.Rule(library="builtins", functions=lambda: [len], accepted={}, without="it runs every time")
    monkeypatch.setitem(code_read.RULES, "compiled", rule)
    assert not code_read.is_read("compiled")
    assert code_read.functions_read("compiled") == []
    assert code_read.status()["compiled"].startswith("off: builtins ")


def test_a_rule_whose_library_is_not_loaded_says_so(code_read, monkeypatch):
    rule = code_read.Rule(library="nowhere", functions=lambda: None, accepted={"x": "1.0"}, without="it runs")
    monkeypatch.setitem(code_read.RULES, "absent", rule)
    assert not code_read.is_read("absent")
    assert code_read.status()["absent"] == "not loaded"


def test_a_rule_is_on_for_an_accepted_text_and_reads_the_file_once(code_read, monkeypatch):
    accepted = {code_read.source_digest([_first, _other]): "example 1.0"}
    rule = code_read.Rule(library="example", functions=lambda: [_first, _other], accepted=accepted, without="it runs")
    monkeypatch.setitem(code_read.RULES, "example", rule)
    assert code_read.is_read("example")
    assert code_read.functions_read("example") == [_first, _other]
    calls = []
    real = code_read.source_digest
    monkeypatch.setattr(code_read, "source_digest", lambda functions: calls.append(1) or real(functions))
    for _ in range(5):
        assert code_read.is_read("example")
    assert calls == []
    code_read._JUDGED.pop("example", None)


def test_cash_status_lists_each_library_rule(code_read, cash_magics):
    rules = cash_magics.cash_status("dict")["library_rules"]
    assert set(rules) == set(code_read.RULES)
    assert all(line.startswith(("on (", "off: ")) or line == "not loaded" for line in rules.values())
