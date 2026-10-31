"""A cell-run plan never restores a name that a statement running after the
restores binds again.

`NotebookSimulator.plan_cell_run` restores first and then runs the rest of
the cell in order. In handson-ml3's chapter 7 cell::

    housing = load_housing_data()
    train_set, test_set = train_test_split(housing, ...)
    housing_labels = train_set["median_house_value"]
    housing = train_set.drop("median_house_value", axis=1)

a Run All in the same kernel restored the drop's ``housing`` and then ran
the load, which writes files and so always runs: ``housing`` kept the
unsplit 20,640 rows, and the next cell's fit failed against the 16,512
labels. Such a plan is refused and the cell runs as usual.
"""

from types import SimpleNamespace

from cash.notebook.upstream.simulator import _plan_makes_each_name


def _entry(code, outputs):
    return SimpleNamespace(stmt_code=code, outputs=set(outputs))


TRACE = [
    _entry("housing = load_housing_data()", ["housing"]),
    _entry("train_set, test_set = split(housing)", ["train_set", "test_set"]),
    _entry("housing_labels = train_set['v']", ["housing_labels"]),
    _entry("housing = train_set.drop('v', axis=1)", ["housing"]),
]


def test_a_restore_an_earlier_run_would_overwrite_is_refused():
    restored = {3: {"restored_vars": ["housing"]}}
    assert not _plan_makes_each_name(TRACE, {"housing"}, {0, 1, 2}, restored)


def test_a_restore_nothing_after_it_rebinds_stands():
    restored = {3: {"restored_vars": ["housing"]}, 1: {"restored_vars": ["train_set", "test_set"]}}
    assert _plan_makes_each_name(TRACE, {"housing"}, {2}, restored)
