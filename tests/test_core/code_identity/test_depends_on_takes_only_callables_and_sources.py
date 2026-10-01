"""``depends_on=`` takes callables and `DataSource` objects, and says so.

Any other entry was skipped without a word. ``depends_on=["data.txt"]``, meant
as ``file_depends_on``, added nothing to the key: after data.txt changed, the
old result was served. ``depends_on="data.txt"`` was iterated character by
character, and ``depends_on=some_function`` raised "'function' object is not
iterable".
"""

from __future__ import annotations

import pathlib

import pytest

from cash import DataSource


def helper():
    return 1


class _Source(DataSource):
    def get_id(self):
        return "src"

    def state_token(self):
        return "v1"


@pytest.mark.parametrize(
    "bad, hint",
    [
        (["data.txt"], "file_depends_on='data.txt'"),
        ("data.txt", "file_depends_on='data.txt'"),
        ([pathlib.Path("data.txt")], "file_depends_on="),
        ([42], "wrap the value in a DataSource"),
        ([helper, None], "not NoneType"),
    ],
)
def test_an_entry_that_is_neither_raises_at_decoration(cash_instance, bad, hint):
    with pytest.raises(TypeError, match="depends_on takes callables and DataSource objects") as info:

        @cash_instance.cache(depends_on=bad)
        def load():
            return 1

    assert hint in str(info.value)


@pytest.mark.parametrize("single", [helper, _Source()])
def test_one_callable_or_source_on_its_own_is_a_list_of_one(cash_instance, single):
    @cash_instance.cache(depends_on=single)
    def load():
        return 1

    assert load() == 1
    [name] = [n for n in cash_instance.functions if n.endswith("load")]
    assert cash_instance.graph.get_dependencies(name), "the dependency was not recorded"


def test_a_list_of_both_kinds_is_accepted(cash_instance):
    @cash_instance.cache(depends_on=(helper, _Source()))
    def load():
        return 1

    assert load() == 1


def test_file_depends_on_takes_a_path_object(cash_instance, tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("v1", encoding="utf-8")

    @cash_instance.cache(file_depends_on=data)
    def load():
        return 1

    assert load() == 1
