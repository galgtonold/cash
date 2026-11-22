"""With numpy loaded, the walks for array views and borrowed memory take a
list of records holding values alone in one step, and find what the walk a
record at a time finds.

Both walks run only once numpy is imported. Over a list of 300,000
``SimpleNamespace`` records, or of dicts each holding a small set of tags,
they took a Python step per record, ran past the share check's budget and
ended the statement in NOTEBOOK-SHARE-UNCHECKED, while the share check's own
walk had taken the same records at C speed. A kernel that had imported
numpy for any earlier cell hit it; one that had not, did not. These tests
build many shapes at random, some with a view or a borrowing frame in one
record, and compare the verdict with the step off.
"""

from __future__ import annotations

import random
import types

import pytest

from cash.notebook import borrowed_memory, shared_objects
from cash.notebook.borrowed_memory import borrowed_memory_names
from cash.notebook.shared_objects import view_base_names

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


class Sub(types.SimpleNamespace):
    pass


class Plain:
    def __init__(self, i):
        self.i = i
        self.x = i * 0.5


def _records(seed: int, big, other):
    """A list of records of one random shape, a few of them holding a view
    of *big* or a pandas Series over *other*'s memory when the seed says so."""
    rng = random.Random(seed)
    shape = rng.choice(["namespace", "sub", "plain", "dict_sets", "dict_values", "mixed"])
    records: list = []
    for i in range(rng.randint(32, 150)):
        if shape == "namespace":
            records.append(types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}"))
        elif shape == "sub":
            records.append(Sub(i=i, x=i * 0.5))
        elif shape == "plain":
            records.append(Plain(i))
        elif shape == "dict_sets":
            records.append({"i": i, "tags": {i % 7, i % 11}, "pair": (i, "a"), "fz": frozenset({i})})
        elif shape == "dict_values":
            records.append({"i": i, "x": i * 0.5})
        else:
            records.append(rng.choice([i, "s", 1.5, None, types.SimpleNamespace(i=i)]))
    for _ in range(rng.choice([0, 0, 1, 2])):
        held = rng.choice([big[rng.randrange(4) :], pd.Series(other, copy=False), [big[1:]], {big[2:3].tobytes()}])
        at = rng.randrange(len(records))
        record = records[at]
        if isinstance(record, dict):
            record[rng.choice(["view", "tags"])] = held
        elif isinstance(record, (types.SimpleNamespace, Plain)):
            record.view = held
        else:
            records[at] = types.SimpleNamespace(view=held)
    return records


@pytest.fixture
def one_at_a_time(monkeypatch):
    def off():
        no = lambda *args: False
        monkeypatch.setattr(shared_objects, "records_hold_values", no)
        monkeypatch.setattr(borrowed_memory, "records_hold_values", no)

    return off


@pytest.mark.parametrize("seed", range(200))
def test_the_step_finds_the_bases_and_borrowers_the_walk_finds(seed, one_at_a_time):
    big = np.arange(10.0)
    other = np.arange(5.0)
    records = _records(seed, big, other)
    ns = {"records": records, "big": big, "other": other}
    fast = (view_base_names([records], ns), borrowed_memory_names([records], ns))
    one_at_a_time()
    assert (view_base_names([records], ns), borrowed_memory_names([records], ns)) == fast


@pytest.mark.parametrize(
    "make",
    [
        lambda i, held: types.SimpleNamespace(i=i, x=held if i == 40 else i * 0.5),
        lambda i, held: {"i": i, "tags": held if i == 40 else {i % 7}},
        lambda i, held: {"i": i, "tags": {i % 7}, "x": held} if i == 40 else {"i": i, "tags": {i % 7}},
    ],
    ids=["namespace", "dict tags", "dict extra key"],
)
def test_a_view_in_one_record_of_many_is_still_found(make):
    big = np.arange(10.0)
    other = np.arange(5.0)
    ns = {"big": big, "other": other}
    assert view_base_names([[make(i, big[2:]) for i in range(5000)]], ns) == {"big"}
    assert borrowed_memory_names([[make(i, pd.Series(other, copy=False)) for i in range(5000)]], ns) == {"other"}


@pytest.mark.parametrize(
    "make",
    [
        lambda i: types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}"),
        lambda i: {"i": i, "tags": {i % 7, i % 11}},
    ],
    ids=["namespaces", "dicts holding sets"],
)
def test_a_long_list_of_records_is_not_walked_a_record_at_a_time(monkeypatch, make):
    records = [make(i) for i in range(20_000)]
    ns = {"records": records, "big": np.arange(10.0)}
    asked = []
    for module in (shared_objects, borrowed_memory):
        real = module.children_of
        monkeypatch.setattr(module, "children_of", lambda value, real=real: asked.append(1) or real(value))

    assert view_base_names([records], ns) == set()
    assert borrowed_memory_names([records], ns) == set()
    assert len(asked) < 100
