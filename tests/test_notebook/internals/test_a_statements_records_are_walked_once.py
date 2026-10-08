"""A statement that builds records walks them once between its share check,
its closure check and the RAM tier's store, and the tier keeps them as bytes.

Each of the three walked a million nested records on its own, and the RAM
tier then copied them a container at a time: 5.5 s on top of a 1.6 s build.
The walks now share one look (`_plain_data.one_look`) and the store writes
the records out once (`InMemoryBackend`, `_Marshalled`): the whole entry, or
the records alone when numpy's RNG state sits beside them.
"""

from __future__ import annotations

from cash import _plain_data
from cash.backends import memory_backend
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

N = 5_000

BUILD = (
    "import time\n"
    "records = [{'id': i, 'tags': ['a', 'b'], 'meta': {'k': i}} "
    f"for i in range({N}) if i or not time.sleep({ABOVE_PERSISTENCE_FLOOR_S})]"
)


def _ram(cash_instance):
    backend = cash_instance.backend
    return backend.backends[0] if hasattr(backend, "backends") else backend


def test_the_records_rows_are_walked_once_and_kept_as_bytes(cash_magics, mock_shell, cash_instance, monkeypatch):
    rows_walked: list[int] = []
    real = _plain_data._keys_size

    def counting(dicts):
        if len(dicts) == N:
            rows_walked.append(id(dicts[0]))
        return real(dicts)

    monkeypatch.setattr(_plain_data, "_keys_size", counting)
    walked_otherwise = []
    for name in ("tree_levels", "_tree_walk"):
        real_walk = getattr(_plain_data, name)

        def counted(value, *args, _real=real_walk, **kwargs):
            if type(value) is list and len(value) == N:
                walked_otherwise.append(1)
            return _real(value, *args, **kwargs)

        monkeypatch.setattr(_plain_data, name, counted)

    run_cash_cell(cash_magics, BUILD)

    records = mock_shell.user_ns["records"]
    assert rows_walked.count(id(records[0])) == 1
    assert walked_otherwise == []
    ram = _ram(cash_instance)
    kept_as_bytes = [key for key, (_meta, value) in ram._store.items() if type(value) is memory_backend._Marshalled]
    assert kept_as_bytes or ram._holds_bytes, "the records are kept as marshal bytes, whole or as a part"


def test_a_hit_restores_the_records_as_they_were_built(cash_magics, mock_shell):
    run_cash_cell(cash_magics, BUILD)
    built = mock_shell.user_ns["records"]
    built[0]["tags"].append("changed after the store")
    run_cash_cell(cash_magics, BUILD)
    statuses = [m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses[-1] != "COMPUTED", statuses
    restored = mock_shell.user_ns["records"]
    assert restored is not built
    assert restored == [{"id": i, "tags": ["a", "b"], "meta": {"k": i}} for i in range(N)]
