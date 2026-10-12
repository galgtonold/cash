"""What the methods of a class made inside a factory captured is keyed,
also when Python calls them for the body: ``lookup[k]``, ``len(lookup)``,
``lookup.size``.

``lookup = make_lookup(table)`` defines its class in the factory, so its
``__getitem__`` reads ``table`` from a closure cell. Called by name
(``lookup.get(k)``) the helper walk keyed the cell; called through ``[]``,
``len()`` or a property nothing did, and a changed table served the old
result. The instance itself pickles by no name (KEY-UNHASHABLE-GLOBAL says
its own state is not keyed); its class's code and captures still are.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import warnings

import pytest

pytestmark = pytest.mark.core

MODULE = textwrap.dedent("""
    def make_lookup(table):
        class Lookup:
            def __getitem__(self, k):
                return table[k]
            def __len__(self):
                return len(table)
            @property
            def size(self):
                return len(table)
        return Lookup

    T = {"a": 1, "b": 2}
    Lookup = make_lookup(T)
    lookup = Lookup()

    def through_an_instance(k):
        return lookup[k], len(lookup), lookup.size
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"lookups_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)


def test_an_instance_used_through_dunders_and_a_property(cash_instance, mod):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn = cash_instance.cache(mod.through_an_instance)
        assert fn("a") == (1, 2, 2)
        mod.T["a"] = 9
        mod.T["z"] = 0
        assert fn("a") == fn.__wrapped__("a") == (9, 3, 3)
