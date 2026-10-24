"""A dict or list subclass whose own methods raise runs as Python runs it.

``class Bad(dict): def values(self): raise ...`` -- cash looked into the
value it bound through the class's own ``values()``, ``keys()``,
``items()`` or ``__iter__``, which plain Python never calls here, and the
cell failed with the class's error (``cacheability_decision._container_items``,
then the shared-object, held-function, size and holder walks). What a
builtin container holds is read through the builtin type's own methods; a
value the store's checks still cannot read is not stored, and the reason
says why.
"""

from __future__ import annotations

import pytest

from cash.analysis.annotations import CacheAnnotation

_PERSIST = CacheAnnotation(persist=True)

CLASSES = (
    "class Bad(dict):\n    def values(self):\n        raise RuntimeError('values raised')\n",
    "class D(dict):\n"
    "    def values(self):\n        raise RuntimeError('values raised')\n"
    "    def items(self):\n        raise RuntimeError('items raised')\n"
    "    def keys(self):\n        raise RuntimeError('keys raised')\n"
    "    def __iter__(self):\n        raise RuntimeError('iter raised')\n",
    "class L(list):\n"
    "    def __iter__(self):\n        raise RuntimeError('iter raised')\n"
    "    def __len__(self):\n        raise RuntimeError('len raised')\n",
)


@pytest.mark.parametrize(
    ("make", "read", "expected"),
    [
        ("bad = Bad(a=1)", "n = dict.__getitem__(bad, 'a') + 1", 2),
        ("bad = D(a=1, b=[1, 2])", "n = dict.__len__(bad) + 1", 3),
        ("bad = L([1, 2])", "n = list.__len__(bad) * 2", 4),
        ("bad = {'inner': D(a=1), 'items': L([3])}", "n = dict.__getitem__(bad['inner'], 'a')", 1),
    ],
    ids=["values_raises", "every_dict_method_raises", "iter_and_len_raise", "held_in_a_dict"],
)
def test_the_cell_runs_and_reruns(statement_processor, mock_shell, make, read, expected):
    for setup in CLASSES:
        statement_processor.process_statement(setup)
    for _ in range(2):
        statement_processor.process_statement(make, annotation=_PERSIST)
        assert "bad" in mock_shell.user_ns
        statement_processor.process_statement(read, annotation=_PERSIST)
        assert mock_shell.user_ns["n"] == expected


PROXIES = {
    "wrapt_dict": "import wrapt\nm = wrapt.ObjectProxy({'a': [1, 2]})",
    "wrapt_list": "import wrapt\nm = wrapt.ObjectProxy([[1], [2]])",
    "mock_spec_dict": "from unittest.mock import MagicMock\nm = MagicMock(spec=dict)",
    "class_property": (
        "class Claims:\n"
        "    def __init__(self):\n        self.rows = [1, 2]\n"
        "    @property\n    def __class__(self):\n        return dict\n"
        "m = Claims()"
    ),
}


@pytest.mark.parametrize("make", PROXIES.values(), ids=PROXIES.keys())
def test_a_value_that_only_claims_to_be_a_dict_or_list_runs(statement_processor, mock_shell, make):
    """A proxy passes ``isinstance`` for a dict or list but holds none of its
    storage: reading it through the builtin type's methods failed the cell
    (``generator raised StopIteration``). It is read as what it really is."""
    pytest.importorskip("wrapt")
    for _ in range(2):
        statement_processor.process_statement(make)
        statement_processor.process_statement("n = m")
        statement_processor.process_statement("k = [m, 1]")
        statement_processor.process_statement("size = len(k)")
        assert mock_shell.user_ns["size"] == 2
        assert mock_shell.user_ns["k"][0] is mock_shell.user_ns["m"]
