"""A module object the notebook builds itself (``types.ModuleType('m')``)
is a value like any other: setting an attribute on it changes it. A rebuild
of it runs the statements that set its attributes too, whether they follow
on the same line or on the next one. Stored, as a slow machine stores
even these quick statements, it is not restored: no import gives it back
by its name, and the name stayed unbound."""

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

SETUPS = {
    "on_the_same_line": "import types\ndef setz(d):\n    d['z'] = 1\nsmod = types.ModuleType('smod'); smod.setz = setz",
    "on_the_next_line": "import types\ndef setz(d):\n    d['z'] = 1\nsmod = types.ModuleType('smod')\nsmod.setz = setz",
}
FRAME = "import pandas as pd\ndf = pd.DataFrame({'o': [{'i': i} for i in range(10)]})"
READER = "_ = df['o'].apply(smod.setz)\nr = df['o'][3]"


@pytest.mark.parametrize("setup", list(SETUPS))
def test_a_module_rebuilt_after_it_is_gone_has_its_attribute(cash_magics, setup):
    cells = [SETUPS[setup], FRAME, READER]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    del cash_magics.shell.user_ns["smod"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["r"] == {"i": 3, "z": 1}


@pytest.mark.parametrize("setup", list(SETUPS))
def test_a_module_built_by_a_stored_statement_is_built_again(cash_magics, monkeypatch, setup):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    test_a_module_rebuilt_after_it_is_gone_has_its_attribute(cash_magics, setup)


#: A module subclass with lookups of its own, next to a dict subclass an
#: ``exec`` defines a function in: the upstream check rebuilt the module.
SPIES = """import types
CALLS = []
class GA:
    def __getattr__(self, name):
        CALLS.append(('GA.__getattr__', name)); raise AttributeError(name)
    def __call__(self, d):
        return None
class SpyModule(types.ModuleType):
    def __getattribute__(self, name):
        CALLS.append(('SpyModule.__getattribute__', name)); return super().__getattribute__(name)
    def __getattr__(self, name):
        CALLS.append(('SpyModule.__getattr__', name)); raise AttributeError(name)
class SpyDict(dict):
    def __getitem__(self, k):
        CALLS.append(('SpyDict.__getitem__', k)); return dict.__getitem__(self, k)
def setz(d):
    d['z'] = 1
smod = SpyModule('smod'); smod.setz = setz
ga = GA()
g2 = SpyDict(setz=setz, ga=ga)
exec("def u_sg(d):\\n    setz(d)\\n    ga(d)", g2)"""


def test_a_module_subclass_keeps_what_was_set_on_it(cash_magics):
    cells = [SPIES, FRAME, READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["r"] == {"i": 3, "z": 1}
