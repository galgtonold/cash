"""A cell's objects carry no cash attributes: ``vars()`` and ``==`` are untouched.

The notebook records a lineage for every object a cell binds. Written into the
object's ``__dict__``, it showed up in ``vars(args)``, ``json.dumps(vars(cfg))``
and made two equal ``SimpleNamespace`` objects compare unequal.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from cash.lineage_tag import own_tag
from tests._cell_driver import run_cash_cell


def test_objects_a_cell_binds_keep_their_own_attributes(cash_magics, mock_shell):
    mock_shell.user_ns.update(SimpleNamespace=SimpleNamespace, argparse=argparse)
    run_cash_cell(cash_magics, "class Config:\n    def __init__(self, **kw):\n        self.__dict__.update(kw)")
    run_cash_cell(
        cash_magics,
        "ns = SimpleNamespace(lr=0.1)\ncfg = Config(lr=0.1, epochs=3)\nargs = argparse.Namespace(lr=0.1)",
    )
    ns = mock_shell.user_ns
    assert ns["ns"] == SimpleNamespace(lr=0.1)
    assert json.dumps(vars(ns["cfg"])) == '{"lr": 0.1, "epochs": 3}'
    assert vars(ns["args"]) == {"lr": 0.1}
    # The lineage is still kept, beside the object.
    assert own_tag(ns["cfg"]) == cash_magics.tracking_state.variable_lineage["cfg"]
