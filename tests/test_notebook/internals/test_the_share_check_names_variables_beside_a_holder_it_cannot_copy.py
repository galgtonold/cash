"""The share check names the variables it found when it gives up.

A holder that is not a variable (a function's defaults, a library object)
makes `share_group` give up: the statement is not stored. The variables
found beside that holder are still handed back (`GaveUp`), paired with the
outputs they share an object with, so the statement links them.
"""

from __future__ import annotations

import argparse

from cash.notebook.shared_objects import GaveUp, share_group


def test_a_function_default_beside_a_variable():
    features = ["a", "b"]

    def pick(cols=features):
        return cols

    ns = {"features": features, "pick": pick, "config": {"features": features}, "unrelated": ["a"]}
    gave_up = GaveUp()
    holders, shared = share_group(["config"], {"config": ns["config"]}, ns, gave_up=gave_up)
    assert holders == {}
    assert shared == {"config"}
    # The function whose default holds the list too: what reads it reads
    # the list (``pick()``).
    assert gave_up.names == {"features", "pick"}
    assert gave_up.complete
    assert any("features" in held for _owners, held in gave_up.links)


def test_a_variable_bound_to_the_holder_itself():
    """``args = argparse.Namespace(features=features)``: the namespace is no
    container the check opens, but a variable binds it."""
    features = ["a", "b"]
    ns = {"features": features, "args": argparse.Namespace(features=features)}
    gave_up = GaveUp()
    _holders, shared = share_group(["features"], {"features": features}, ns, gave_up=gave_up)
    assert shared == {"features"}
    assert gave_up.names == {"args"}
    assert gave_up.links == [(frozenset({"features"}), frozenset({"args"}))]


def test_without_a_record_nothing_changes():
    features = ["a", "b"]
    ns = {"features": features, "args": argparse.Namespace(features=features)}
    assert share_group(["features"], {"features": features}, ns) == ({}, {"features"})


def test_a_class_attribute_holding_the_object_names_the_class():
    """``Reg.items.append(rows)``: what reads ``Reg.items`` reads ``rows``."""
    rows = [1, 2]

    class Reg:
        items = [rows]

    ns = {"rows": rows, "Reg": Reg}
    gave_up = GaveUp()
    share_group(["rows"], {"rows": rows}, ns, gave_up=gave_up)
    assert gave_up.names == {"Reg"}
