"""A ``frozen=True`` result bound by a statement keeps the decorator's tag
while the statement's lineage stays; another lineage (a change the notebook
saw) gives it the statement's tag, which the decorator keys it by.
"""

from __future__ import annotations

from cash.lineage_tag import own_tag, set_tags
from cash.notebook.lineage_store import frozen_tag, restore_frozen_tag, statement_tag, tag_value


class Model:
    pass


def _frozen() -> Model:
    model = Model()
    set_tags(model, _cash_lineage_src="frozen", _cash_lineage_hash="call", _cash_lineage_producer="lib.train")
    return model


def test_binding_keeps_the_frozen_tag_and_reports_the_statements():
    model = _frozen()
    tag_value(model, "stmt")
    tag_value(model, "stmt")
    assert own_tag(model) == "call"
    assert own_tag(model, "_cash_lineage_src") == "frozen"
    assert statement_tag(model) == "stmt"


def test_another_lineage_gives_it_the_statements_tag():
    model = _frozen()
    tag_value(model, "stmt")
    tag_value(model, "stmt-after-a-change")
    assert own_tag(model) == "stmt-after-a-change"
    assert own_tag(model, "_cash_lineage_src") == "statement"
    assert frozen_tag(model) is None


def test_a_restored_value_takes_the_frozen_tag_it_was_stored_with():
    stored = frozen_tag(_frozen())
    assert stored == ["call", "lib.train"]
    restored = Model()
    restore_frozen_tag(restored, stored)
    tag_value(restored, "stmt")
    assert own_tag(restored) == "call"
    assert own_tag(restored, "_cash_lineage_producer") == "lib.train"
    assert statement_tag(restored) == "stmt"


def test_a_malformed_stored_tag_is_ignored():
    restored = Model()
    restore_frozen_tag(restored, {"not": "a tag"})
    assert own_tag(restored) is None
