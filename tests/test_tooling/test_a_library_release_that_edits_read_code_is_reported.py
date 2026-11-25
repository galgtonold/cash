"""A library release that edits code a rule was read in is reported.

Two notebook rules hold for the text of scikit-learn's and numpy's
pickling functions as it was read (``cash.notebook.code_read``). On a
release that edits them the rule is off, which is safe and silent: the
user's statement runs every time. ``tools/library_code_read.py`` is what
tells a maintainer, in the nightly run on the newest releases: it exits 1
and prints the digest to add once the new text has been read.
"""

from __future__ import annotations

import pytest

from tools import library_code_read


@pytest.fixture
def code_read():
    from cash.notebook import code_read

    return code_read


def _one(value):
    return value + 1


def _two(value):
    return value + 2


@pytest.fixture
def one_rule(code_read, monkeypatch):
    """`RULES` holding one rule over this module's two functions."""

    def use(accepted: dict[str, str]):
        rule = code_read.Rule(library="pytest", functions=lambda: [_one, _two], accepted=accepted, without="it runs")
        monkeypatch.setattr(code_read, "RULES", {"example": rule})
        monkeypatch.setattr(library_code_read, "IMPORTS", {"example": "pytest"})
        code_read._JUDGED.pop("example", None)
        return code_read.source_digest([_one, _two])

    yield use
    code_read._JUDGED.pop("example", None)


def test_code_that_was_read_passes(one_rule, capsys):
    digest = one_rule({})
    one_rule({digest: "example 1.0"})
    assert library_code_read.main() == 0
    assert capsys.readouterr().out.startswith("example: on (as read in example 1.0; installed: ")


def test_code_that_was_not_read_fails_with_the_digest_to_add(one_rule, capsys):
    digest = one_rule({"0" * 64: "example 0.9"})
    assert library_code_read.main() == 1
    out = capsys.readouterr().out
    assert "::error::example is off: read _one, _two in the installed pytest again" in out
    assert f"add the digest {digest} to cash.notebook.code_read.RULES" in out


def test_a_rule_with_no_import_listed_fails_instead_of_passing_unloaded(code_read, monkeypatch, capsys):
    """A rule whose library the script never imports would read "not
    loaded" and be skipped: a new rule must be listed."""
    rule = code_read.Rule(library="pytest", functions=lambda: None, accepted={}, without="it runs")
    monkeypatch.setattr(code_read, "RULES", {**code_read.RULES, "new_rule": rule})
    assert library_code_read.main() == 1
    assert "no import listed here for the rules ['new_rule']" in capsys.readouterr().out


def test_every_rule_has_its_import_listed(code_read):
    assert set(library_code_read.IMPORTS) == set(code_read.RULES)
