"""``# @cash:assume-safe`` -- a purity waiver scoped to one statement.

``assume_safe=True`` on the decorator silences the whole function permanently.
Audit the call you meant to audit, add an unrelated ``session.post(...)`` three
months later, and nothing says a word: the waiver outlives the audit it was
granted for. Measured -- the analyzer detects the added POST and the flag
suppresses it (``test_the_blanket_flag_hides_a_later_addition`` pins exactly
that, so the hazard is recorded rather than assumed).

A waiver written next to the statement cannot do that. New code arrives
unannotated, so it is reported, and the scope of the exemption is visible in
the diff that grants it.

Placement:

* on the statement's own line, or on its own line directly above it
* on the ``def`` line it waives the FUNCTION-scoped findings instead --
  ``mutable_global`` is a property of the whole body and carries no line
  number, so there is no statement to attach it to.

It is honoured in ``strict=True`` too: a line you audited is audited.
"""
from __future__ import annotations

import os
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpureFunctionError, CashImpurityWarning
from cash.purity_analyzer import ISSUE_MUTABLE_GLOBAL, PurityAnalyzer

from . import audited_helper

REGISTRY: dict = {}


def sink(payload):
    """Stands in for a client method whose name says it writes."""
    return payload


def _writes_through_a_module(fd, data):
    """Never called. It exists so this MODULE contains a write-named call on a
    module global (``os.write``), which is the exact condition that produced
    the false positive -- the scanner is module-scoped, so the call does not
    have to be in the function being analyzed."""
    os.write(fd, data)


def _kinds(fn) -> list[str]:
    return [i.kind for i in PurityAnalyzer().analyze(fn).issues]


def _run(fn, *args, strict: bool = False, assume_safe: bool = False, tmp_path=None):
    """Return 'raised' | 'warned' | ''."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            c.cache(strict=strict, assume_safe=assume_safe)(fn)(*args)
        except CashImpureFunctionError:
            return "raised"
        hits = [w for w in rec if issubclass(type(w.message), CashImpurityWarning)]
    return "warned" if hits else ""


# ------------------------------------------------------- the motivating hazard
def test_the_blanket_flag_hides_a_later_addition(tmp_path):
    """Pins the behaviour this annotation exists to avoid.

    Not a bug report -- ``assume_safe=True`` is documented to silence the
    function. It is here so that if the blanket flag is ever narrowed, this
    test fails and somebody reads the docstring above.
    """
    def audited_then_extended(uid):
        sink({"unaudited": uid})          # added long after the audit
        return uid

    assert _kinds(audited_then_extended), (
        "control: the analyzer must see this call at all"
    )
    assert _run(audited_then_extended, 1, assume_safe=True, tmp_path=tmp_path) == ""


# ------------------------------------------------------------- the annotation
def test_an_annotated_statement_is_waived(tmp_path):
    def audited(uid):
        sink({"audited": uid})            # @cash:assume-safe
        return uid

    assert _kinds(audited) == []
    assert _run(audited, 1, tmp_path=tmp_path) == ""


def test_an_annotation_on_the_line_above_also_waives(tmp_path):
    """The ``# noqa``-above idiom, for when the line is already long."""
    def audited(uid):
        # @cash:assume-safe
        sink({"audited": uid})
        return uid

    assert _kinds(audited) == []


def test_a_statement_added_later_is_reported_again(tmp_path):
    """The whole point: the waiver covers what it was written for, not more."""
    def audited_then_extended(uid):
        sink({"audited": uid})            # @cash:assume-safe
        sink({"added": uid})              # no annotation -- new, unaudited
        return uid

    issues = PurityAnalyzer().analyze(audited_then_extended).issues
    assert len(issues) == 1, [i.description for i in issues]
    assert issues[0].line == 3, "the second statement, not the annotated first"
    assert _run(audited_then_extended, 1, tmp_path=tmp_path) == "warned"


def test_the_def_line_waives_function_scoped_findings_only(tmp_path):
    """``mutable_global`` has no line, so it needs somewhere else to live --
    and waiving it must not quietly waive the statements too."""
    def reads_and_writes(uid):            # @cash:assume-safe
        REGISTRY["seen"] = uid            # a real scope mutation, still flagged
        return REGISTRY.get(uid)

    kinds = _kinds(reads_and_writes)
    assert ISSUE_MUTABLE_GLOBAL not in kinds
    assert kinds, "the def-line waiver must not swallow line-anchored findings"


def test_strict_mode_honours_the_annotation(tmp_path):
    def audited(uid):
        sink({"audited": uid})            # @cash:assume-safe
        return uid

    def not_audited(uid):
        sink({"nope": uid})
        return uid

    assert _run(audited, 1, strict=True, tmp_path=tmp_path) == ""
    assert _run(not_audited, 1, strict=True, tmp_path=tmp_path) == "raised"


def test_it_waives_the_class_that_raises(tmp_path):
    """``untrackable_dep`` raises by default; the per-line waiver is the
    statement-scoped equivalent of ``assume_safe=True`` for it."""
    def audited(name):
        import importlib
        return importlib.import_module(name).__name__   # @cash:assume-safe

    def bare(name):
        import importlib
        return importlib.import_module(name).__name__

    assert _run(audited, "json", tmp_path=tmp_path) == ""
    assert _run(bare, "json", tmp_path=tmp_path) == "raised"


def test_a_multiline_call_is_waived_from_its_OPENING_line(tmp_path):
    """The finding is anchored where the call starts, so the waiver goes there.

    Worth pinning because the closing-paren line is the tempting place to put
    it, and putting it there does nothing.
    """
    def opening(uid):
        sink(                              # @cash:assume-safe
            {"audited": uid},
        )
        return uid

    def closing(uid):
        sink(
            {"audited": uid},
        )                                  # @cash:assume-safe
        return uid

    # Bare statements, not `return sink(...)`: a call whose value is USED is
    # not a finding at all, so the first version of this test passed for the
    # wrong reason -- both arms were clean because neither had anything to
    # waive.
    assert _kinds(opening) == []
    assert _kinds(closing), "the closing line is not where the finding lives"


def test_an_unrelated_comment_does_not_waive(tmp_path):
    """Only the directive counts -- a passing mention must not silence."""
    def looks_similar(uid):
        sink({"x": uid})                  # assume this is safe, cash
        return uid

    assert _kinds(looks_similar), "a passing mention must not act as a waiver"


def test_a_waiver_in_a_HELPER_covers_that_helper_for_every_caller():
    """The waiver lives in the source of the function that has the finding.

    So a helper's own waiver applies wherever it is called from -- which is
    right (the audit is about the helper's body) but worth knowing: you cannot
    accept a helper's side effect for one caller only.
    """
    def calls_audited(uid):
        return audited_helper.audited_helper(uid)

    def calls_unaudited(uid):
        return audited_helper.unaudited_helper(uid)

    assert _kinds(calls_audited) == []
    assert _kinds(calls_unaudited), "control: the unwaived helper is still reported"


# ------------------------- the module-global false positive this work exposed
def test_a_write_named_call_on_a_MODULE_does_not_flag_the_module(tmp_path):
    """``json.dump(...)`` does not mutate ``json``.

    The mutation scanner treated any write-METHOD call on a global as mutating
    that global, so one such call anywhere in a file made every function in it
    that merely READ the module report "reads module global 'x' that is
    reassigned or mutated elsewhere". Measured on a sibling function that did
    not make the call itself -- and on a finding with no line number, so the
    annotation above could not even waive it per statement.
    """
    def only_reads():
        return os.getpid()

    assert ISSUE_MUTABLE_GLOBAL not in _kinds(only_reads), (
        "`os.write` in a sibling function must not make `os` a mutated global"
    )


def test_a_real_container_mutation_still_flags(tmp_path):
    """Control arm: the exclusion is for modules, not for everything.

    ``REGISTRY`` is a module-level dict this file mutates inside a function, so
    reading it must still be reported -- otherwise the fix above would have
    turned the whole rule off.
    """
    def reads_registry():
        return REGISTRY.get("seen")

    def writer():
        REGISTRY["seen"] = 1

    assert ISSUE_MUTABLE_GLOBAL in _kinds(reads_registry)
