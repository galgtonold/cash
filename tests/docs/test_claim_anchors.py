"""Ground doc *prose* against source — the layer the fence harness cannot reach.

``test_tutorials.py`` executes python fences; ``test_doc_claims.py`` lints
structured claims (env vars, anchors, config tables). Neither reads prose, and
prose is where every doc failure in this repo has lived: CAS-114 documented a
warning that never emitted, and thread-safety.md's central thesis was inverted
by CAS-112 — both with a green suite throughout.

A claim anchor links a prose claim to the source that decides it:

    <!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
    Cash keys a call on the function source plus its arguments.

This module does NOT prove any doc correct. It proves that nobody has re-read a
claim since the code beneath it changed — a weaker statement, and the strongest
one a machine can actually back.

**Severity split.** Three checks block every PR: an anchor that does not resolve,
a documented literal that disagrees with source, and a page that lost anchors.
All are unambiguous errors needing no judgement. Fingerprint *drift* is
different — "the code moved" is a prompt for a human, and blocking it per-PR
would either stall unrelated code work or teach people to re-pin without
reading, which manufactures false assurance. So drift is advisory here and
blocking at release, where ``CASH_CLAIMS_STRICT=1`` is set.
"""
from __future__ import annotations

import os

import pytest

from tests.docs._claims import Problem, check_manifest, check_page, published_pages

STRICT = os.environ.get("CASH_CLAIMS_STRICT") == "1"


def _all_problems() -> list[Problem]:
    out: list[Problem] = []
    for page in published_pages():
        out.extend(check_page(page))
    return out


def _fmt(problems: list[Problem]) -> str:
    return "\n".join(f"  {p.page}:{p.line} [{p.kind}] {p.message}" for p in problems)


def test_every_anchor_resolves():
    """A renamed or deleted symbol leaves the claim above it ungrounded."""
    bad = [p for p in _all_problems() if p.kind == "unresolved"]
    assert not bad, "Claim anchors point at source that no longer exists:\n" + _fmt(bad)


def test_documented_literals_match_source():
    """A `== <literal>` anchor proves the documented number IS the real one."""
    bad = [p for p in _all_problems() if p.kind == "value"]
    assert not bad, "Documented values disagree with source:\n" + _fmt(bad)


def test_anchors_are_narrow_and_pinned():
    """Class/module anchors are noise; an unfilled `@?` never got verified."""
    bad = [p for p in _all_problems() if p.kind in {"broad", "unpinned"}]
    assert not bad, "Claim anchors need narrowing or pinning:\n" + _fmt(bad)


def test_coverage_ratchet():
    """An audited page may gain anchors, never lose them."""
    bad = check_manifest()
    assert not bad, "Claim coverage regressed:\n" + _fmt(bad)


def test_no_fingerprint_drift():
    """Advisory on a PR, blocking at release (CASH_CLAIMS_STRICT=1).

    Skips rather than fails when advisory, so the queue is visible in the run
    output instead of silently passing.
    """
    drifted = [p for p in _all_problems() if p.kind == "drift"]
    if not drifted:
        return
    message = (
        f"{len(drifted)} doc claim(s) rest on code that has changed. Re-read "
        f"each, then `python scripts/claims.py --accept <page> --yes`:\n"
        + _fmt(drifted)
    )
    if STRICT:
        pytest.fail(message)
    pytest.skip(message)


def test_manifest_covers_every_published_page():
    """A new page must be triaged, not land unanchored and unnoticed."""
    from tests.docs._claims import REPO_ROOT, load_manifest

    manifest = set(load_manifest())
    actual = {p.relative_to(REPO_ROOT).as_posix() for p in published_pages()}
    assert manifest == actual, (
        f"manifest/page mismatch — missing: {sorted(actual - manifest)}, "
        f"stale: {sorted(manifest - actual)}"
    )
