"""Unit tests for the claim-anchor library."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs._claims import AnchorError, Target, parse_anchors, published_pages

FIXTURES = Path(__file__).parent / "_fixtures" / "claims"


def _anchors(name: str):
    page = FIXTURES / name
    return parse_anchors(page.read_text(encoding="utf-8"), page)


def test_parses_every_claim_comment():
    assert len(_anchors("clean.md")) == 4


def test_existence_anchor_has_no_pin_and_no_value():
    a = _anchors("clean.md")[0]
    assert a.targets == (Target("cash/config.py", "CashConfig.compress", None, None),)


def test_fingerprint_anchor_captures_the_pin():
    a = _anchors("clean.md")[1]
    assert a.targets[0].pin == "deadbeef"
    assert a.targets[0].symbol == "compute_cache_key"


def test_one_comment_can_carry_several_targets():
    a = _anchors("clean.md")[2]
    assert [t.symbol for t in a.targets] == ["Cash.cache", "CashConfig.max_cache_size"]
    assert a.targets[1].value == "None"


def test_broad_attribute_is_captured_and_not_parsed_as_a_target():
    a = _anchors("clean.md")[3]
    assert a.broad == "the claim is about the config object as a whole"
    assert len(a.targets) == 1


def test_claim_text_is_the_following_line_not_the_comment():
    a = _anchors("clean.md")[0]
    assert a.claim == "Entries can be stored compressed."


def test_line_number_points_at_the_comment():
    text = (FIXTURES / "clean.md").read_text(encoding="utf-8").splitlines()
    a = _anchors("clean.md")[0]
    assert text[a.line - 1].lstrip().startswith("<!-- claim:")


@pytest.mark.parametrize(
    "body",
    [
        "<!-- claim: -->",
        "<!-- claim: not a path -->",
        "<!-- claim: cash/core.py:Cash.cache @nothex -->",
    ],
)
def test_unparseable_anchors_raise_rather_than_being_skipped(body):
    """Silently dropping a malformed anchor would hide the claim it grounds."""
    with pytest.raises(AnchorError):
        parse_anchors(body, Path("x.md"))


def test_whitespace_around_punctuation_is_insignificant():
    tight = parse_anchors("<!-- claim: cash/core.py:Cash.cache@0badcafe -->", Path("x.md"))
    loose = parse_anchors("<!-- claim:  cash/core.py : Cash.cache @ 0badcafe  -->", Path("x.md"))
    assert tight[0].targets == loose[0].targets


def test_published_pages_excludes_superpowers_and_unbuilt_adr():
    rels = {p.relative_to(Path(__file__).resolve().parents[2] / "docs").as_posix()
            for p in published_pages()}
    assert "architecture_decisions.md" not in rels
    assert not any(r.startswith("superpowers/") for r in rels)
    assert "index.md" in rels
    assert len(rels) == 56
