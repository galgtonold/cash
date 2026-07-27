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


# --------------------------------------------------------------------------- #
# Resolution and fingerprinting                                               #
# --------------------------------------------------------------------------- #
import ast  # noqa: E402

from tests.docs._claims import fingerprint, normalize, resolve  # noqa: E402


def test_resolves_a_method_through_its_class():
    nodes, _ = resolve(Target("cash/core.py", "Cash.cache"))
    assert all(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes)
    assert {n.name for n in nodes} == {"cache"}


def test_overloads_all_resolve_not_just_the_first():
    """Cash.cache is two @overload stubs plus the real implementation.

    Taking the first match would pin a one-line stub that never changes, so the
    anchor would stay green forever while the implementation drifted. That is
    the exact false negative this whole mechanism exists to prevent.
    """
    nodes, _ = resolve(Target("cash/core.py", "Cash.cache"))
    assert len(nodes) == 3
    spans = [n.end_lineno - n.lineno + 1 for n in nodes]
    assert max(spans) > 100, "the real implementation must be among them"


def test_resolves_a_module_level_function():
    nodes, _ = resolve(Target("cash/notebook/cache_key.py", "compute_cache_key"))
    assert [n.name for n in nodes] == ["compute_cache_key"]


def test_resolves_a_dataclass_field_to_its_annassign():
    nodes, _ = resolve(Target("cash/config.py", "CashConfig.compress"))
    assert isinstance(nodes[-1], ast.AnnAssign)


def test_no_symbol_resolves_to_the_module():
    nodes, _ = resolve(Target("cash/config.py", None))
    assert isinstance(nodes[-1], ast.Module)


def test_missing_file_is_an_anchor_error_naming_the_path():
    with pytest.raises(AnchorError, match="no such source file"):
        resolve(Target("cash/does_not_exist.py", "X"))


def test_missing_symbol_is_an_anchor_error_naming_the_symbol():
    with pytest.raises(AnchorError, match="nope"):
        resolve(Target("cash/core.py", "Cash.nope"))


def test_fingerprint_is_eight_hex_chars():
    nodes, src = resolve(Target("cash/core.py", "Cash.cache"))
    fp = fingerprint(nodes, src)
    assert len(fp) == 8 and all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_covers_every_overload_not_just_one():
    """Changing any one definition must move the digest."""
    nodes, src = resolve(Target("cash/core.py", "Cash.cache"))
    whole = fingerprint(nodes, src)
    assert whole != fingerprint(nodes[:1], src)
    assert whole != fingerprint(nodes[-1:], src)


def test_normalize_strips_comments_blank_lines_and_trailing_space():
    src = 'def f():\n    # a comment\n    x = 1  # trailing\n\n    return x   \n'
    node = ast.parse(src).body[0]
    assert normalize(node, src) == "def f():\n    x = 1\n    return x"


def _fp(src: str) -> str:
    return fingerprint([ast.parse(src).body[0]], src)


def test_comment_only_edits_do_not_change_the_fingerprint():
    """The whole point: churn must not fire, real change must."""
    assert _fp('def f():\n    return 1\n') == _fp(
        'def f():\n    # explain the 1\n    return 1   \n'
    )


def test_a_real_code_edit_does_change_the_fingerprint():
    assert _fp('def f():\n    return 1\n') != _fp('def f():\n    return 2\n')


def test_decorator_changes_are_inside_the_fingerprint():
    """A decorator decides behaviour; excluding it would hide a real change."""
    assert _fp('@property\ndef f(self):\n    return 1\n') != _fp(
        '@cached_property\ndef f(self):\n    return 1\n'
    )


def test_hash_of_a_known_string_is_pinned():
    """Cross-version determinism guard.

    CI runs 3.12, development runs 3.14, and ast.dump digests differ between
    them. If a Python upgrade ever moves this constant, it fails here — one
    obvious test — rather than as a wall of false drift across every page.
    """
    assert _fp("def f():\n    return 1\n") == "8795b1c4"
