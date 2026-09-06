"""Unit tests for the claim-anchor library."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs._claims import AnchorError, Target, parse_anchors, published_pages

FIXTURES = Path(__file__).parent / "_fixtures" / "claims"
# clean.md's fingerprint anchors are pinned against this synthetic tree, not
# the real src/ -- see clean.md's own leading comment for why.
CLEAN_SRC_ROOT = FIXTURES / "src"


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
    assert a.targets[0].pin == "37576d6b"
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


def test_anchor_inside_a_fence_is_ignored_but_one_outside_is_found():
    """An anchor shown as an illustrative example in a ``` fence (exactly what
    the README's own "Claim anchors" section does) must not be parsed as a
    live one -- it names no real target and would sit as permanent,
    uncleavable drift the moment such a section reaches a published page.
    """
    anchors = _anchors("fenced.md")
    assert len(anchors) == 1
    assert anchors[0].targets == (
        Target("cash/config.py", "CashConfig.compress", None, None),
    )


def test_published_pages_excludes_superpowers_and_unbuilt_adr():
    rels = {p.relative_to(Path(__file__).resolve().parents[2] / "docs").as_posix()
            for p in published_pages()}
    assert "architecture_decisions.md" not in rels
    assert not any(r.startswith("superpowers/") for r in rels)
    assert "index.md" in rels
    assert len(rels) == 59      # +1: warnings.md


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


# --------------------------------------------------------------------------- #
# Value anchors                                                               #
# --------------------------------------------------------------------------- #
from tests.docs._claims import literal_value, values_match  # noqa: E402


def test_literal_value_reads_a_plain_assignment():
    src = "X = 42\n"
    assert literal_value(ast.parse(src).body[0]) == 42


def test_literal_value_reads_a_dataclass_field_default():
    src = "from dataclasses import field\nx: int = field(default=7)\n"
    assert literal_value(ast.parse(src).body[1]) == 7


def test_default_factory_is_rejected_not_silently_passed():
    """It has no comparable literal; passing it would be a false green."""
    src = "from dataclasses import field\nx: list = field(default_factory=list)\n"
    with pytest.raises(AnchorError, match="default_factory"):
        literal_value(ast.parse(src).body[1])


def test_bare_annotation_with_no_value_is_rejected():
    """``x: int`` with no ``= ...`` has ``node.value is None`` -- nothing to
    compare a documented literal against.
    """
    src = "x: int\n"
    with pytest.raises(AnchorError, match="no assigned value"):
        literal_value(ast.parse(src).body[0])


def test_field_with_neither_default_is_rejected():
    """``field()`` with no ``default=`` and no ``default_factory=`` at all --
    distinct from the default_factory-present case above, and from the
    field-with-a-real-default case: this hits the for/else with nothing to
    break out of at all.
    """
    src = "from dataclasses import field\nx: int = field()\n"
    with pytest.raises(AnchorError, match="no default= to compare"):
        literal_value(ast.parse(src).body[1])


def test_a_non_literal_default_is_rejected():
    src = "X = compute()\n"
    with pytest.raises(AnchorError, match="not a literal"):
        literal_value(ast.parse(src).body[0])


def test_value_anchor_on_a_function_is_rejected():
    src = "def f():\n    return 1\n"
    with pytest.raises(AnchorError, match="assignment"):
        literal_value(ast.parse(src).body[0])


@pytest.mark.parametrize(
    "documented, actual, expected",
    [
        ("0.01", 0.01, True),
        ("0.01", 0.02, False),
        ("None", None, True),
        ("None", 0, False),
        ("True", True, True),
        ("False", True, False),
        ('"sha256"', "sha256", True),
        ('"md5"', "sha256", False),
        ("10", 10.0, True),      # int/float cross-compare is intentional
    ],
)
def test_values_match(documented, actual, expected):
    assert values_match(documented, actual) is expected


def test_a_non_literal_documented_value_is_an_error_not_a_mismatch():
    """'roughly 10' must be fixed by the author, not silently reported wrong."""
    with pytest.raises(AnchorError, match="not a Python literal"):
        values_match("roughly 10", 10)


# --------------------------------------------------------------------------- #
# Page checking                                                               #
# --------------------------------------------------------------------------- #
from tests.docs._claims import anchor_count, check_page  # noqa: E402


def test_a_clean_page_reports_nothing():
    """Positive control: without this the checker could pass by rejecting all."""
    assert check_page(FIXTURES / "clean.md", src_root=CLEAN_SRC_ROOT) == []


@pytest.mark.parametrize(
    "fixture, kind, fragment",
    [
        ("bad_path.md", "unresolved", "no such source file"),
        ("bad_symbol.md", "unresolved", "no_such_method"),
        ("stale_pin.md", "drift", "re-read the claim"),
        ("bare_class.md", "broad", "narrow it"),
        ("unpinned.md", "unpinned", "--pin"),
    ],
)
def test_each_defect_is_caught_exactly_once(fixture, kind, fragment):
    problems = check_page(FIXTURES / fixture)
    assert len(problems) == 1, problems
    assert problems[0].kind == kind
    assert fragment in problems[0].message


def test_a_drift_message_quotes_the_claim_it_grounds():
    """The reviewer needs the claim, not just a hash, to re-verify it."""
    (problem,) = check_page(FIXTURES / "stale_pin.md")
    assert "A claim whose pin no longer matches" in problem.message


def test_anchor_count_counts_targets_not_comments():
    assert anchor_count(FIXTURES / "clean.md") == 5


def test_broad_justification_does_not_suppress_drift_detection():
    """The only branch proven by inspection rather than by a test until now.

    A `broad=`-justified anchor must fall through to the fingerprint check
    rather than skip the target entirely — this fixture pairs `broad="..."`
    with a deliberately stale pin on a real class, and the sole problem
    reported must be the drift, not a (suppressed) breadth complaint.
    """
    problems = check_page(FIXTURES / "broad_pinned.md")
    assert len(problems) == 1, problems
    assert problems[0].kind == "drift"


# --------------------------------------------------------------------------- #
# Manifest / coverage ratchet                                                 #
# --------------------------------------------------------------------------- #
import json  # noqa: E402

from tests.docs import _claims  # noqa: E402
from tests.docs._claims import Problem, check_manifest  # noqa: E402


def _write_manifest(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _synthetic_pages(tmp_path, monkeypatch):
    """A tiny, fully-controlled published-page set, standing in for the real
    56-page docs/ tree.

    These tests used to derive ``some_page`` from ``_claims.published_pages()``
    (the REAL docs tree) and assume it carries 0 anchors -- true today only
    because every real page is still unaudited. The coming 56-page audit will
    break that assumption for whichever page sorts first, and it would fail
    for a reason that looks like something else entirely. Building a
    synthetic ``docs/`` tree (the same ``src_root=``-style technique the CLI
    tests use, applied to REPO_ROOT/published_pages instead) removes that
    coupling: these tests now exercise check_manifest's logic against content
    they fully control.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    page_a = docs / "page_a.md"
    page_a.write_text("# Page A\n\nNo anchors here.\n", encoding="utf-8")
    page_b = docs / "page_b.md"
    page_b.write_text(
        "# Page B\n\n<!-- claim: mod.py:Foo -->\nAn existence-only claim.\n",
        encoding="utf-8",
    )
    pages = [page_a, page_b]
    monkeypatch.setattr(_claims, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_claims, "published_pages", lambda: pages)
    return page_a, page_b


def test_manifest_missing_a_published_page_is_a_problem(tmp_path, monkeypatch):
    page_a, page_b = _synthetic_pages(tmp_path, monkeypatch)
    manifest_path = tmp_path / "claim_manifest.json"
    # page_a is published but absent from the manifest entirely.
    _write_manifest(manifest_path, {"docs/page_b.md": {"audited": None, "anchors": 0}})
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert problems == [
        Problem(
            "docs/page_a.md", 0, "manifest",
            'not in claim_manifest.json; add {"audited": null, "anchors": 0} '
            "and triage the page",
        )
    ]


def test_manifest_entry_for_a_nonexistent_page_is_a_problem(tmp_path, monkeypatch):
    _synthetic_pages(tmp_path, monkeypatch)
    manifest_path = tmp_path / "claim_manifest.json"
    data = {
        "docs/page_a.md": {"audited": None, "anchors": 0},
        "docs/page_b.md": {"audited": None, "anchors": 0},
        "docs/does-not-exist.md": {"audited": None, "anchors": 0},
    }
    _write_manifest(manifest_path, data)
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert problems == [
        Problem(
            "docs/does-not-exist.md", 0, "manifest",
            "listed in claim_manifest.json but is not a published page; "
            "remove the entry or restore the page",
        )
    ]


def test_manifest_anchor_count_regression_is_a_problem(tmp_path, monkeypatch):
    page_a, page_b = _synthetic_pages(tmp_path, monkeypatch)
    manifest_path = tmp_path / "claim_manifest.json"
    data = {
        "docs/page_a.md": {"audited": None, "anchors": 0},
        # Audited, and recorded as having more anchors than page_b really has
        # (1, from the single existence-only claim in _synthetic_pages).
        "docs/page_b.md": {"audited": "2026-07-27", "anchors": 2},
    }
    _write_manifest(manifest_path, data)
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert problems == [
        Problem(
            "docs/page_b.md", 0, "manifest",
            "anchor count fell from 2 to 1; a claim was removed along with "
            "its anchor",
        )
    ]


# --------------------------------------------------------------------------- #
# CLI (scripts/claims.py) -- the two correctness risks named in the task 6    #
# brief: does a prefix symbol (Cash.cache vs Cash.cache_info) or a duplicated #
# target cross-wire the regex-based rewrites in --pin / --accept?            #
# --------------------------------------------------------------------------- #
import functools  # noqa: E402

import scripts.claims as _cli  # noqa: E402


def _patch_src_root(monkeypatch, src_root):
    """Route the CLI's resolve()/check_page() calls at a throwaway source tree.

    ``resolve``/``check_page`` default their ``src_root`` parameter to the
    real repo's ``src/`` at *definition* time, so monkeypatching the module
    constant after the fact would not reach an already-bound default. Instead
    we replace the names ``scripts.claims`` looks up at call time with
    partials pinned to the fixture tree.
    """
    monkeypatch.setattr(_cli, "resolve", functools.partial(resolve, src_root=src_root))
    monkeypatch.setattr(_cli, "check_page", functools.partial(check_page, src_root=src_root))


def test_pin_does_not_cross_wire_a_prefix_symbol_pair(tmp_path, monkeypatch):
    """Cash.cache vs Cash.cache_info: a shorter symbol must not eat a longer one's @?.

    Both placeholders sit on the same page, in the order the longer name
    comes FIRST -- the case where a naive leftmost regex would be most likely
    to misfire if the needle for ``cache`` could match inside the text for
    ``cache_info``.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text(
        "def cache():\n    return 1\n\n\ndef cache_info():\n    return 2\n",
        encoding="utf-8",
    )
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py:cache_info @? -->\n"
        "Claim about cache_info.\n\n"
        "<!-- claim: mod.py:cache @? -->\n"
        "Claim about cache.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "published_pages", lambda: [page])
    # _cmd_pin's progress print does page.relative_to(REPO_ROOT); the fixture
    # page lives under tmp_path, not the real repo, so REPO_ROOT must follow.
    monkeypatch.setattr(_cli, "REPO_ROOT", tmp_path)

    assert _cli._cmd_pin() == 0

    cache_nodes, cache_src = resolve(Target("mod.py", "cache"), src_root=src_root)
    want_cache = fingerprint(cache_nodes, cache_src)
    info_nodes, info_src = resolve(Target("mod.py", "cache_info"), src_root=src_root)
    want_cache_info = fingerprint(info_nodes, info_src)
    assert want_cache != want_cache_info, "fixture must exercise genuinely different digests"

    new_text = page.read_text(encoding="utf-8")
    assert "@?" not in new_text
    assert f"mod.py:cache_info @{want_cache_info}" in new_text
    assert f"mod.py:cache @{want_cache}" in new_text
    # The precise cross-wire this test guards against: cache's line must not
    # have picked up cache_info's digest, or vice versa.
    assert f"mod.py:cache @{want_cache_info}" not in new_text
    assert f"mod.py:cache_info @{want_cache}" not in new_text


def test_accept_rewrites_both_occurrences_of_a_duplicated_stale_pin(tmp_path, monkeypatch):
    """Two anchors on the same target with the same stale pin must both update.

    A naive ``str.replace(old, new, count=1)`` would only touch the first
    occurrence, silently leaving the second one stale while still reporting
    success.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py:foo @00000000 -->\n"
        "First claim about foo.\n\n"
        "<!-- claim: mod.py:foo @00000000 -->\n"
        "Second claim about foo.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)

    assert _cli._cmd_accept(str(page), write=True) == 0

    new_text = page.read_text(encoding="utf-8")
    assert new_text.count("@00000000") == 0
    nodes, source = resolve(Target("mod.py", "foo"), src_root=src_root)
    want = fingerprint(nodes, source)
    assert new_text.count(f"mod.py:foo @{want}") == 2


def test_pin_matches_an_anchor_with_whitespace_around_the_colon(tmp_path, monkeypatch):
    """The parser accepts ``path : symbol`` (see test_whitespace_around_...
    above); the writer must recognize the exact same on-disk text or --pin
    reports success while writing nothing (the CRITICAL 1 regression).
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py : foo @? -->\nA claim about foo.\n", encoding="utf-8"
    )

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "published_pages", lambda: [page])
    monkeypatch.setattr(_cli, "REPO_ROOT", tmp_path)

    assert _cli._cmd_pin() == 0

    new_text = page.read_text(encoding="utf-8")
    assert "@?" not in new_text, "the placeholder must actually be replaced"
    nodes, source = resolve(Target("mod.py", "foo"), src_root=src_root)
    want = fingerprint(nodes, source)
    assert f"mod.py : foo @{want}" in new_text


def test_accept_matches_an_anchor_with_whitespace_around_the_colon(tmp_path, monkeypatch):
    """Same regression as above, on the --accept --yes rewrite path."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py : foo @00000000 -->\nA claim about foo.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)

    assert _cli._cmd_accept(str(page), write=True) == 0

    new_text = page.read_text(encoding="utf-8")
    assert "@00000000" not in new_text, "the stale pin must actually be replaced"
    nodes, source = resolve(Target("mod.py", "foo"), src_root=src_root)
    want = fingerprint(nodes, source)
    assert f"mod.py : foo @{want}" in new_text


def test_pin_raises_rather_than_silently_reporting_success_on_a_needle_mismatch(
    tmp_path, monkeypatch
):
    """The re.subn zero-match guard: if the needle EVER fails to match the
    on-disk text (whatever the reason), --pin must error rather than print
    "N placeholder(s) filled" while writing nothing. This is the safety net
    for a third instance of the CRITICAL 1 bug class, not a specific needle
    rule.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    page = tmp_path / "page.md"
    original = "<!-- claim: mod.py:foo @? -->\nA claim.\n"
    page.write_text(original, encoding="utf-8")

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "published_pages", lambda: [page])
    monkeypatch.setattr(_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_cli, "_needle", lambda t: "this-never-appears-in-the-page")

    with pytest.raises(AnchorError, match="could not find"):
        _cli._cmd_pin()
    assert page.read_text(encoding="utf-8") == original


def test_accept_raises_rather_than_silently_reporting_success_on_a_needle_mismatch(
    tmp_path, monkeypatch
):
    """Same guard, on the --accept --yes rewrite path."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    page = tmp_path / "page.md"
    original = "<!-- claim: mod.py:foo @00000000 -->\nA claim.\n"
    page.write_text(original, encoding="utf-8")

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "_needle", lambda t: "this-never-appears-in-the-page")

    with pytest.raises(AnchorError, match="could not find"):
        _cli._cmd_accept(str(page), write=True)
    assert page.read_text(encoding="utf-8") == original


def test_accept_does_not_repin_a_broad_unjustified_anchor(tmp_path, monkeypatch):
    """A class-level anchor with no ``broad="reason"`` is a "broad" problem,
    not "drift" (see check_page). --accept must not quietly re-pin it just
    because another, genuinely drifted, anchor shares the page -- that would
    paper over a real breadth problem as if someone had re-verified it.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text(
        "class Foo:\n    def bar(self):\n        return 1\n", encoding="utf-8"
    )
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py:Foo @00000000 -->\n"
        "A bare class-level claim (no broad= justification).\n\n"
        "<!-- claim: mod.py:Foo.bar @00000000 -->\n"
        "A real, narrow claim that has genuinely drifted.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)

    assert _cli._cmd_accept(str(page), write=True) == 0

    new_text = page.read_text(encoding="utf-8")
    bar_nodes, bar_src = resolve(Target("mod.py", "Foo.bar"), src_root=src_root)
    want_bar = fingerprint(bar_nodes, bar_src)
    # The narrow, genuinely-drifted anchor is re-pinned.
    assert f"mod.py:Foo.bar @{want_bar}" in new_text
    # The bare class anchor is a breadth problem, not drift -- left untouched.
    assert "mod.py:Foo @00000000" in new_text


def test_accept_prints_the_count_actually_rewritten_not_len_drifted(
    tmp_path, monkeypatch, capsys
):
    """MINOR 8: the printed count must reflect substitutions actually made.

    Without the breadth gate, this page's ``drifted`` (from check_page) has
    length 1 (the narrow Foo.bar anchor only -- Foo's mismatch is "broad",
    not "drift"), but a version of --accept that re-pinned both anchors while
    still printing ``len(drifted)`` would say "re-pinned 1 claim(s)" despite
    having rewritten two. Assert the message names exactly what was rewritten.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text(
        "class Foo:\n    def bar(self):\n        return 1\n", encoding="utf-8"
    )
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- claim: mod.py:Foo @00000000 -->\n"
        "A bare class-level claim (no broad= justification).\n\n"
        "<!-- claim: mod.py:Foo.bar @00000000 -->\n"
        "A real, narrow claim that has genuinely drifted.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)
    capsys.readouterr()

    assert _cli._cmd_accept(str(page), write=True) == 0

    out = capsys.readouterr().out
    assert "re-pinned 1 claim(s)" in out


def test_accept_rewrites_a_module_level_broad_anchor(tmp_path, monkeypatch):
    """A symbol-less (module) target's real text has no ``:<module>`` suffix.

    ``--accept``'s display string uses ``<module>`` for readability, but the
    text it rewrites in the page must match what is actually on disk -- using
    the display string as the rewrite needle would silently no-op here while
    still printing "re-pinned 1 claim(s)".
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("X = 1\nY = 2\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        '<!-- claim: mod.py @00000000 broad="whole module" -->\n'
        "A claim about the whole module.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)

    assert _cli._cmd_accept(str(page), write=True) == 0

    new_text = page.read_text(encoding="utf-8")
    assert "@00000000" not in new_text
    assert "<module>" not in new_text  # never leaks into the actual file
    nodes, source = resolve(Target("mod.py", None), src_root=src_root)
    want = fingerprint(nodes, source)
    assert f"mod.py @{want}" in new_text


def test_pin_does_not_pin_a_fenced_example_anchor(tmp_path, monkeypatch):
    """A fenced example anchor for the same target, appearing BEFORE the live
    anchor in raw document order, must not be the occurrence --pin rewrites.

    ``parse_anchors`` masks code fences so the example is never treated as
    live (it names no real claim), but ``_cmd_pin`` rewrites the *raw* text
    with ``re.subn(..., count=1)``. A naive leftmost-match rewrite would hit
    the fenced example first and leave the real, live anchor un-pinned --
    exactly the false-assurance failure this mechanism exists to prevent.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "```markdown\n"
        "<!-- claim: mod.py:foo @? -->\n"
        "An example claim shown for illustration.\n"
        "```\n"
        "\n"
        "<!-- claim: mod.py:foo @? -->\n"
        "The real, live claim about foo.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "published_pages", lambda: [page])
    monkeypatch.setattr(_cli, "REPO_ROOT", tmp_path)

    assert _cli._cmd_pin() == 0

    nodes, source = resolve(Target("mod.py", "foo"), src_root=src_root)
    want = fingerprint(nodes, source)

    lines = page.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "<!-- claim: mod.py:foo @? -->", (
        "the fenced EXAMPLE anchor must stay byte-identical -- it is not a "
        "live claim and must never be pinned"
    )
    assert lines[5] == f"<!-- claim: mod.py:foo @{want} -->", (
        "the live anchor after the fence must be the one that gets pinned"
    )


def test_accept_does_not_repin_a_fenced_example_anchor(tmp_path, monkeypatch):
    """Same defect as above, on the ``--accept --yes`` rewrite path.

    A fenced example anchor sharing the live anchor's stale pin, and
    appearing before it in raw text, must not be the occurrence rewritten.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "```markdown\n"
        "<!-- claim: mod.py:foo @00000000 -->\n"
        "An example claim shown for illustration.\n"
        "```\n"
        "\n"
        "<!-- claim: mod.py:foo @00000000 -->\n"
        "The real, live claim about foo.\n",
        encoding="utf-8",
    )

    _patch_src_root(monkeypatch, src_root)

    assert _cli._cmd_accept(str(page), write=True) == 0

    nodes, source = resolve(Target("mod.py", "foo"), src_root=src_root)
    want = fingerprint(nodes, source)

    lines = page.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "<!-- claim: mod.py:foo @00000000 -->", (
        "the fenced EXAMPLE anchor must stay byte-identical -- it is not a "
        "live claim and must never be re-pinned"
    )
    assert lines[5] == f"<!-- claim: mod.py:foo @{want} -->", (
        "the live anchor after the fence must be the one that gets re-pinned"
    )


# --------------------------------------------------------------------------- #
# Display: full claim text stored, ellipsized only where shown                #
# --------------------------------------------------------------------------- #
from tests.docs._claims import ellipsize  # noqa: E402


def test_claim_text_is_stored_in_full_not_truncated(tmp_path):
    """Truncating at parse time silently shortens the value for every consumer.

    The old code cut at 120 chars inside ``_claim_text``, so a long claim was
    lossy before any caller saw it -- including a future one that compares
    claim text rather than printing it.
    """
    long_claim = "A " + "very " * 60 + "long claim sentence."
    assert len(long_claim) > 120
    page = tmp_path / "p.md"
    page.write_text(f"<!-- claim: cash/core.py:Cash.cache -->\n{long_claim}\n", encoding="utf-8")
    (anchor,) = parse_anchors(page.read_text(encoding="utf-8"), page)
    assert anchor.claim == long_claim


def test_ellipsize_breaks_on_a_word_boundary():
    text = "the quick brown fox jumps over the lazy dog"
    out = ellipsize(text, 20)
    assert out.endswith("...")
    assert not out[:-3].endswith(" ")
    # never splits a word in half
    assert all(w in text.split() for w in out[:-3].split())


def test_ellipsize_leaves_short_text_untouched():
    assert ellipsize("short", 100) == "short"


def test_ellipsize_handles_a_single_unbroken_token():
    """No space to break on -- must still bound the length, not loop or crash."""
    out = ellipsize("x" * 50, 10)
    assert out == "x" * 10 + "..."


# --------------------------------------------------------------------------- #
# Value anchors over constant arithmetic (8 * 1024 * 1024)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("X = 8 * 1024 * 1024", 8388608),
        ("X = 256 * 1024", 262144),
        ("X = 60 * 60", 3600),
        ("X = 1024 ** 2", 1048576),
        ("X = 100 // 3", 33),
        ("X = -5 * 2", -10),
        ("X = 1.5 * 2", 3.0),
    ],
)
def test_value_anchor_folds_constant_arithmetic(expr, expected):
    """`8 * 1024 * 1024` is how every byte threshold here is written, and byte
    thresholds are exactly the numbers docs quote."""
    assert literal_value(ast.parse(expr).body[0]) == expected


@pytest.mark.parametrize(
    "expr, why",
    [
        ("X = SOME_NAME * 1024", "name lookup"),
        ("X = compute() * 2", "call"),
        ("X = mod.ATTR * 2", "attribute access"),
        ("X = 'a' * 3", "non-numeric operand"),
        ("X = [1] * 3", "non-numeric operand"),
        ("X = 8 % 3", "operator not in the allowlist"),
    ],
)
def test_value_anchor_refuses_anything_that_is_not_pure_numeric(expr, why):
    """Folding must never execute or resolve anything -- it is a constant
    folder, not an evaluator. Anything it cannot prove constant must raise, so
    the anchor fails loudly instead of comparing against a guess."""
    with pytest.raises(AnchorError, match="not a literal"):
        literal_value(ast.parse(expr).body[0])


def test_folded_arithmetic_compares_against_a_documented_number():
    """End to end: the doc writes 8388608, the source writes 8 * 1024 * 1024."""
    value = literal_value(ast.parse("X = 8 * 1024 * 1024").body[0])
    assert values_match("8388608", value)
    assert not values_match("4194304", value)


def test_unpinned_class_anchor_needs_no_broad_justification(tmp_path):
    """An existence anchor on a class carries no fingerprint, so it cannot
    drift -- the breadth rule has nothing to protect against and must stay
    quiet. `redis_backend.py:RedisBackend` with no pin is the correct way to
    ground "Cash ships a Redis backend"."""
    page = tmp_path / "p.md"
    page.write_text(
        "<!-- claim: cash/backends/redis_backend.py:RedisBackend -->\nCash ships a Redis backend.\n",
        encoding="utf-8",
    )
    assert check_page(page) == []


def test_pinned_class_anchor_still_needs_broad_justification(tmp_path):
    """The rule must keep biting where the noise actually comes from."""
    page = tmp_path / "p.md"
    page.write_text(
        "<!-- claim: cash/backends/redis_backend.py:RedisBackend @00000000 -->\nA claim.\n",
        encoding="utf-8",
    )
    (problem,) = check_page(page)
    assert problem.kind == "broad"


def test_value_anchor_on_a_class_still_needs_broad_justification(tmp_path):
    """A value anchor is pinned too -- it must not slip through the new gate."""
    page = tmp_path / "p.md"
    page.write_text(
        "<!-- claim: cash/backends/redis_backend.py:RedisBackend == 1 -->\nA claim.\n",
        encoding="utf-8",
    )
    (problem,) = check_page(page)
    assert problem.kind == "broad"


# --------------------------------------------------------------------------- #
# Unanchored prose triage                                                     #
#                                                                             #
# Regression fixtures for two REAL misses. The JupyterLab live-cell branch    #
# falsified two documented statements and the drift queue saw neither, because #
# it only ever reads sentences that carry an anchor:                          #
#                                                                             #
#   docs/magics.md            -- the %%cash behaviour list, in a sibling      #
#                               section of the page that anchors %cash_on     #
#   quickstart.md             -- "Google Colab is the exception", in a section #
#                               with no anchor at all, naming no symbol        #
#                                                                             #
# The two fixtures below are those two shapes, reduced. Each needs a DIFFERENT #
# rule to reach it, which is why check_unanchored has two.                    #
# --------------------------------------------------------------------------- #
from tests.docs._claims import check_unanchored, mention_pattern  # noqa: E402


def _page(tmp_path, name, body):
    page = tmp_path / name
    page.write_text(body, encoding="utf-8")
    return page


MAGICS_SHAPE = """\
# Magics

## Enabling and configuring

### `%thing_on`
<!-- claim: mod.py:Magics.thing_on @00000000 -->

Enable the thing. Turn it off and on with `%thing_on` again.

### `%%thing`
<!-- claim: mod.py:Magics.thing @11111111 -->

Cell magic, with the same processing as `%thing_on`.

**Behaviour:**

- Runs upstream simulation against the on-disk notebook.
"""

QUICKSTART_SHAPE = """\
# Quickstart

## Walk-back

!!! warning "Save the notebook first"
    Cash reads the cells it did not execute from the saved file, not from
    your editor's buffer. **Google Colab is the exception**: there cash reads
    cells live from the frontend, so there is nothing to save.

## Configuration

<!-- claim: mod.py:Magics.thing_on @00000000 -->
`%thing_on` takes only an optional `ttl=N`.
"""

THING_ON = Target("mod.py", "Magics.thing_on")


def test_a_sibling_section_of_the_anchoring_page_is_surfaced(tmp_path):
    """The docs/magics.md miss: same page, one section over, unpinned.

    The false statement lived under `### %%cash` on the very page that anchors
    `%cash_on`. Excluding the whole anchoring page -- the obvious rule, and the
    one this was first specified with -- surfaces nothing here at all. The
    exclusion has to be the anchored SECTION.
    """
    page = _page(tmp_path, "magics.md", MAGICS_SHAPE)
    hits = check_unanchored(THING_ON, [page])

    assert [h.line for h in hits] == [13], hits
    assert hits[0].kind == "unanchored"
    assert "same processing as `%thing_on`" in hits[0].message
    # ...and NOT the anchor's own section: the `### %thing_on` heading on line 5
    # and the sentence under it on line 8 both name the target.
    assert all(h.line not in (5, 8) for h in hits)


def test_a_closed_enumeration_on_the_anchoring_page_is_surfaced(tmp_path):
    """The quickstart.md miss: names no symbol, so only co-location can reach it.

    "Google Colab is the exception" contains no identifier of any kind. No
    name-based rule can find it from the target's side; what links the two is
    that the page is already pinned to this code and this sentence about it is
    not.
    """
    page = _page(tmp_path, "quickstart.md", QUICKSTART_SHAPE)
    hits = check_unanchored(THING_ON, [page])

    lines = {h.line: h for h in hits}
    assert 7 in lines, hits
    assert "Google Colab is the exception" in lines[7].message
    assert "closed enumeration" in lines[7].message
    # The anchored section's own `%thing_on` sentence (line 13) is the working
    # case and must stay quiet.
    assert all(h.line < 10 for h in hits), hits


def test_a_mention_on_a_page_that_never_anchors_the_target_is_surfaced(tmp_path):
    """The plain case: another page talks about the code and pins nothing."""
    page = _page(
        tmp_path, "faq.md",
        "# FAQ\n\nNo. `%thing_on` caches your existing cells as-is.\n",
    )
    hits = check_unanchored(THING_ON, [page])
    assert [h.line for h in hits] == [3], hits


def test_a_closed_enumeration_needs_the_page_to_anchor_the_target(tmp_path):
    """Scoping control: without it this becomes a global prose lint.

    The same sentence, on a page that pins nothing, must stay silent -- a
    standing "find every absolute claim" check would fire across the whole
    corpus and get switched off within a week.
    """
    page = _page(
        tmp_path, "loose.md",
        "# Loose\n\n**Google Colab is the exception**: it reads cells live.\n",
    )
    assert check_unanchored(THING_ON, [page]) == []


def test_absolute_emphasis_is_not_a_closed_enumeration(tmp_path):
    """"never" / "only" / "always" alone are ordinary emphasis, not a list.

    Measured on the pre-fix tree, admitting them added 23 lines to one target's
    triage and every one was still true. The failure class is prose that closes
    an enumeration a new code path then joins.
    """
    page = _page(
        tmp_path, "emphasis.md",
        "# Emphasis\n\n"
        "The repair never fires, and only the cheap append happens; it always\n"
        "reproduces the value.\n\n"
        "## Config\n\n"
        "<!-- claim: mod.py:Magics.thing_on @00000000 -->\n"
        "Takes a ttl.\n",
    )
    assert check_unanchored(THING_ON, [page]) == []


def test_a_mention_inside_a_code_fence_is_not_surfaced(tmp_path):
    """An example is not a claim -- the same rule parse_anchors already applies."""
    page = _page(
        tmp_path, "example.md",
        "# Example\n\n```python\n%thing_on\n%thing_on ttl=60\n```\n",
    )
    assert check_unanchored(THING_ON, [page]) == []


def test_html_comments_are_not_prose(tmp_path):
    """A comment naming the target is not a page making a claim about it.

    Two failures in one fixture, because one fixture can carry both:

    * the editorial comment on line 4 names the symbol. Unmasked, it is
      reported as unpinned prose -- and so is every claim anchor for a
      same-named symbol in another module, which on the real corpus was four
      of ``UpstreamChecker``'s six hits.
    * it also spans lines. Masking a comment by ``"\\0" * len(match)`` eats the
      newlines inside it and every line after comes out short -- on
      ``docs/magics.md``, by five. The real mention sits AFTER it, so a
      swallowed newline moves the number this reports.
    """
    page = _page(
        tmp_path, "anchored.md",
        "# Page\n\n"
        "<!--\nEditorial note: document `%thing_on`'s ttl handling\nhere.\n-->\n\n"
        "## Reference\n\n"
        "<!-- claim: mod.py:Magics.thing_on @00000000 -->\n"
        "Enable the thing.\n\n"
        "## Elsewhere\n\n"
        "Run `%thing_on` first.\n",
    )
    hits = check_unanchored(THING_ON, [page])
    assert [h.line for h in hits] == [15], hits
    assert page.read_text(encoding="utf-8").splitlines()[14].strip() == (
        "Run `%thing_on` first."
    )


def test_a_generic_final_component_is_not_searched_bare(tmp_path):
    """`Cash.cache` must not go hunting for the word "cache".

    It occurs on 633 prose lines across 51 published pages; attaching that to a
    drift entry buries the entry it is attached to. The dotted form still works.
    """
    assert mention_pattern(Target("cash/core.py", "Cash.cache")).search("Cash.cache")
    page = _page(
        tmp_path, "prose.md",
        "# Prose\n\nThe cache is unbounded, and a cache entry is compressed.\n",
    )
    assert check_unanchored(Target("cash/core.py", "Cash.cache"), [page]) == []


def test_a_module_target_has_no_prose_name(tmp_path):
    assert mention_pattern(Target("cash/core.py", None)) is None


def test_triage_never_becomes_a_gate(tmp_path):
    """check_page is what blocks; it must not learn about any of this.

    A page that check_unanchored has plenty to say about still has to come back
    clean from the checker the test suite actually asserts on -- otherwise this
    is a new gate, it floods, and it gets disabled.
    """
    src_root = tmp_path / "src"
    (src_root / "").mkdir(parents=True, exist_ok=True)
    (src_root / "mod.py").write_text(
        "class Magics:\n    def thing_on(self):\n        return 1\n",
        encoding="utf-8",
    )
    body = QUICKSTART_SHAPE.replace("@00000000", "@?")
    page = _page(tmp_path, "quickstart.md", body)
    nodes, source = resolve(Target("mod.py", "Magics.thing_on"), src_root=src_root)
    page.write_text(
        body.replace("@?", "@" + fingerprint(nodes, source)), encoding="utf-8"
    )

    assert check_unanchored(THING_ON, [page]), "fixture must have triage to report"
    assert check_page(page, src_root=src_root) == []


def test_queue_lists_unpinned_prose_under_the_drifted_target(tmp_path, monkeypatch,
                                                             capsys):
    """The whole point, end to end: drift, then the unpinned prose beside it."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "mod.py").write_text(
        "class Magics:\n    def thing_on(self):\n        return 1\n",
        encoding="utf-8",
    )
    anchored = _page(
        tmp_path, "anchored.md",
        "# Anchored\n\n"
        "<!-- claim: mod.py:Magics.thing_on @00000000 -->\n"
        "Enable the thing.\n",
    )
    loose = _page(
        tmp_path, "loose.md",
        "# Loose\n\nRun `%thing_on` before anything else.\n",
    )

    _patch_src_root(monkeypatch, src_root)
    monkeypatch.setattr(_cli, "published_pages", lambda: [anchored, loose])
    monkeypatch.setattr(_cli, "REPO_ROOT", tmp_path)

    assert _cli._cmd_queue() == 1
    out = capsys.readouterr().out
    assert "Unpinned prose about the same code" in out
    assert "mod.py:Magics.thing_on" in out
    assert ":3  names mod.py:Magics.thing_on but pins nothing" in out
    assert "Run `%thing_on` before anything else." in out
    # The drift entry itself is still the headline, and still the exit status.
    assert "1 claim(s) rest on code that has changed" in out
