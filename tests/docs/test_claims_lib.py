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
from tests.docs._claims import check_manifest  # noqa: E402


def _write_manifest(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_manifest_missing_a_published_page_is_a_problem(tmp_path, monkeypatch):
    manifest_path = tmp_path / "claim_manifest.json"
    pages = {p.relative_to(_claims.REPO_ROOT).as_posix() for p in _claims.published_pages()}
    some_page = sorted(pages)[0]
    _write_manifest(manifest_path, {p: {"audited": None, "anchors": 0}
                                     for p in pages if p != some_page})
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert any(p.kind == "manifest" and p.page == some_page for p in problems)


def test_manifest_entry_for_a_nonexistent_page_is_a_problem(tmp_path, monkeypatch):
    manifest_path = tmp_path / "claim_manifest.json"
    pages = {p.relative_to(_claims.REPO_ROOT).as_posix() for p in _claims.published_pages()}
    data = {p: {"audited": None, "anchors": 0} for p in pages}
    data["docs/does-not-exist.md"] = {"audited": None, "anchors": 0}
    _write_manifest(manifest_path, data)
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert any(
        p.kind == "manifest" and p.page == "docs/does-not-exist.md" for p in problems
    )


def test_manifest_anchor_count_regression_is_a_problem(tmp_path, monkeypatch):
    manifest_path = tmp_path / "claim_manifest.json"
    pages = {p.relative_to(_claims.REPO_ROOT).as_posix() for p in _claims.published_pages()}
    some_page = sorted(pages)[0]
    data = {p: {"audited": None, "anchors": 0} for p in pages}
    # Audited, and recorded as having more anchors than the real page has (0).
    data[some_page] = {"audited": "2026-07-27", "anchors": 1}
    _write_manifest(manifest_path, data)
    monkeypatch.setattr(_claims, "MANIFEST", manifest_path)

    problems = check_manifest()
    assert any(
        p.kind == "manifest" and p.page == some_page and "fell from" in p.message
        for p in problems
    )


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
