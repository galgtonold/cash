#!/usr/bin/env python
"""Inspect and maintain doc claim anchors.

    python scripts/claims.py --queue                     # what needs re-reading
    python scripts/claims.py --pin                       # fill in every `@?`
    python scripts/claims.py --accept docs/page.md       # dry run: show the code
    python scripts/claims.py --accept docs/page.md --yes # re-pin after reading
    python scripts/claims.py --report cash/core.py       # claims resting on a file

`--accept` prints the claim beside the target's CURRENT source, because that is
what re-verification needs: you check the claim against the code as it now is.
The old source is not recoverable from an 8-char digest, and reconstructing it
from git history would be fragile -- so this deliberately does not attempt a
before/after diff.

`--accept` without `--yes` is a dry run by design. Re-pinning without reading
is worse than having no mechanism at all: it manufactures the appearance that
someone checked the claim, when nobody did. `--yes` exists so re-pinning is
always a second, conscious step after reading the printed source.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.docs._claims import (  # noqa: E402
    AnchorError,
    Target,
    check_page,
    fingerprint,
    normalize,
    parse_anchors,
    published_pages,
    resolve,
)


def _needle(t: Target) -> str:
    """A regex fragment matching a target's path/symbol text in an anchor comment.

    This must accept exactly the whitespace ``_TARGET_RE`` accepts around the
    ``:`` separator (``\\s*:\\s*``) -- the parser blesses ``path : symbol`` as
    valid (see ``test_whitespace_around_punctuation_is_insignificant``), so a
    needle built as a plain literal string with a bare ``:`` would silently
    fail to match that on-disk text. "Silently" is the operative danger: see
    the ``re.subn`` + zero-match guard at both call sites, which exists
    precisely because a needle/parser mismatch must never be allowed to reach
    a success message.

    A module-only target (``t.symbol`` is ``None``) is written in the doc as
    just ``path`` -- there is no ``:<module>`` suffix in the real text, even
    though that suffix is useful in *display* strings (see ``_display``).
    Building the search/replace needle from the same rule ``--pin`` uses for
    the ``@?`` case keeps both mutating code paths honest about what they are
    actually matching against.
    """
    pattern = re.escape(t.path)
    if t.symbol:
        pattern += r"\s*:\s*" + re.escape(t.symbol)
    return pattern


def _display(t: Target) -> str:
    """Human-readable target name for printed output (never used as a needle)."""
    return f"{t.path}:{t.symbol or '<module>'}"


def _cmd_queue() -> int:
    drifted = []
    for page in published_pages():
        drifted += [p for p in check_page(page) if p.kind == "drift"]
    if not drifted:
        print("No drifted claims. Docs are pinned to current source.")
        return 0
    print(f"{len(drifted)} claim(s) rest on code that has changed:\n")
    for p in drifted:
        print(f"{p.page}:{p.line}")
        print(f"    {p.message}")
        print()
    print("Re-read each, then: python scripts/claims.py --accept <page> --yes")
    return 1


def _cmd_pin() -> int:
    """Fill in every ``@?`` placeholder with the target's current digest.

    This is the ergonomic hinge of the whole mechanism: an author writes
    ``@?`` and never hand-computes or hand-copies a hash. The substitution is
    a regex (not a plain string replace) so that whatever whitespace the
    author put around ``@?`` survives untouched -- only the ``?`` itself is
    replaced.

    Each page is parsed once, up front, from the text on disk. The loop below
    then edits a working copy (``new``) anchor by anchor, in the same source
    order ``parse_anchors`` found them, and -- critically -- confines every
    substitution to that ONE anchor's own ``span`` (the exact character range
    of its ``<!-- claim: ... -->`` comment, computed by the parser against the
    fence-masked text and equally valid as an index into the raw text; see
    ``Anchor.span``). Searching the needle anywhere in the page, rather than
    within its own anchor's text, is exactly how a fenced *example* anchor
    (which parse_anchors correctly never treats as live) could still end up
    being the occurrence a leftmost ``re.subn`` rewrites, if it happens to sit
    before the real, live anchor for the same target. Confining the
    substitution to the span the parser actually decided was live eliminates
    that by construction -- it cannot pick the wrong occurrence because it is
    never even shown one. The same confinement is also why a target whose
    symbol is a prefix of another (``Cash.cache`` vs ``Cash.cache_info``) or a
    duplicated target on the page cannot cross-wire: each rewrite only ever
    sees the text of the one anchor it belongs to.

    Because a rewrite can change an anchor's length (``@?`` is one character,
    a digest is nine with the ``@``), ``delta`` tracks the cumulative shift so
    each subsequent anchor's span -- computed against the original, unedited
    ``text`` -- is translated into the correct offset into ``new``.
    """
    filled = 0
    for page in published_pages():
        text = page.read_text(encoding="utf-8")
        if "@?" not in text:
            continue
        new = text
        delta = 0
        for anchor in parse_anchors(text, page):
            start, end = anchor.span[0] + delta, anchor.span[1] + delta
            original_segment = new[start:end]
            segment = original_segment
            for t in anchor.targets:
                if t.pin != "?":
                    continue
                nodes, source = resolve(t)
                fp = fingerprint(nodes, source)
                needle = _needle(t)
                segment, n = re.subn(
                    rf"({needle}\s*)@\s*\?",
                    rf"\g<1>@{fp}",
                    segment,
                    count=1,
                )
                if n == 0:
                    # A needle that does not match on-disk text must never be
                    # reported as filled -- that is exactly the false-assurance
                    # failure this mechanism exists to prevent (see _needle).
                    raise AnchorError(
                        f"{page}: could not find {_display(t)} @? to pin -- "
                        f"the anchor text does not match what the parser saw"
                    )
                filled += 1
            if segment != original_segment:
                new = new[:start] + segment + new[end:]
                delta += len(segment) - len(original_segment)
        if new != text:
            page.write_text(new, encoding="utf-8")
            print(f"pinned {page.relative_to(REPO_ROOT).as_posix()}")
    print(f"{filled} placeholder(s) filled")
    return 0


def _cmd_accept(page_arg: str, write: bool) -> int:
    """Re-pin one page's drifted claims -- print-only unless ``write`` is set.

    Printing the claim beside the CURRENT normalized source (not a diff
    against the old pin) is the point: an 8-char digest carries no old
    source to diff against, and reconstructing one from git history would be
    fragile and easy to get subtly wrong. What re-verification actually needs
    is "is this claim still true of the code as it stands" -- which the
    current source answers directly.

    The rewrite uses the same needle rule as ``--pin`` (see ``_needle``) and
    the same tolerant-whitespace regex, rather than a hardcoded
    ``f"{name} @{pin}"`` string: a plain string match would silently no-op
    (while still printing a success message) for a module-only target, whose
    real text has no ``:<module>`` suffix, or for an anchor written with
    different spacing than assumed. As with ``--pin``, every substitution is
    confined to the ONE anchor's own ``span`` -- the exact character range of
    its comment, computed by the parser -- rather than searched for anywhere
    on the page (see ``_cmd_pin``'s docstring for why: a fenced *example*
    anchor sharing the same stale pin, sitting earlier in the page than the
    real, live anchor, would otherwise be the occurrence a leftmost match
    rewrites instead). That confinement is also what lets two anchors sharing
    a target and stale pin, or a bare class/module anchor beside a narrow one
    on the same target, be rewritten independently without cross-wiring.
    """
    page = (REPO_ROOT / page_arg).resolve()
    if not page.is_file():
        print(f"no such page: {page_arg}", file=sys.stderr)
        return 2
    text = page.read_text(encoding="utf-8")
    drifted = [p for p in check_page(page) if p.kind == "drift"]
    if not drifted:
        print(f"{page_arg}: no drifted claims")
        return 0

    new = text
    delta = 0
    rewritten = 0
    for anchor in parse_anchors(text, page):
        start, end = anchor.span[0] + delta, anchor.span[1] + delta
        original_segment = new[start:end]
        segment = original_segment
        for t in anchor.targets:
            if not t.pin or t.pin == "?":
                continue
            nodes, source = resolve(t)
            node = nodes[-1]
            # Mirror check_page's breadth gate: a class/module anchor with no
            # broad="reason" is reported as a "broad" problem, not "drift" --
            # --accept must not silently re-pin it, or a real breadth problem
            # gets papered over as if someone had re-verified it.
            if isinstance(node, (ast.Module, ast.ClassDef)) and not anchor.broad:
                continue
            fp = fingerprint(nodes, source)
            if fp == t.pin:
                continue
            display = _display(t)
            print("=" * 72)
            print(f"{page_arg}:{anchor.line}  ->  {display}  @{t.pin} -> @{fp}")
            print(f"\nCLAIM:\n  {anchor.claim}\n")
            print("CURRENT SOURCE:")
            for node in nodes:
                for line in normalize(node, source).splitlines():
                    print(f"  {line}")
            print()
            needle = _needle(t)
            segment, n = re.subn(
                rf"({needle}\s*@\s*){re.escape(t.pin)}",
                rf"\g<1>{fp}",
                segment,
                count=1,
            )
            if n == 0:
                # Same guard as --pin: a rewrite that matched nothing must
                # never reach the "re-pinned N claim(s)" success message.
                raise AnchorError(
                    f"{page_arg}: could not find {display} @{t.pin} to re-pin -- "
                    f"the anchor text does not match what the parser saw"
                )
            rewritten += 1
        if segment != original_segment:
            new = new[:start] + segment + new[end:]
            delta += len(segment) - len(original_segment)

    if not write:
        print("=" * 72)
        print("Dry run. Re-read each claim above against the source shown, then")
        print(f"re-run with --yes to re-pin: --accept {page_arg} --yes")
        return 0
    page.write_text(new, encoding="utf-8")
    # The count of substitutions actually performed, not len(drifted): the
    # breadth gate above can (correctly) skip a target check_page counted as
    # "drift" -- most tellingly it excludes broad-unjustified anchors, which
    # check_page reports as a separate "broad" problem, not "drift", so they
    # were never in `drifted` to begin with; this count is honest either way.
    print(f"re-pinned {rewritten} claim(s) in {page_arg}")
    return 0


def _cmd_report(src_path: str) -> int:
    hits = 0
    for page in published_pages():
        text = page.read_text(encoding="utf-8")
        for anchor in parse_anchors(text, page):
            for t in anchor.targets:
                if t.path != src_path:
                    continue
                rel = page.relative_to(REPO_ROOT).as_posix()
                print(f"{rel}:{anchor.line}  {t.symbol or '<module>'}")
                print(f"    {anchor.claim}")
                hits += 1
    if not hits:
        print(f"No doc claims rest on src/{src_path}")
    else:
        print(f"\n{hits} claim(s) rest on src/{src_path} — check them before merging.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--queue", action="store_true", help="list drifted claims")
    g.add_argument("--pin", action="store_true", help="fill in every @? placeholder")
    g.add_argument("--accept", metavar="PAGE", help="re-pin one page's drifted claims")
    g.add_argument("--report", metavar="SRC", help="claims resting on a source file")
    ap.add_argument("--yes", action="store_true", help="with --accept: actually write")
    args = ap.parse_args(argv)

    try:
        if args.queue:
            return _cmd_queue()
        if args.pin:
            return _cmd_pin()
        if args.accept:
            return _cmd_accept(args.accept, write=args.yes)
        return _cmd_report(args.report)
    except AnchorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
