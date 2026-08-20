"""Claim anchors: link a prose claim to the source that decides it.

An anchor is an HTML comment immediately before the claim it grounds::

    <!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
    Cash keys a call on the function source plus its arguments.

An unresolvable anchor, a mismatched ``== <literal>``, or a page that lost
anchors under the manifest ratchet blocks every PR. Fingerprint drift alone is
advisory on a PR and blocking only at release, via ``CASH_CLAIMS_STRICT=1``
(set by the ``build`` job in ``.github/workflows/publish.yml``) — see
``tests/docs/README.md``'s "Claim anchors" section for the authoring rules.

Anchored prose is the only prose any of that reads. ``check_unanchored`` covers
the other half — when a target drifts, it surfaces the *unpinned* pages that
talk about the same code, as triage attached to the queue entry rather than as
a check of its own. See its section for why that question is only worth asking
at the moment of drift.

This module is pure — no pytest import — so ``scripts/claims.py`` can use it.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SRC_ROOT = REPO_ROOT / "src"

# ``superpowers/`` is internal planning (gitignored, not built).
# ``architecture_decisions.md`` is in mkdocs.yml ``exclude_docs`` — never built.
_BLACKLIST_DIRS = {"superpowers"}
_EXCLUDED_FILES = {"architecture_decisions.md"}


class AnchorError(Exception):
    """A claim anchor is malformed, unresolvable, or not comparable."""


def published_pages() -> list[Path]:
    """Every Markdown page mkdocs actually builds."""
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(part in _BLACKLIST_DIRS for part in p.relative_to(DOCS_ROOT).parts)
        and p.relative_to(DOCS_ROOT).as_posix() not in _EXCLUDED_FILES
    )


@dataclass(frozen=True)
class Target:
    path: str                 # relative to src/, e.g. "cash/core.py"
    symbol: str | None = None  # dotted, e.g. "Cash.cache"; None means the module
    pin: str | None = None     # 8 hex chars, or "?" for an unfilled placeholder
    value: str | None = None   # raw literal text from `== ...`


@dataclass(frozen=True)
class Anchor:
    page: Path
    line: int                 # 1-based, the line the comment starts on
    claim: str                # first non-blank line after the comment
    targets: tuple[Target, ...]
    broad: str | None = None
    span: tuple[int, int] = (0, 0)  # (start, end) char offsets of the WHOLE
    # `<!-- claim: ... -->` comment in the page's raw text. `_CLAIM_RE` is
    # matched against the fence-masked text (see `parse_anchors`), but
    # masking only blanks characters -- it never changes the text's length or
    # line structure -- so `m.start()`/`m.end()` computed there index the raw
    # text identically. A rewriter (`scripts/claims.py`) that needs to edit
    # THIS anchor and no other must confine its substitution to this span:
    # matching the needle text anywhere in the page (the alternative) finds
    # whatever occurrence sorts first, including one sitting inside a fenced
    # *example* that this same masking correctly decided was not live.


def strip_code_fences(text: str) -> str:
    r"""Blank out fenced code blocks so their contents can't parse as anchors.

    An anchor shown as an *example* inside a ```` ``` ```` fence (the
    README's own "Claim anchors" section does exactly this) must not be
    parsed as a live one -- it names no real target and would sit forever as
    uncleavable drift once such a section lands on a published page.

    Each fenced line is replaced with NUL characters (``\0``) equal to its own
    length, not with an empty string: a blanked line must occupy exactly as
    many characters as the line it replaces, or every character offset after
    the first fence would drift away from its true position in the raw text.
    That property is not just cosmetic -- ``parse_anchors`` computes each
    ``Anchor.span`` from a match against this masked text and hands that span
    to callers (``scripts/claims.py``) as a raw-text index, so length
    preservation is exactly what makes a span computed here valid to slice out
    of the original, unmasked ``text``. Preserving line *count* alone (as
    blanking to ``""`` would) is not sufficient for that.

    ``\0``, not spaces: a run of plain spaces can itself accidentally satisfy
    a downstream consumer's regex -- ``test_doc_claims.py``'s table-separator
    detector (``^\s*\|?[\s:|-]+\|?\s*$``) matches a whitespace-only line, so a
    fenced line blanked to spaces right after a ``| looks | like a header |``
    line would be misread as that table's separator row. ``\0`` cannot satisfy
    any of this module's regexes (they all require a literal non-whitespace
    token -- ``#``, ``<!--``, ``[``, `` id=" ``, or one of ``\s:|-``), so it
    reliably reads as "nothing here" everywhere this masked text is consumed.

    This is the single definition: ``test_doc_claims.py`` and
    ``test_claim_anchors.py``'s false-assurance guards import it from here
    rather than keeping their own copies, which is exactly the kind of
    duplication an anti-drift mechanism shouldn't itself have.
    """
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            out.append("\0" * len(line))
            continue
        if in_fence:
            out.append("\0" * len(line))
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        out.append(line)
    return "\n".join(out)


_CLAIM_RE = re.compile(r"<!--\s*claim:\s*(?P<body>.*?)-->", re.DOTALL)
_BROAD_RE = re.compile(r'broad\s*=\s*"(?P<reason>[^"]*)"')
_TARGET_RE = re.compile(
    r"""^\s*
        (?P<path>[\w./-]+\.py)
        (?:\s*:\s*(?P<symbol>[\w.]+))?
        (?:\s*(?:@\s*(?P<pin>[0-9a-f]{8}|\?)|==\s*(?P<value>\S.*?)))?
        \s*$""",
    re.VERBOSE,
)


def ellipsize(text: str, width: int = 100) -> str:
    """Shorten *text* to *width*, breaking on a word boundary.

    Display-time only. ``Anchor.claim`` keeps the full sentence so a consumer
    that wants all of it (or wants to match against it) is not fighting a
    truncation baked in at parse time.
    """
    if len(text) <= width:
        return text
    cut = text[:width].rsplit(" ", 1)[0].rstrip(",;:.")
    return f"{cut or text[:width]}..."


def _claim_text(lines: list[str], end_line_idx: int) -> str:
    """The first non-blank line after the comment: the claim being grounded.

    Returned in full. Callers that display it apply ``ellipsize`` themselves —
    truncating here would silently shorten the value every consumer sees,
    including any future one that compares claim text rather than printing it.
    """
    for line in lines[end_line_idx + 1:]:
        if line.strip():
            return line.strip()
    return ""


def parse_anchors(text: str, page: Path) -> list[Anchor]:
    """Every claim anchor in *text*, in source order.

    Raises AnchorError on a malformed anchor. Skipping one silently would hide
    the claim it was meant to ground, which is the whole failure this exists to
    prevent.
    """
    lines = text.splitlines()
    # Masked, not the raw text: an anchor written as an illustrative example
    # inside a fenced code block must not be parsed as a live one (see
    # strip_code_fences). Blanking rather than stripping preserves every
    # line number, so positions computed against `masked` below still index
    # correctly into `lines`, which comes from the untouched original text.
    masked = strip_code_fences(text)
    out: list[Anchor] = []
    for m in _CLAIM_RE.finditer(masked):
        line_no = masked.count("\n", 0, m.start()) + 1
        end_idx = masked.count("\n", 0, m.end())
        body = m.group("body")

        broad_m = _BROAD_RE.search(body)
        broad = broad_m.group("reason") if broad_m else None
        if broad_m:
            body = _BROAD_RE.sub("", body)

        targets: list[Target] = []
        for chunk in body.split(","):
            if not chunk.strip():
                continue
            tm = _TARGET_RE.match(chunk)
            if tm is None:
                raise AnchorError(
                    f"{page}:{line_no}: cannot parse claim target {chunk.strip()!r}"
                )
            raw_value = tm.group("value")
            targets.append(
                Target(
                    path=tm.group("path"),
                    symbol=tm.group("symbol"),
                    pin=tm.group("pin"),
                    value=raw_value.strip() if raw_value else None,
                )
            )
        if not targets:
            raise AnchorError(f"{page}:{line_no}: claim comment names no target")

        out.append(
            Anchor(
                page=page,
                line=line_no,
                claim=_claim_text(lines, end_idx),
                targets=tuple(targets),
                broad=broad,
                span=(m.start(), m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _children_named(node: ast.AST, name: str) -> list[ast.AST]:
    """EVERY child binding *name*, in source order — not just the first.

    One name can have several definitions: ``@overload`` stubs before the real
    implementation, or a conditional redefinition. ``Cash.cache`` is three
    ``def cache``s, the first a one-line stub. Returning only that stub would
    pin something that never changes, leaving the anchor green forever while
    the implementation drifted -- the precise false negative this exists to
    prevent.
    """
    found: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEF_NODES) and child.name == name:
            found.append(child)
        elif isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in child.targets
        ):
            found.append(child)
        elif (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and child.target.id == name
        ):
            found.append(child)
    return found


# Parsed-source memo, keyed by resolved path. Every anchor target re-reads and
# re-``ast.parse``s its module, and the four anchor tests each resolve the whole
# corpus independently -- so ``core.py`` (4.5k lines) was being parsed hundreds
# of times per run. At 378 anchors that had pushed each of those tests to
# 13-18s against pytest's 30s per-test timeout: a spurious CI failure waiting
# for the next few anchors to be added. Memoising makes it O(files) instead of
# O(targets). Safe because the suite never edits src/ mid-run; a process
# boundary is the invalidation.
_PARSE_MEMO: dict[Path, tuple[ast.Module, str]] = {}


def _parsed(file: Path) -> tuple[ast.Module, str]:
    hit = _PARSE_MEMO.get(file)
    if hit is None:
        source = file.read_text(encoding="utf-8")
        hit = (ast.parse(source), source)
        _PARSE_MEMO[file] = hit
    return hit


def resolve(target: Target, src_root: Path = SRC_ROOT) -> tuple[list[ast.AST], str]:
    """Resolve *target* to its AST node(s) and the source of its module.

    Returns a LIST: see ``_children_named``. Callers wanting a single node
    (a value anchor) take ``nodes[-1]``, the effective definition.
    """
    file = src_root / target.path
    if not file.is_file():
        raise AnchorError(f"no such source file: src/{target.path}")
    tree, source = _parsed(file)
    if not target.symbol:
        return [tree], source

    nodes: list[ast.AST] = [tree]
    seen: list[str] = []
    for part in target.symbol.split("."):
        # Descend through the LAST binding of each intermediate name.
        found = _children_named(nodes[-1], part)
        if not found:
            where = ".".join(seen) if seen else "module scope"
            raise AnchorError(
                f"src/{target.path}: no symbol {part!r} in {where}"
            )
        nodes = found
        seen.append(part)
    return nodes, source


# --------------------------------------------------------------------------- #
# Fingerprinting                                                              #
# --------------------------------------------------------------------------- #


def normalize(node: ast.AST, source: str) -> str:
    """The node's source, stripped of everything that is not code.

    Comments, blank lines and trailing whitespace go; the result is dedented.
    Deterministic across Python versions *by construction*, unlike ``ast.dump``
    (whose node fields change between releases) — which matters because CI runs
    3.12 and development runs 3.14.
    """
    if isinstance(node, ast.Module):
        segment = source
    else:
        start = node.lineno
        # A decorator is part of what the node does, so include it.
        for deco in getattr(node, "decorator_list", []) or []:
            start = min(start, deco.lineno)
        lines = source.splitlines()
        segment = "\n".join(lines[start - 1: node.end_lineno])

    segment = textwrap.dedent(segment)

    # Column at which a comment starts, per line. tokenize (not a regex) so a
    # '#' inside a string literal is not mistaken for one.
    cuts: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
            if tok.type == tokenize.COMMENT:
                line, col = tok.start
                cuts[line] = min(cuts.get(line, col), col)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An unparseable slice keeps its comments rather than losing content.
        pass

    out: list[str] = []
    for i, line in enumerate(segment.splitlines(), 1):
        if i in cuts:
            line = line[: cuts[i]]
        line = line.rstrip()
        if line:
            out.append(line)
    return "\n".join(out)


def fingerprint(nodes: list[ast.AST], source: str) -> str:
    """8-hex-char digest over the normalized source of every node.

    Takes a list so that all of a name's definitions are covered — a change to
    any ``@overload`` signature moves the digest, not only a change to the
    implementation.
    """
    blob = "\n".join(normalize(n, source) for n in nodes)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Value anchors                                                               #
# --------------------------------------------------------------------------- #


def _is_field_call(call: ast.Call) -> bool:
    fn = call.func
    return (isinstance(fn, ast.Name) and fn.id == "field") or (
        isinstance(fn, ast.Attribute) and fn.attr == "field"
    )


# Arithmetic that ``ast.literal_eval`` refuses but that is unambiguously a
# constant: ``8 * 1024 * 1024``. That idiom is how every byte threshold in this
# codebase is written, and byte thresholds are exactly the numbers docs quote
# ("files over 8 MiB are sampled"), so without this the most drift-prone
# constants could only get a fingerprint anchor -- which proves someone looked,
# not that the documented number is right.
#
# Deliberately NOT a general evaluator: operands must bottom out in numeric
# literals, so no name lookup, no call, no attribute access, nothing that could
# execute. A non-numeric operand raises rather than guessing.
_ARITH_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Pow: lambda a, b: a**b,
}


def _fold_numeric(node: ast.AST):
    """Constant-fold a numeric expression, or raise ValueError."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("not a numeric literal")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _fold_numeric(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](
            _fold_numeric(node.left), _fold_numeric(node.right)
        )
    raise ValueError("not a constant numeric expression")


def literal_value(node: ast.AST) -> object:
    """The literal a symbol is assigned, for ``== <literal>`` anchors.

    Takes a single node — Task 4 passes ``nodes[-1]``, the effective
    definition, since only one binding can hold the current value.
    """
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        raise AnchorError(
            "a value anchor only applies to an assignment; use a fingerprint "
            "anchor for a function or class"
        )
    value = node.value
    if value is None:
        raise AnchorError("annotation has no assigned value to compare")

    if isinstance(value, ast.Call) and _is_field_call(value):
        # Checked BEFORE the default= scan below, and unconditionally (not
        # elif'd with it): a field can carry both default_factory= and a
        # stray default=, and default_factory must always win the raise, or
        # such a field would slip through the default= branch untested.
        for kw in value.keywords:
            if kw.arg == "default_factory":
                raise AnchorError(
                    "dataclasses.field(default_factory=...) has no comparable "
                    "literal; use a fingerprint anchor instead"
                )
        for kw in value.keywords:
            if kw.arg == "default":
                value = kw.value
                break
        else:
            raise AnchorError("dataclasses.field() with no default= to compare")

    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError):
        pass
    # ``literal_eval`` rejects ``8 * 1024 * 1024``; fold it ourselves, over
    # numeric literals only. See ``_fold_numeric``.
    try:
        return _fold_numeric(value)
    except ValueError as exc:
        raise AnchorError(
            f"not a literal: {ast.unparse(value)[:60]}"
        ) from exc


def values_match(documented: str, actual: object) -> bool:
    """Compare a documented literal against the value in source.

    ``documented`` must itself be a Python literal (``ast.literal_eval``-able)
    or this raises ``AnchorError`` — a malformed anchor like ``== roughly 10``
    is the author's mistake to fix, not a claim to report as "mismatched"
    against some value it never actually named.
    """
    try:
        want = ast.literal_eval(documented)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise AnchorError(
            f"documented value {documented!r} is not a Python literal; write it "
            f"as one (0.01, None, \"sha256\") or use a fingerprint anchor"
        ) from exc
    # bool FIRST and by `is`: bool is a subclass of int, so `True == 1` and
    # `False == 0` in plain Python. Falling through to `want == actual` below
    # would let a documented `True` silently match a source `1`.
    if isinstance(want, bool) or isinstance(actual, bool):
        return want is actual
    if isinstance(want, float) or isinstance(actual, float):
        try:
            return float(want) == float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    return want == actual


# --------------------------------------------------------------------------- #
# Page checking                                                               #
# --------------------------------------------------------------------------- #

MANIFEST = Path(__file__).resolve().parent / "claim_manifest.json"


@dataclass(frozen=True)
class Problem:
    page: str
    line: int
    # unresolved | drift | value | broad | unpinned | manifest | unanchored
    kind: str
    message: str
    # The anchor target this problem is about, when there is one. Defaulted so
    # every existing 4-argument construction (and the equality assertions in
    # test_claims_lib.py that spell one out) keeps working unchanged. It exists
    # so a consumer holding a drift Problem can ask the follow-up question --
    # "what else in the docs talks about THIS target?" -- without re-deriving
    # the target by parsing the message string back apart.
    target: "Target | None" = None


def anchor_count(page: Path) -> int:
    """Total anchored *targets* on a page (a comment may carry several)."""
    text = page.read_text(encoding="utf-8")
    return sum(len(a.targets) for a in parse_anchors(text, page))


def _rel(page: Path) -> str:
    """A page's repo-relative posix path, or its own path if it is outside.

    A fixture page under ``tmp_path`` is not under ``REPO_ROOT``; falling back
    to its own path keeps the unit tests reporting something readable instead
    of raising.
    """
    try:
        return page.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return page.as_posix()


def check_page(page: Path, src_root: Path = SRC_ROOT) -> list[Problem]:
    """Every problem with the claim anchors on one page."""
    rel = _rel(page)
    problems: list[Problem] = []

    for anchor in parse_anchors(page.read_text(encoding="utf-8"), page):
        for t in anchor.targets:
            try:
                nodes, source = resolve(t, src_root)
            except AnchorError as exc:
                problems.append(
                    Problem(rel, anchor.line, "unresolved", str(exc), t)
                )
                continue

            name = f"{t.path}:{t.symbol or '<module>'}"
            node = nodes[-1]  # the effective definition

            # A class or module anchor fires on every unrelated edit inside it.
            # That noise is what trains people to re-pin without reading.
            #
            # It only applies to a PINNED anchor, though. The rule exists to
            # stop fingerprint noise, and an existence anchor carries no
            # fingerprint -- ``cash/backends/redis_backend.py:RedisBackend``
            # with no pin says "this class exists", can never drift, and is the
            # correct way to ground a claim like "Cash ships a Redis backend".
            # Demanding broad="reason" for that is friction with nothing behind
            # it, and a rule that fires on correct authoring is one people learn
            # to satisfy by rote.
            is_pinned = t.pin is not None or t.value is not None
            if (
                is_pinned
                and isinstance(node, (ast.Module, ast.ClassDef))
                and not anchor.broad
            ):
                what = "module" if isinstance(node, ast.Module) else "class"
                problems.append(
                    Problem(
                        rel, anchor.line, "broad",
                        f"{name} is a {what}-level anchor; narrow it to the "
                        f"function or attribute the claim is actually about, or "
                        f'justify it with broad="reason"',
                        t,
                    )
                )
                continue

            if t.value is not None:
                try:
                    actual = literal_value(node)
                    ok = values_match(t.value, actual)
                except AnchorError as exc:
                    problems.append(
                        Problem(rel, anchor.line, "value", str(exc), t)
                    )
                    continue
                if not ok:
                    problems.append(
                        Problem(
                            rel, anchor.line, "value",
                            f"docs say {name} == {t.value}, source says {actual!r}",
                            t,
                        )
                    )
                continue

            if t.pin == "?":
                problems.append(
                    Problem(
                        rel, anchor.line, "unpinned",
                        f"{name} has an unfilled pin placeholder; run "
                        f"`python scripts/claims.py --pin`",
                        t,
                    )
                )
                continue

            if t.pin is None:
                continue  # existence-only anchor: resolving it was the check

            actual_fp = fingerprint(nodes, source)
            if actual_fp != t.pin:
                problems.append(
                    Problem(
                        rel, anchor.line, "drift",
                        f"{name} changed (@{t.pin} -> @{actual_fp}); re-read the "
                        f"claim: {ellipsize(anchor.claim)!r}",
                        t,
                    )
                )
    return problems


# --------------------------------------------------------------------------- #
# Unanchored prose about a drifted target                                     #
# --------------------------------------------------------------------------- #
#
# Everything above only ever looks at a sentence that carries an anchor. Prose
# with no anchor is invisible to it however false it becomes -- and that is not
# hypothetical: the JupyterLab live-cell branch made ``docs/magics.md``'s
# ``%%cash`` behaviour list and ``docs/getting-started/quickstart.md``'s "Google
# Colab is the exception" warning both false, and the queue could not see
# either. Both were found by hand with grep.
#
# A standing "find the unanchored claims" lint would be hopeless -- every page
# is mostly unanchored prose, and a check that fires on everything gets turned
# off. The question is only worth asking at ONE moment: when a target's
# fingerprint actually drifts, someone is already re-reading the claims that
# name it, and that is exactly when to also hand them the prose about the same
# code that names nothing.
#
# So this is TRIAGE, not a gate. It detects no falsehood; it decides no claim.
# It puts unpinned prose about the drifted target in front of the person who is
# already reading, in the same pass. Nothing here is wired into ``check_page``,
# so nothing here can turn a PR red.

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# The vocabulary that CLOSES an enumeration. Both real misses were the same
# shape -- a description of a code path that gained a new member -- and that
# shape has a lexical signature: prose asserting that the list it just gave is
# the whole list.
#
# Deliberately NOT "only" / "always" / "never" on their own, which is where the
# first cut of this went. Measured against the pre-fix tree those three add 23
# more lines to a single drifted target's triage, and every one of them was
# ordinary emphasis that was still true ("the repair never fires", "only the
# cheap append itself happens"). They describe behaviour absolutely; they do not
# close an enumeration, which is the thing a new code path falsifies.
_ENUMERATION_RE = re.compile(
    r"\b(the only|the exception|an exception|one exception|no other"
    r"|nothing else|the sole|solely)\b",
    re.IGNORECASE,
)


def _mask_html_comments(text: str) -> str:
    r"""Blank HTML comments, preserving every newline.

    A claim anchor IS an HTML comment naming its target, so without this every
    anchor would match its own target's needle and report itself as unpinned
    prose.

    Only non-newline characters are blanked. Replacing a multi-line comment with
    ``"\0" * len(match)`` -- the obvious version, and the one this started as --
    eats the newlines inside it, and every line number after the first
    multi-line comment on the page comes out short. On ``docs/magics.md`` that
    was a five-line error: exactly the kind of close-enough-to-look-right
    wrongness that makes a report worse than useless, because it names a real
    line, just not the one it found.
    """
    return _HTML_COMMENT_RE.sub(
        lambda m: re.sub(r"[^\n]", "\0", m.group(0)), text
    )


def _headings(text: str) -> list[tuple[int, int]]:
    """``(line, level)`` for every Markdown heading, 1-based, in source order."""
    out: list[tuple[int, int]] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _HEADING_RE.match(line)
        if m:
            out.append((i, len(m.group("hashes"))))
    return out


def _enclosing_section(
    headings: list[tuple[int, int]], total_lines: int, line: int
) -> tuple[int, int]:
    """The INNERMOST heading-delimited section containing *line*, 1-based.

    Innermost, not outermost, and that is load-bearing. An anchor under
    ``### %cash_on`` covers its own subsection; taking the enclosing ``##``
    instead would swallow every sibling subsection under the same parent --
    including, on ``docs/magics.md``, the ``### %%cash`` section that carried
    one of the two statements this whole mechanism exists to surface.
    """
    above = [h for h in headings if h[0] <= line]
    if not above:
        return (1, (headings[0][0] - 1) if headings else total_lines)
    start, level = above[-1]
    below = [ln for ln, lvl in headings if ln > start and lvl <= level]
    return (start, (below[0] - 1) if below else total_lines)


def _same_symbol(a: Target, b: Target) -> bool:
    """Same code target, ignoring how it happens to be pinned."""
    return a.path == b.path and a.symbol == b.symbol


def _within(spans: list[tuple[int, int]], line: int) -> bool:
    return any(start <= line <= end for start, end in spans)


def mention_pattern(target: Target) -> "re.Pattern[str] | None":
    """How *target* would be written in prose, or None if it cannot be searched.

    Two surface forms, and the second one is rationed:

    * the dotted symbol as written in the anchor (``CashMagics.cash_on``,
      ``UpstreamChecker.check_and_reexecute``), always;
    * the bare final component (``cash_on``) -- but ONLY when it cannot be
      mistaken for an ordinary English word, which here means it carries an
      underscore or an uppercase letter.

    That ration is the difference between a report and a flood, and the number
    behind it is not a guess. ``cash/core.py:Cash.cache``'s final component is
    ``cache``, which occurs on 633 prose lines across 51 published pages;
    attaching that to a drift entry would bury the entry it is attached to.
    Rationed, the same target yields 4 lines. ``cash_on`` -- underscored, so
    searchable -- yields 61 lines across 26 pages, which is simply the true
    answer for a magic that 26 pages talk about.

    A module target (no symbol) has no prose name at all and returns None; the
    enumeration rule in ``check_unanchored`` still applies to it.
    """
    if not target.symbol:
        return None
    parts = target.symbol.split(".")
    forms = [target.symbol] if len(parts) > 1 else []
    last = parts[-1]
    if "_" in last or not last.islower():
        forms.append(last)
    if not forms:
        return None
    # Longest-first so the dotted form wins over its own final component and the
    # reported match is the more specific one.
    alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    # ``%{0,2}`` so ``%cash_on`` and ``%%cash`` read as mentions of the method
    # behind them. The lookbehind rejects ``foo.cash_on``-style attribute access
    # on some other object; the lookahead stops ``cache`` matching inside
    # ``cache_info``.
    return re.compile(r"(?<![\w.])%{0,2}(?:" + alternatives + r")(?![\w])")


def check_unanchored(
    target: Target, pages: list[Path] | None = None
) -> list[Problem]:
    """Published prose about *target* that pins nothing -- triage, never a gate.

    Two rules, because the two real misses needed two different ones:

    ``names``
        A prose line anywhere in the docs that writes the target's symbol (see
        ``mention_pattern``), outside any section already anchored to that same
        target. Section-scoped, NOT page-scoped: the ``docs/magics.md`` miss sat
        on the page that anchors ``CashMagics.cash_on``, several sections away
        from the anchor, so excluding the whole anchoring page -- the obvious
        rule -- surfaces nothing at all there.

    ``enumeration``
        A line that closes an enumeration (see ``_ENUMERATION_RE``) on a page
        that anchors this target, sitting outside EVERY anchored section on that
        page. This is the rule that reaches the ``quickstart.md`` miss, whose
        false sentence -- "**Google Colab is the exception**" -- names no symbol
        at all and so cannot be found by name from any direction. What links it
        to the target is co-location: the page is already pinned to this code,
        and this sentence about it is not.

    Fences are masked (an example is not a claim) and HTML comments are masked
    (an anchor names its target by definition, and would otherwise report
    itself).

    Honest about its reach: neither rule knows whether the prose is TRUE. This
    answers "what else talks about this code without pinning it" -- a question a
    human settles in a glance and a machine cannot settle at all.
    """
    needle = mention_pattern(target)
    name = f"{target.path}:{target.symbol or '<module>'}"
    out: list[Problem] = []

    for page in (published_pages() if pages is None else pages):
        rel = _rel(page)
        text = page.read_text(encoding="utf-8")
        prose = _mask_html_comments(strip_code_fences(text)).splitlines()
        headings = _headings(text)
        total = len(text.splitlines())

        anchored: list[tuple[int, int]] = []
        anchored_here: list[tuple[int, int]] = []
        for anchor in parse_anchors(text, page):
            span = _enclosing_section(headings, total, anchor.line)
            anchored.append(span)
            if any(_same_symbol(t, target) for t in anchor.targets):
                anchored_here.append(span)

        for i, line in enumerate(prose, 1):
            if not line.strip():
                continue
            if needle and needle.search(line) and not _within(anchored_here, i):
                out.append(
                    Problem(
                        rel, i, "unanchored",
                        f"names {name} but pins nothing: "
                        f"{ellipsize(line.strip())!r}",
                        target,
                    )
                )
            elif (
                anchored_here
                and _ENUMERATION_RE.search(line)
                and not _within(anchored, i)
            ):
                out.append(
                    Problem(
                        rel, i, "unanchored",
                        f"closed enumeration on a page that pins {name}, but "
                        f"this line pins nothing: {ellipsize(line.strip())!r}",
                        target,
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# Manifest / coverage ratchet                                                 #
# --------------------------------------------------------------------------- #


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def check_manifest() -> list[Problem]:
    """The coverage ratchet: an audited page may gain anchors, never lose them."""
    manifest = load_manifest()
    pages = {p.relative_to(REPO_ROOT).as_posix(): p for p in published_pages()}
    problems: list[Problem] = []

    for rel in sorted(set(manifest) - set(pages)):
        problems.append(
            Problem(
                rel, 0, "manifest",
                "listed in claim_manifest.json but is not a published page; "
                "remove the entry or restore the page",
            )
        )

    for rel, page in sorted(pages.items()):
        entry = manifest.get(rel)
        if entry is None:
            problems.append(
                Problem(
                    rel, 0, "manifest",
                    'not in claim_manifest.json; add {"audited": null, '
                    '"anchors": 0} and triage the page',
                )
            )
            continue
        if not entry.get("audited"):
            continue  # unaudited pages are exempt until their tranche lands
        actual = anchor_count(page)
        if actual < entry["anchors"]:
            problems.append(
                Problem(
                    rel, 0, "manifest",
                    f"anchor count fell from {entry['anchors']} to {actual}; a "
                    f"claim was removed along with its anchor",
                )
            )
    return problems
