"""The interactive checker on the Safety page gives the verdicts cash gives.

``docs/javascripts/cacheability-checker.js`` restates the notebook policy as a
list of snippets, each with a verdict, and ``docs/how-it-works/safety.md``
carries the same list as the table shown without JavaScript. Both are prose
about ``decide_cacheability``; nothing tied them to it, so they could say
"cached" about a statement cash runs every time. Each snippet is judged here
by the real analysis and the real forbidden-function scan.
"""

from __future__ import annotations

import ast
import json
import re
import types
from pathlib import Path

import pytest

from cash.analysis.annotations import get_statement_annotations
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.analysis.code_analyzer import CodeAnalyzer
from cash.tracking.randomness import RandomnessDetector

DOCS = Path(__file__).resolve().parents[2] / "docs"
SCRIPT = DOCS / "javascripts" / "cacheability-checker.js"
PAGE = DOCS / "how-it-works" / "safety.md"

_SNIPPET = re.compile(r'code:\s*"((?:[^"\\]|\\.)*)",\s*verdict:\s*"(\w+)"')
_ROW = re.compile(r"<tr><td><code>(.*?)</code></td><td>(.*?)</td></tr>")

#: Decided from the live receiver by the statement processor (the in-place
#: fit rule), not by the static analysis judged here.
_DECIDED_AT_RUN_TIME = {"model.fit(X, y)"}


def _snippets() -> list[tuple[str, str]]:
    found = [(json.loads(f'"{code}"'), verdict) for code, verdict in _SNIPPET.findall(SCRIPT.read_text())]
    assert found, "no snippets parsed from the checker script"
    return found


def _fallback_rows() -> dict[str, str]:
    text = PAGE.read_text()
    block = text[text.index('<div class="cash-cacheability-checker"') :]
    block = block[: block.index("</div>")]
    rows = {code.replace("&#39;", "'").replace("&quot;", '"'): verdict for code, verdict in _ROW.findall(block)}
    assert rows, "no rows parsed from the fallback table"
    return rows


def _verdict(code: str) -> str:
    tree = ast.parse(code)
    inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree)
    namespace: dict[str, object] = {name: object() for name in inputs}
    namespace["requests"] = types.ModuleType("requests")
    cacheable, _reasons = decide_cacheability(
        code=code,
        tree=tree,
        inputs=inputs,
        outputs=outputs,
        annotation=get_statement_annotations(code, tree.body[0]),
        analysis=analyze_statement(code, tree),
        user_ns=namespace,
        variable_lineage={name: "lineage" for name in inputs},
        is_stateful_call=lambda _name: False,
        scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
    )
    if not cacheable:
        return "no"
    unseeded, _messages, _seeded = RandomnessDetector().analyze_code(code)
    return "warn" if unseeded else "ok"


@pytest.mark.parametrize(("code", "verdict"), _snippets(), ids=[code for code, _ in _snippets()])
def test_each_snippet_gets_the_verdict_it_shows(code, verdict):
    if code in _DECIDED_AT_RUN_TIME:
        pytest.skip("decided from the live receiver, not by the static analysis")
    assert _verdict(code) == verdict


def test_the_fallback_table_lists_the_same_snippets_and_verdicts():
    label = {"ok": "Cached —", "warn": "Cached + warning", "no": "Not cached"}
    rows = _fallback_rows()
    snippets = _snippets()
    assert list(rows) == [code for code, _ in snippets]
    for code, verdict in snippets:
        assert rows[code].startswith(label[verdict]), (code, rows[code])
