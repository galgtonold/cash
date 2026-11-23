"""Every place the analysis reads source text tells the walk recording it.

A stored purity report is checked against the text of what the walk
visited and against what else of a file it read (`cash.source_reads`), not
against whole files: an edit to a function nothing read keeps the report. So
a read that is not noted is an edit the report would miss. This pins the two
ways a read can go unnoted:

* a function of the analysis that reads source past `cash.source_reading`
  (whose `getsourcelines` and `read_code_file` note it themselves): with
  ``inspect``, ``linecache`` or ``open``;
* a memo keyed by a function's code that answers before the source is read
  again: it notes the function before it looks.

A new site fails here until it notes what it reads, or is listed below with
why it need not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cash
from cash import source_reads
from cash.analysis import purity_flow

_ANALYSIS = Path(cash.__file__).parent / "analysis"

#: Direct reads.
_READERS = {
    ("inspect", "getsource"),
    ("inspect", "getsourcelines"),
    ("inspect", "findsource"),
    ("linecache", "getlines"),
    ("linecache", "getline"),
    (None, "open"),
}

#: (module, function) -> why it reads without noting.
_EXEMPT = {
    # A key of the analyzer's own memo for code with no file, outside any walk.
    ("purity_analyzer", "_code_memo_key"): "a memo key, outside the walk",
    # Whether a pseudo-file (``<string>``) has lines at all: no file.
    ("helper_code", "_function_is_exec_built"): "a pseudo file",
    # A module not loaded yet, read to vouch for what importing it binds:
    # all the walk takes from it is whether it can (`source_digest` is not
    # None), which the stored report works out again on every check
    # (`report_store._guards`, "imports").
    ("unloaded_modules", "_source_digest"): "checked again as an import",
    ("unloaded_modules", "_file_digest"): "checked again as an import",
    ("unloaded_modules", "_users_function"): "checked again as an import",
}


def _call_target(node: ast.Call) -> tuple[str | None, str] | None:
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id, f.attr
    if isinstance(f, ast.Name):
        return None, f.id
    return None


def _notes(fn: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "source_reads"
        and n.func.attr.startswith("note_")
        for n in ast.walk(fn)
    )


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_every_direct_source_read_notes_it_or_says_why_not():
    unnoted = []
    for path in sorted(_ANALYSIS.glob("*.py")):
        if path.stem in ("scan_store", "text_store", "fact_store"):
            continue  # they read their own records, never source
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in _functions(tree):
            reads = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and _call_target(n) in _READERS
                # a nested function is judged on its own
                and not any(n in ast.walk(inner) for inner in _functions(fn) if inner is not fn)
            ]
            if reads and not _notes(fn) and (path.stem, fn.name) not in _EXEMPT:
                unnoted.append(f"{path.stem}.{fn.name} (line {reads[0].lineno})")
    assert not unnoted, unnoted


#: The functions that consult a memo keyed by a function's code before
#: reading its source: each notes the function first.
_CODE_KEYED = [
    ("ambient_reads", "clock_helper_read"),
    ("ambient_reads", "_is_log_helper_function"),
    ("handed_callables", "_source_of"),
    ("handed_callables", "_call_facts"),
    ("namespace_effects", "code_waived_lines"),
    ("namespace_effects", "_examined_body"),
    ("namespace_effects", "resolve"),
]


def test_a_memo_keyed_by_code_notes_the_function_before_it_answers():
    missing = []
    for module, name in _CODE_KEYED:
        tree = ast.parse((_ANALYSIS / f"{module}.py").read_text(encoding="utf-8"))
        found = [fn for fn in _functions(tree) if fn.name == name]
        if not found or not all(_notes(fn) for fn in found):
            missing.append(f"{module}.{name}")
    assert not missing, missing


def test_a_memoised_returns_answer_notes_what_working_it_out_read(tmp_path):
    """`purity_flow.returns_fresh` keeps its answers for one walk: a body
    asking again notes the source the first asking read."""
    (tmp_path / "rf_mod.py").write_text("def made(n):\n    return [0] * n\n", encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("rf_mod", tmp_path / "rf_mod.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with purity_flow.returns_memo():
        with source_reads.recording() as first:
            purity_flow.returns_fresh(module.made)
        with source_reads.recording() as again:
            purity_flow.returns_fresh(module.made)
    assert id(module.made) in first.objects
    assert again.objects.keys() == first.objects.keys()
