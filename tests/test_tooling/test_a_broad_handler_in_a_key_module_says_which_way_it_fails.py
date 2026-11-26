"""A broad ``except`` in a module that decides a key says which way it fails.

``except Exception`` in code that builds a key, a digest or a lineage hides
one of two very different things. Either the failure costs a miss, an
uncached run or a value hashed again (**fails closed**), or it lets a key
be built without something the result depends on, so that a stale result
can be served (**fails open**). The written reason beside a handler
("never break a call") does not tell a reader which, and nothing told a
test.

Each such handler in the modules below carries one of the two tags on its
``except`` line, after the reason marker::

    except Exception:  # noqa: BLE001 - fails closed: pandas internals changed: read the column
    except Exception:  # noqa: BLE001 - fails open: no metadata: this library's version does not reach the key

A handler whose last statement re-raises decides nothing and needs none.
A handler that fails open is a decision, counted here per module: a new
one fails this test until it is made to fail closed or the count is raised
on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "cash"

#: The modules that build a key, a digest or a lineage, or decide whether
#: a remembered one still holds.
KEY_MODULES = (
    "_pandas_compat.py",
    "canonical_form.py",
    "code_digest.py",
    "content_hashers.py",
    "cython_digest.py",
    "kept_state.py",
    "mutation_fingerprint.py",
    "source_norm.py",
    "source_reading.py",
    "value_hash.py",
    "value_watch.py",
    "decorator/arg_hashing.py",
    "decorator/arg_key.py",
    "decorator/cash_key.py",
    "decorator/class_data.py",
    "decorator/class_memo.py",
    "decorator/closure_fold.py",
    "decorator/code_args.py",
    "decorator/code_surface.py",
    "decorator/code_trace.py",
    "decorator/dynamic_sources.py",
    "decorator/environment_fold.py",
    "decorator/file_deps.py",
    "decorator/function_identity.py",
    "decorator/global_reads.py",
    "decorator/global_values.py",
    "decorator/globals_fold.py",
    "decorator/helper_fold_memo.py",
    "decorator/hit_memo.py",
    "decorator/library_versions.py",
    "decorator/module_attrs.py",
    "decorator/runtime.py",
    "decorator/state_memo.py",
    "notebook/cache_key.py",
    "notebook/call_key.py",
    "notebook/callee_reach.py",
    "notebook/helper_memos.py",
    "notebook/lineage_formula.py",
    "notebook/lineage_store.py",
    "notebook/statement/processor.py",
    "notebook/upstream/statement_lineage.py",
)

#: Key modules whose handlers are not tagged yet, each owned by other work
#: in flight when the rule came in. Tag a file's handlers, then drop it
#: here: the test fails for a file listed that has none left untagged.
PENDING = frozenset({"notebook/call_key.py"})

#: How many handlers fail open, per module. The number may only go down:
#: make one fail closed (a miss, an uncached run, a refusal) and lower it.
FAILS_OPEN = {
    "content_hashers.py": 1,
    "mutation_fingerprint.py": 1,
    "source_reading.py": 1,
    "decorator/code_args.py": 1,
    "decorator/code_trace.py": 2,
    "decorator/global_values.py": 2,
    "decorator/library_versions.py": 4,
    "decorator/module_attrs.py": 1,
    "decorator/runtime.py": 1,
    "notebook/cache_key.py": 1,
    "notebook/callee_reach.py": 4,
    "notebook/helper_memos.py": 2,
    "notebook/statement/processor.py": 1,
}

TAGS = ("fails closed", "fails open")


def _is_broad(handler: ast.ExceptHandler) -> bool:
    kind = handler.type
    if kind is None:
        return True
    names = kind.elts if isinstance(kind, ast.Tuple) else [kind]
    return any(isinstance(name, ast.Name) and name.id in ("Exception", "BaseException") for name in names)


def broad_handlers(source: str) -> list[tuple[int, str | None]]:
    """``(line, tag)`` for each broad handler in *source* that does not end
    by raising again; *tag* is None for one that carries neither tag."""
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
            continue
        if isinstance(node.body[-1], ast.Raise):
            continue
        comment = lines[node.lineno - 1].partition("#")[2]
        tag = next((tag for tag in TAGS if tag in comment), None)
        found.append((node.lineno, tag))
    return sorted(found)


def _handlers(module: str) -> list[tuple[int, str | None]]:
    return broad_handlers((ROOT / module).read_text(encoding="utf-8"))


def test_every_key_module_exists():
    missing = [module for module in KEY_MODULES if not (ROOT / module).is_file()]
    assert not missing, f"moved or renamed (update KEY_MODULES): {missing}"


@pytest.mark.parametrize("module", [m for m in KEY_MODULES if m not in PENDING])
def test_each_broad_handler_is_tagged(module):
    untagged = [line for line, tag in _handlers(module) if tag is None]
    assert not untagged, (
        f"cash/{module}: broad handler(s) at line(s) {untagged} do not say which way they fail. "
        "Write `fails closed: <why>` (the failure costs a miss or an uncached run) or "
        "`fails open: <what goes unkeyed>` after the reason marker on the `except` line."
    )


@pytest.mark.parametrize("module", [m for m in KEY_MODULES if m not in PENDING])
def test_no_new_handler_fails_open(module):
    count = sum(1 for _line, tag in _handlers(module) if tag == "fails open")
    allowed = FAILS_OPEN.get(module, 0)
    assert count <= allowed, (
        f"cash/{module} has {count} handlers that fail open, {allowed} are accounted for: a failure there "
        "could let a wrong key or a stale result through. Make the new one fail closed."
    )
    assert count == allowed, f"cash/{module}: lower FAILS_OPEN to {count}; one was made to fail closed"


def test_a_pending_file_still_has_untagged_handlers():
    done = [module for module in PENDING if all(tag for _line, tag in _handlers(module))]
    assert not done, f"all tagged now; drop from PENDING: {done}"


def test_the_scan_sees_each_shape():
    source = (
        "try:\n    pass\nexcept Exception:  # noqa: BLE001 - fails closed: read again\n    x = 1\n"
        "try:\n    pass\nexcept (KeyError, Exception):  # fails open: left out of the key\n    x = 2\n"
        "try:\n    pass\nexcept BaseException:\n    x = 3\n"
        "try:\n    pass\nexcept:\n    x = 4\n"
        "try:\n    pass\nexcept Exception as e:\n    raise ValueError('x') from e\n"
        "try:\n    pass\nexcept KeyError:\n    x = 5\n"
    )
    assert broad_handlers(source) == [(3, "fails closed"), (7, "fails open"), (11, None), (15, None)]


def test_a_tag_in_the_body_does_not_count():
    source = "try:\n    pass\nexcept Exception:\n    x = 1  # fails closed\n"
    assert broad_handlers(source) == [(3, None)]
