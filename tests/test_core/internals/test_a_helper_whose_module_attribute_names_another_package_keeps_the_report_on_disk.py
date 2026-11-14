"""A cached function whose walk reaches a helper that names, as its
``__module__``, a package that does not hold it still has its purity report
stored, and a fresh process reads it instead of walking again.

NumPy's ``set_module`` names ``numpy._core.numerictypes.issubclass_`` as
``numpy.issubclass_``, which NumPy 2 no longer exports, so ``matrix_rank``
(reached from a nilearn GLM) could not be found again by name and was
walked in every process. The module the helper was defined in holds it,
and finds it again.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app
    from cash.analysis import report_store
    top = cash.cache(app.top)
    value = top(3)
    key = top.explain(3).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "key": key, "value": value}))
    """
)

_APP = """
import pkg


def top(n):
    return pkg.outer(n) * 2
"""

#: The package exports `outer` alone, and both name it as their module.
_INIT = "from pkg._core import outer\n"

_CORE = """
def set_module(name):
    def mark(func):
        func.__module__ = name
        return func
    return mark


@set_module("pkg")
def inner(x):
    return x + 1


@set_module("pkg")
def outer(x):
    return inner(x) * 3
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_report_through_a_helper_renamed_by_set_module_is_read_in_a_new_process(tmp_path):
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / "pkg" / "__init__.py").write_text(_INIT, encoding="utf-8")  # holds no `inner`
    (code / "pkg" / "_core.py").write_text(_CORE, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")
    first = _run(code)
    assert first["value"] == 24
    assert first["stats"]["stored"] >= 1, first["stats"]
    second = _run(code)
    assert second["stats"]["loaded"] >= 1, second["stats"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 1


def test_an_edit_to_the_renamed_helper_walks_again_and_changes_the_key(tmp_path):
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / "pkg" / "__init__.py").write_text(_INIT, encoding="utf-8")
    (code / "pkg" / "_core.py").write_text(_CORE, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")
    first = _run(code)
    (code / "pkg" / "_core.py").write_text(_CORE.replace("x + 1", "x + 2"), encoding="utf-8")
    edited = _run(code)
    assert edited["value"] == 30
    assert edited["key"] != first["key"]
    assert edited["stats"]["loaded"] == 0, edited["stats"]
