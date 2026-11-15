"""A cached function's helpers, analysed in one process and read back in the
next, give what plain Python gives after every edit.

A fresh process reads back what earlier ones worked out from a helper's text
(`cash.analysis.text_store`: the globals its body changes, the imports in its
body, its key digest, its annotation names) and the purity report of the
whole walk (`report_store`). Each is used only while everything it was
worked out from is the same, so an edit to a helper, to a module a body
imports, a new helper file and a new library version each give the value
plain Python gives, in a new process, with the stores on.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_MAIN = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    sys.path.insert(0, sys.argv[2])
    import cash
    from app.helpers import lookup, rate, annotated

    def plain(k):
        return lookup(k) * rate() + annotated(k)

    cached = cash.cache(plain)
    value = cached(1)
    from cash.analysis import report_store, text_store
    print(json.dumps({
        "value": value,
        "plain": plain(1),
        "hits": cached.cache_info()["hits"],
        "text": text_store.status(),
        "reports": report_store.STATS,
    }))
    """
)

_HELPERS = textwrap.dedent(
    """
    from app.settings import TABLE
    import fakelib


    def lookup(k):
        return TABLE[k]


    def rate():
        from app import settings as s
        return fakelib.scale(s.RATE)


    def annotated(x: "int") -> int:
        return x + 1
    """
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _library(site: Path, version: str, factor: int) -> None:
    _write(site / "fakelib" / "__init__.py", f"def scale(x):\n    return x * {factor}\n")
    for old in site.glob("fakelib-*.dist-info"):
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    info = site / f"fakelib-{version}.dist-info"
    _write(info / "METADATA", f"Metadata-Version: 2.1\nName: fakelib\nVersion: {version}\n")
    _write(info / "top_level.txt", "fakelib\n")
    _write(info / "RECORD", "")


@pytest.fixture
def project(tmp_path):
    code = tmp_path / "code"
    site = tmp_path / "site-packages"
    _write(code / "app" / "__init__.py", "")
    _write(code / "app" / "settings.py", "RATE = 2\nTABLE = [1, 2, 3]\n")
    _write(code / "app" / "helpers.py", _HELPERS)
    _write(code / "main.py", _MAIN)
    _library(site, "1.0", 2)
    (code / ".cash").mkdir()  # the stores live in an existing cache dir
    return code, site


def _run(project, verify: bool = False) -> dict:
    code, site = project
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("main.py", str(site), str(code), cwd=code, env=env)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["value"] == result["plain"]
    return result


def test_a_second_process_reads_the_analysis_back_and_hits(project):
    first = _run(project)
    assert first["hits"] == 0 and first["text"]["computed"] > 0
    second = _run(project)
    assert second["hits"] == 1 and second["value"] == first["value"]
    assert second["text"]["loaded"] > 0 and second["text"]["computed"] == 0
    # Each stored answer is what working it out again gives.
    third = _run(project, verify=True)
    assert third["hits"] == 1 and third["text"]["verified"] == third["text"]["loaded"] > 0


def test_every_edit_gives_what_plain_python_gives(project):
    code, site = project
    helpers = code / "app" / "helpers.py"
    values = [_run(project)["value"]]
    _run(project)  # read back from the stores
    edits = [
        # A value a body's own import reads.
        lambda: _write(code / "app" / "settings.py", "RATE = 5\nTABLE = [1, 2, 3]\n"),
        # A global the helper reads.
        lambda: _write(code / "app" / "settings.py", "RATE = 5\nTABLE = [10, 20, 30]\n"),
        # The helper's own text.
        lambda: _write(helpers, _HELPERS.replace("return TABLE[k]", "return TABLE[k] * 10")),
        # The helper now changes the global it reads.
        lambda: _write(helpers, _HELPERS.replace("return TABLE[k]", "TABLE[0] += 0\n    return TABLE[k] * 7")),
        # A new helper file, imported in a body.
        lambda: (
            _write(code / "app" / "extra.py", "def bump(x):\n    return x + 100\n"),
            _write(helpers, _HELPERS.replace("return x + 1", "from app import extra\n    return extra.bump(x)")),
        ),
        # An edit to that new file.
        lambda: _write(code / "app" / "extra.py", "def bump(x):\n    return x + 1000\n"),
        # A new version of the library, with new code.
        lambda: _library(site, "2.0", 3),
    ]
    for edit in edits:
        edit()
        changed = _run(project)
        assert changed["value"] not in values, changed
        values.append(changed["value"])
        again = _run(project, verify=True)
        assert again["value"] == changed["value"]
