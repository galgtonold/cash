"""cash's dependencies with compiled code, and the Pythons they release for.

``blake3`` and ``psutil`` ship compiled wheels. Where one has no wheel, pip
builds it from source or, told to use wheels only, walks back to the last
cash release that did not depend on it and installs that without a word.
Two guards:

- ``blake3`` is asked for on CPython only (it has no wheels elsewhere, and
  cash hashes with SHA-256 without it);
- ``tools/dependency_floor.py``, run nightly on the oldest supported Python,
  reports a dependency whose newest release no longer installs there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import dependency_floor

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def _dependencies() -> list[str]:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("\ndependencies = [", 1)[1].split("\n]\n", 1)[0]
    return re.findall(r'^\s*"([^"]+)",\s*$', block, re.MULTILINE)


def test_blake3_is_asked_for_on_cpython_only():
    [blake3] = [d for d in _dependencies() if d.startswith("blake3")]
    assert blake3.endswith("; platform_python_implementation == 'CPython'"), blake3


def test_on_cpython_every_dependency_applies_and_on_pypy_blake3_does_not():
    cpython = {"platform_python_implementation": "CPython", "python_version": "3.12"}
    pypy = {"platform_python_implementation": "PyPy", "python_version": "3.11"}
    assert dependency_floor.applying_dependencies(_dependencies(), cpython) == ["psutil", "blake3"]
    assert dependency_floor.applying_dependencies(_dependencies(), pypy) == ["psutil"]
    old = {"platform_python_implementation": "CPython", "python_version": "3.10"}
    assert dependency_floor.applying_dependencies(_dependencies(), old) == ["psutil", "tomli", "blake3"]


def test_an_extra_is_not_a_dependency():
    lines = ["psutil>=5", 'mkdocs>=1.5; extra == "docs"', "pytest; python_version >= '3.10' and extra == 'dev'"]
    assert dependency_floor.applying_dependencies(lines) == ["psutil"]


def test_only_a_dependency_older_than_the_newest_release_is_behind():
    installed = {"blake3": "1.0.10", "psutil": "7.2.2", "tomli": "2.3.0"}
    newest = {"blake3": "1.0.11", "psutil": "7.2.2", "tomli": "2.3.0"}
    assert dependency_floor.behind(installed, newest) == [("blake3", "1.0.10", "1.0.11")]
    assert dependency_floor.behind(newest, newest) == []
    # Compared as versions, not text: 1.0.10 is newer than 1.0.9.
    assert dependency_floor.behind({"blake3": "1.0.9"}, {"blake3": "1.0.10"}) == [("blake3", "1.0.9", "1.0.10")]
    assert dependency_floor.behind({"blake3": "1.0.10"}, {"blake3": "1.0.9"}) == []


def test_the_report_warns_and_never_fails_the_job(monkeypatch, capsys, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(dependency_floor.metadata, "requires", lambda name: ["blake3>=1.0,<2", "psutil>=5.0,<8"])
    monkeypatch.setattr(
        dependency_floor.metadata, "version", lambda name: {"blake3": "1.0.10", "psutil": "7.2.2"}[name]
    )
    monkeypatch.setattr(dependency_floor, "_newest_on_pypi", lambda name: {"blake3": "1.0.11", "psutil": "7.2.2"}[name])
    assert dependency_floor.main([]) == 0
    out = capsys.readouterr().out
    [warning] = [line for line in out.splitlines() if line.startswith("::warning ")]
    assert "blake3 1.0.11 does not install on Python" in warning and "pip installed blake3 1.0.10" in warning
    assert "psutil" not in warning
    assert "`blake3`: installed 1.0.10, newest 1.0.11" in summary.read_text(encoding="utf-8")


def test_nothing_behind_says_nothing_more(monkeypatch, capsys, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(dependency_floor.metadata, "requires", lambda name: ["psutil>=5.0,<8"])
    monkeypatch.setattr(dependency_floor.metadata, "version", lambda name: "7.2.2")
    monkeypatch.setattr(dependency_floor, "_newest_on_pypi", lambda name: "7.2.2")
    assert dependency_floor.main([]) == 0
    assert "::warning" not in capsys.readouterr().out
    assert not summary.exists()


@pytest.mark.parametrize("page", ["docs/getting-started/installation.md"])
def test_the_install_page_says_which_interpreter_is_tested(page):
    text = " ".join((REPO_ROOT / page).read_text(encoding="utf-8").split())
    assert "tested on CPython" in text
    assert "hashes with SHA-256" in text
