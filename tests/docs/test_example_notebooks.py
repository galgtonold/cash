"""The example notebooks must run, top to bottom, the way a reader runs them.

Two layers:

* **Static checks, always on.** Every notebook under ``examples/`` opens with the
  setup cell the docs teach (``import cash`` then ``%cash_on``, alone) and
  carries none of the leftovers that made the examples mislead: ``%load_ext
  cash``, ``%%time`` (a timed cell is never cached), debug toggles, or the
  ``time.sleep(1.1)`` mtime workaround.
* **Execution, opt-in.** ``CASH_RUN_EXAMPLE_NOTEBOOKS=1`` runs each notebook on
  ``RUNNABLE`` in a fresh kernel with nbclient, offline, in a scratch copy of
  ``examples/``, and fails on any error output or unexpected stderr (a
  ``UsageError`` from a magic, a cash warning). The ``example-notebooks`` job in
  ``.github/workflows/ci.yml`` sets it; locally::

      CASH_RUN_EXAMPLE_NOTEBOOKS=1 pytest tests/docs/test_example_notebooks.py -n 4

Every notebook is either on ``RUNNABLE`` or in ``NOT_RUN`` with a reason, and
a test fails when a new one is on neither, so nothing is silently left out.

The examples link to the published docs by absolute URL, which ``mkdocs build``
never checks, so a test resolves each such link to a page under ``docs/`` and,
when it names one, to a heading anchor on that page.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
DOCS = REPO / "docs"

#: Run by the execution test: offline, small data, well under a minute each.
RUNNABLE = [
    "cache_calls_demo.ipynb",
    "cfd_simulation_demo.ipynb",
    "demo_cell_caching.ipynb",
    "demo_notebook_caching.ipynb",
    "file_caching_demo.ipynb",
    "file_tracking_demo.ipynb",
    "financial_analysis_demo.ipynb",
    "try_cash_binder.ipynb",
]

#: Never executed here, and why. Glob patterns relative to ``examples/``.
NOT_RUN = {
    "try_cash_colab.ipynb": "installs cash from PyPI; its content is try_cash_binder.ipynb's, which runs",
    "large_scale_projects/*.ipynb": "downloads real datasets of hundreds of MB to several GB",
}

#: stderr that reflects the test environment rather than the notebook: nbclient
#: has no Jupyter server for cash to find the notebook through.
_ENV_STDERR = ("NOTEBOOK-NOT-FOUND",)

_FORBIDDEN = {
    r"^\s*%load_ext\s+cash": "set cash up with `import cash` then `%cash_on`, not `%load_ext cash`",
    r"^\s*%%time\b": "a `%%time` cell is never cached; the badge already reports timing",
    r"^\s*%cash_debug\b": "debug toggles do not belong in an example",
    r"time\.sleep\(1\.1\)": "cash checks file content, so no mtime workaround is needed",
}


def _all_notebooks() -> list[Path]:
    return sorted(p for p in EXAMPLES.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _rel(path: Path) -> str:
    return path.relative_to(EXAMPLES).as_posix()


def _ids(paths: list[Path]) -> list[str]:
    return [_rel(p) for p in paths]


def test_every_notebook_is_classified():
    unclassified = [
        _rel(p) for p in _all_notebooks() if _rel(p) not in RUNNABLE and not any(p.match(g) for g in NOT_RUN)
    ]
    assert not unclassified, (
        f"add these to RUNNABLE, or to NOT_RUN with a reason, in {Path(__file__).name}: {unclassified}"
    )
    missing = [name for name in RUNNABLE if not (EXAMPLES / name).is_file()]
    assert not missing, f"RUNNABLE names notebooks that do not exist: {missing}"


@pytest.mark.parametrize("path", _all_notebooks(), ids=_ids(_all_notebooks()))
def test_first_code_cell_is_the_setup_cell(path: Path):
    cells = _code_cells(path)
    if path.name == "try_cash_colab.ipynb":
        cells = cells[1:]  # Colab installs cash first; see scripts/build_try_cash_colab.py
    assert cells and cells[0].split() == ["import", "cash", "%cash_on"], (
        f"{_rel(path)}: the first code cell must be `import cash` then `%cash_on`, and nothing else "
        f"(work in the %cash_on cell is never cached). Found: {cells[0] if cells else None!r}"
    )


@pytest.mark.parametrize("path", _all_notebooks(), ids=_ids(_all_notebooks()))
def test_no_misleading_leftovers(path: Path):
    problems = []
    for i, src in enumerate(_code_cells(path)):
        for pattern, why in _FORBIDDEN.items():
            if re.search(pattern, src, re.MULTILINE):
                problems.append(f"code cell {i}: {why}")
    assert not problems, f"{_rel(path)}:\n  " + "\n  ".join(problems)


_DOCS_URL = re.compile(
    r"https://cash-lib\.readthedocs\.io/en/latest/(?P<page>[\w/.-]*?)/?(?:#(?P<anchor>[\w-]+))?(?=[)\s\"'>]|$)"
)
_HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*(?:\{[^}]*#(?P<id>[\w-]+)[^}]*\})?\s*$", re.MULTILINE)


def _slug(text: str) -> str:
    """python-markdown's default heading slug, as mkdocs uses it."""
    text = unicodedata.normalize("NFKD", re.sub(r"<[^>]+>", "", text)).encode("ascii", "ignore").decode()
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text).strip().lower())


_RAW_ID = re.compile(r"<a\s+(?:name|id)=\"(?P<id>[\w-]+)\"")


def _anchors(md: Path) -> set[str]:
    """Heading anchors (slug or ``{ #id }``) and raw ``<a id>`` anchors on a page."""
    text = md.read_text(encoding="utf-8")
    headings = {m.group("id") or _slug(m.group("text")) for m in _HEADING.finditer(text)}
    return headings | {m.group("id") for m in _RAW_ID.finditer(text)}


def _example_files() -> list[Path]:
    return [p for p in sorted(EXAMPLES.rglob("*")) if p.suffix in (".ipynb", ".md", ".py")]


def test_docs_links_in_the_examples_resolve():
    broken = []
    for path in _example_files():
        for m in _DOCS_URL.finditer(path.read_text(encoding="utf-8")):
            page = m.group("page")
            candidates = [DOCS / "index.md"] if not page else [DOCS / f"{page}.md", DOCS / page / "index.md"]
            md = next((c for c in candidates if c.is_file()), None)
            if md is None:
                broken.append(f"{_rel(path)}: no docs page for {m.group(0)}")
            elif m.group("anchor") and m.group("anchor") not in _anchors(md):
                broken.append(f"{_rel(path)}: no heading #{m.group('anchor')} on {md.relative_to(REPO)}")
    assert not broken, "\n".join(broken)


def _unexpected_stderr(nb) -> list[str]:
    out = []
    for i, cell in enumerate(nb.cells):
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream" and o.get("name") == "stderr":
                text = o.get("text", "")
                if text.strip() and not any(marker in text for marker in _ENV_STDERR):
                    out.append(f"cell {i}: {text.strip()[:500]}")
    return out


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get("CASH_RUN_EXAMPLE_NOTEBOOKS") != "1",
    reason="executes notebooks in real kernels; set CASH_RUN_EXAMPLE_NOTEBOOKS=1 (the example-notebooks CI job does)",
)
@pytest.mark.parametrize("name", RUNNABLE)
def test_notebook_runs_top_to_bottom(name: str, tmp_path: Path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    # A scratch copy, so the data files the notebooks write stay out of the repo.
    work = tmp_path / "examples"
    shutil.copytree(EXAMPLES, work, ignore=shutil.ignore_patterns("large_scale_projects", ".ipynb_checkpoints"))
    monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / ".cash"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Offline: any HTTP(S) request goes to a closed port and fails loudly.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")

    path = work / name
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(work)}},
    )
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as exc:
        pytest.fail(f"{name} failed:\n{str(exc)[:3000]}")

    unexpected = _unexpected_stderr(nb)
    assert not unexpected, f"{name} wrote to stderr:\n  " + "\n  ".join(unexpected)
