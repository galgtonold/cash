"""The Binder and Colab feature-tour notebooks must stay in lock-step.

``examples/try_cash_binder.ipynb`` is the single source of truth; the Colab copy
is generated from it by ``scripts/build_try_cash_colab.py`` (only the setup cell
differs). These tests fail if the committed Colab notebook drifts from what the
generator produces, or if the two notebooks diverge anywhere but the setup cell.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BINDER = ROOT / "examples" / "try_cash_binder.ipynb"
COLAB = ROOT / "examples" / "try_cash_colab.ipynb"
GENERATOR = ROOT / "scripts" / "build_try_cash_colab.py"


def _generator():
    spec = importlib.util.spec_from_file_location("build_try_cash_colab", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_colab_notebook_matches_generator():
    expected = _generator().render()
    actual = COLAB.read_text(encoding="utf-8")
    assert actual == expected, (
        "examples/try_cash_colab.ipynb is stale — "
        "run `python scripts/build_try_cash_colab.py` and commit the result."
    )


def test_notebooks_differ_only_in_setup_cell():
    b = json.loads(BINDER.read_text(encoding="utf-8"))
    c = json.loads(COLAB.read_text(encoding="utf-8"))
    assert len(b["cells"]) == len(c["cells"])
    diffs = [i for i in range(len(b["cells"])) if b["cells"][i]["source"] != c["cells"][i]["source"]]
    assert diffs == [1], f"the notebooks should differ only in the setup cell, got {diffs}"


def test_setup_cells_are_environment_appropriate():
    b = json.loads(BINDER.read_text(encoding="utf-8"))
    c = json.loads(COLAB.read_text(encoding="utf-8"))
    binder_setup = "".join(b["cells"][1]["source"])
    colab_setup = "".join(c["cells"][1]["source"])
    assert "%pip" not in binder_setup, "Binder pre-installs cash; the setup cell must not pip-install"
    assert "%pip install" in colab_setup, "Colab has no requirements.txt; it must pip-install cash"
    # No indented magic (would false-trigger CashUpstreamSyntaxWarning); the only
    # magics are top-level.
    for setup in (binder_setup, colab_setup):
        for line in setup.splitlines():
            if line.lstrip().startswith(("%", "!")):
                assert line == line.lstrip(), f"magic must be top-level, not indented: {line!r}"


def test_committed_notebooks_have_no_outputs():
    for f in (BINDER, COLAB):
        nb = json.loads(f.read_text(encoding="utf-8"))
        code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        assert all(not cell.get("outputs") for cell in code), f"{f.name} has execution outputs"
        assert all(cell.get("execution_count") is None for cell in code), f"{f.name} has execution counts"
