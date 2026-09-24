"""The Binder and Colab feature-tour notebooks must stay in lock-step.

``examples/try_cash_binder.ipynb`` is the single source of truth; the Colab copy
is generated from it by ``scripts/build_try_cash_colab.py``, which adds one
install cell. These tests fail if the committed Colab notebook drifts from what
the generator produces, if the two notebooks diverge anywhere else, or if the
version the tour installs drifts from the version this checkout describes.
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
        "examples/try_cash_colab.ipynb is stale — run `python scripts/build_try_cash_colab.py` and commit the result."
    )


def _sources(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def test_notebooks_differ_only_by_the_install_cell():
    binder, colab = _sources(BINDER), _sources(COLAB)
    assert len(colab) == len(binder) + 1
    assert colab[:1] + colab[2:] == binder, "the notebooks should differ only by Colab's install cell"
    assert "%pip install" in colab[1], "Colab has no requirements.txt; it must pip-install cash"
    assert not any("%pip" in src for src in binder), "Binder pre-installs cash; the tour must not pip-install"


def test_setup_cell_is_import_and_cash_on_alone():
    # Work in the %cash_on cell is never cached, so the setup cell holds nothing else.
    assert _sources(BINDER)[1].split() == ["import", "cash", "%cash_on"]


def test_install_pins_match_the_version():
    pin = _generator().version_pin()
    colab_install = _sources(COLAB)[1]
    assert f'"cash-lib[pandas]{pin}"' in colab_install
    requirements = (ROOT / "binder" / "requirements.txt").read_text(encoding="utf-8")
    pinned = [line for line in requirements.splitlines() if line.startswith("cash-lib")]
    assert pinned == [f"cash-lib[pandas]{pin}"], (
        f"binder/requirements.txt should pin cash-lib[pandas]{pin}, the release this checkout describes"
    )


def test_committed_notebooks_have_no_outputs():
    for f in (BINDER, COLAB):
        nb = json.loads(f.read_text(encoding="utf-8"))
        code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        assert all(not cell.get("outputs") for cell in code), f"{f.name} has execution outputs"
        assert all(cell.get("execution_count") is None for cell in code), f"{f.name} has execution counts"
