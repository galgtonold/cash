"""Generate ``examples/try_cash_colab.ipynb`` from ``examples/try_cash_binder.ipynb``.

The Binder and Colab feature tours are identical except for one thing: Binder
pre-installs cash from ``binder/requirements.txt``, while Colab has no such file
and must ``pip install`` it. So the Binder notebook is the single source of
truth, and this script writes the Colab copy with one extra cell, the install
cell, just before the setup cell (``import cash`` / ``%cash_on``). The install
pins cash to the release this checkout describes, so a tour opened from GitHub
runs against the version it was written for.

Workflow: edit ``examples/try_cash_binder.ipynb``, then run::

    python scripts/build_try_cash_colab.py

``tests/docs/test_try_cash_notebooks_in_sync.py`` fails if the committed
Colab notebook doesn't match what this script would produce.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDER = ROOT / "examples" / "try_cash_binder.ipynb"
COLAB = ROOT / "examples" / "try_cash_colab.ipynb"


def _version() -> str:
    """``__version__`` from ``src/cash/__init__.py``, the single source of truth."""
    text = (ROOT / "src" / "cash" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find __version__ in src/cash/__init__.py")
    return match.group(1)


def version_pin() -> str:
    """The compatible-release pin for this version, e.g. ``~=0.11.0``."""
    major, minor, *_ = _version().split(".")
    return f"~={major}.{minor}.0"


def install_cell() -> dict:
    """The Colab-only cell. A plain top-level ``%pip`` line, in its own cell, so
    the setup cell after it stays exactly ``import cash`` / ``%cash_on``."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-install",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Colab does not ship cash: install it (a no-op when it is already there).\n",
            f'%pip install -q "cash-lib{version_pin()}"',
        ],
    }


def build() -> dict:
    """Return the Colab notebook as a dict (does not write it)."""
    nb = json.loads(BINDER.read_text(encoding="utf-8"))
    setup = nb["cells"][1]
    assert setup["cell_type"] == "code" and "%cash_on" in "".join(setup["source"]), (
        "cell 1 of the Binder notebook is expected to be the setup cell"
    )
    nb["cells"].insert(1, install_cell())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    COLAB.write_text(render(), encoding="utf-8")
    print(f"wrote {COLAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
