"""Generate ``examples/try_cash_colab.ipynb`` from ``examples/try_cash_binder.ipynb``.

The Binder and Colab feature tours are identical except for the **setup cell**:
Binder pre-installs cash from ``binder/requirements.txt``, while Colab has no such
file and must ``pip install`` it. To avoid maintaining two copies of the whole
tour (and letting them drift), the Binder notebook is the single source of truth
for the shared content; this script swaps in the Colab setup cell and writes the
Colab notebook.

Workflow: edit ``examples/try_cash_binder.ipynb``, then run::

    python scripts/build_try_cash_colab.py

``tests/test_docs/test_try_cash_notebooks_in_sync.py`` fails if the committed
Colab notebook doesn't match what this script would produce.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDER = ROOT / "examples" / "try_cash_binder.ipynb"
COLAB = ROOT / "examples" / "try_cash_colab.ipynb"

# The one cell that differs. Plain top-level ``%pip`` (no ``if``, no indented
# magic) so cash can dependency-parse the cell on any version.
COLAB_SETUP = [
    "# Install cash from GitHub — Colab has no requirements.txt to preinstall it.\n",
    "# --force-reinstall pulls the current main even in a reused runtime, where\n",
    "# cash's constant version number would make a plain `pip install` a silent\n",
    "# no-op. After the PyPI release this becomes:  %pip install -q \"cash-lib[pandas]\"\n",
    "%pip install -q --force-reinstall --no-deps \"git+https://github.com/galgtonold/cash.git\"\n",
    "\n",
    "import cash\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "%cash_on",
]


def build() -> dict:
    """Return the Colab notebook as a dict (does not write it)."""
    nb = json.loads(BINDER.read_text(encoding="utf-8"))
    setup = nb["cells"][1]
    assert setup["cell_type"] == "code" and "%cash_on" in "".join(setup["source"]), (
        "cell 1 of the Binder notebook is expected to be the setup cell"
    )
    setup["source"] = COLAB_SETUP
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
