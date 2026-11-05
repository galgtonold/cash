"""A repair does not re-run statements below an import that would fail.

Found by replaying a real session (JuNE expert_8). The import cell's
``from parse import parse`` fails (the package is not installed), and a
``from typing import Dict, List`` comes after it. A cell below reading
``Dict`` fails with ``NameError`` in a plain kernel, and a run from the top
stops at the failing import. When cash did not know the import cell had
failed as it now reads (an edit not run yet, or a fresh kernel), its repair
re-ran only ``from typing import Dict, List``, skipping the failing import,
and the cell ran: the error was hidden. The repair is now refused, and the
cell runs as written, with ``NOTEBOOK-REPAIR-GAP``.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.replay import UpstreamRepairGap
from tests._cell_driver import run_cash_cell

IMPORTS = "import json\nfrom nonexistent_pkg_for_cash_tests import parse\nfrom typing import Dict, List"
IMPORTS_EDITED = IMPORTS + ", Tuple"
READER = "data: Dict[str, List[int]] = {}"


def test_a_cell_run_in_a_fresh_kernel_is_not_repaired_past_the_import(cash_magics):
    ns = cash_magics.shell.user_ns
    with pytest.raises(UpstreamRepairGap, match="nonexistent_pkg_for_cash_tests"):
        run_cash_cell(cash_magics, READER, cells=[IMPORTS, READER])
    assert "Dict" not in ns and "data" not in ns


def test_an_edit_not_yet_run_is_not_repaired_past_the_import(cash_magics):
    reader = "data: Dict[str, List[Tuple[int, int]]] = {}"
    ns = cash_magics.shell.user_ns
    with pytest.raises(ImportError):
        run_cash_cell(cash_magics, IMPORTS, cells=[IMPORTS, reader])
    with pytest.raises(UpstreamRepairGap, match="which would fail"):
        run_cash_cell(cash_magics, reader, cells=[IMPORTS_EDITED, reader])
    assert "Tuple" not in ns and "data" not in ns


def test_a_missing_name_taken_from_a_loaded_module_stops_it_too(cash_magics):
    imports = "import json\nfrom typing import Typevar_not_there\nfrom collections import OrderedDict"
    reader = "d = OrderedDict(a=1)"
    with pytest.raises(UpstreamRepairGap, match="Typevar_not_there"):
        run_cash_cell(cash_magics, reader, cells=[imports, reader])


def test_an_import_that_would_succeed_is_repaired_as_before(cash_magics):
    """The positive control: the same cells with an import that can be found
    re-run what the cell needs, and the cell runs."""
    imports = "import json\nimport textwrap\nfrom typing import Dict, List"
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, READER, cells=[imports, READER])
    assert ns["data"] == {}
    assert "Dict" in ns
