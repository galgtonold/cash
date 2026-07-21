"""ADR-017 detection core: find edited-but-not-rerun seed cells (CAS-225).

`seed_cells_not_yet_run` is the pure, cell-granular primitive that spots the
one situation nothing runtime can see: a notebook cell that seeds a drawn RNG
module but whose exact source has not executed this session. These tests pin it
in isolation; the checker integration (warn) and reconstruction replay build on
it separately.
"""
from __future__ import annotations

import hashlib

from cash.notebook.randomness import seed_cells_not_yet_run


def _h(src: str) -> str:
    return hashlib.sha256(src.encode('utf-8')).hexdigest()


NP_SEED0 = "import numpy as np\nnp.random.seed(0)"
NP_SEED1 = "import numpy as np\nnp.random.seed(1)"
NP_DRAW = "x = np.random.rand(100)"
STD_SEED0 = "import random\nrandom.seed(0)"


def test_edited_seed_cell_not_rerun_is_flagged():
    """The core case: seed cell edited to seed(1), only seed(0) ran."""
    cells = [NP_SEED1, NP_DRAW]
    executed = {_h(NP_SEED0)}          # the OLD seed(0) ran; the edited seed(1) did not
    result = seed_cells_not_yet_run({'numpy.random'}, cells, executed)
    assert result == [('numpy.random', 0)]


def test_seed_cell_that_did_run_is_not_flagged():
    """No false positive when the current seed cell source has executed."""
    cells = [NP_SEED1, NP_DRAW]
    executed = {_h(NP_SEED1)}
    assert seed_cells_not_yet_run({'numpy.random'}, cells, executed) == []


def test_no_draw_means_no_check():
    """A cell that doesn't draw triggers nothing, even with an unrun seed present."""
    cells = [NP_SEED1, NP_DRAW]
    assert seed_cells_not_yet_run(set(), cells, set()) == []


def test_seed_for_a_different_module_is_ignored():
    """A stale stdlib seed must not flag a numpy-only draw."""
    cells = [STD_SEED0, NP_DRAW]
    assert seed_cells_not_yet_run({'numpy.random'}, cells, set()) == []


def test_module_must_match_between_seed_and_draw():
    """Both-modules drawn: only the module with an unrun seed is flagged."""
    cells = [NP_SEED1]                       # numpy seed, unrun
    executed: set[str] = set()
    got = seed_cells_not_yet_run({'numpy.random', 'random'}, cells, executed)
    assert got == [('numpy.random', 0)]


def test_multiple_unrun_seed_cells_all_flagged_in_order():
    cells = [NP_SEED1, "import numpy as np\nnp.random.seed(7)", NP_DRAW]  # seeds at 0,1; draw at 2
    got = seed_cells_not_yet_run({'numpy.random'}, cells, set())
    assert got == [('numpy.random', 0), ('numpy.random', 1)]


def test_ran_seed_after_an_unran_one_is_partial():
    """Only the cell whose source didn't run is flagged."""
    ran = "import numpy as np\nnp.random.seed(7)"
    cells = [NP_SEED1, ran, NP_DRAW]
    got = seed_cells_not_yet_run({'numpy.random'}, cells, {_h(ran)})
    assert got == [('numpy.random', 0)]


def test_syntax_error_cell_is_skipped_not_fatal():
    cells = ["def (", NP_DRAW]
    assert seed_cells_not_yet_run({'numpy.random'}, cells, set()) == []


def test_empty_notebook_is_safe():
    assert seed_cells_not_yet_run({'numpy.random'}, [], set()) == []


def test_multi_statement_seed_cell_is_detected():
    """A seed buried in a larger cell still counts (visitor walks the whole tree)."""
    cell = "import numpy as np\nconfig = {'a': 1}\nnp.random.seed(3)\nprint('ready')"
    got = seed_cells_not_yet_run({'numpy.random'}, [cell, NP_DRAW], set())
    assert got == [('numpy.random', 0)]


def test_stdlib_seed_edit_is_flagged():
    cells = ["import random\nrandom.seed(9)", "y = random.random()"]
    executed = {_h(STD_SEED0)}
    assert seed_cells_not_yet_run({'random'}, cells, executed) == [('random', 0)]
