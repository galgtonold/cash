"""A loop run as one unit is named by the files its entry was stored with.

A loop that reads files is named by its key and the files it read (see
``test_a_loop_reading_a_file_is_named_by_its_key_and_the_file``). Restored,
it reads nothing, so its name was the key alone: the loop restored before
and after a same-size edit to one of its files got one name for two
different lists, and a name other than the computed loop's, so the cell
after it missed once after every restore. Computed or restored, the loop is
now named by the file state its entry recorded: one name for one state, a
new one for another.
"""

import pytest

from .test_a_loop_filling_lists_is_named_by_its_inputs import SETUP, UNCOUNT

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Installed afresh on every use, over the real function, as COUNT is.
NAMES = (
    "import cash.notebook.control_structures.helpers as h\n"
    "h.update_mutated_variable_lineages = getattr(h, 'real_update', h.update_mutated_variable_lineages)\n"
    "h.real_update = h.update_mutated_variable_lineages\n"
    "h.unit_names = []\n"
    "def naming(shell, sp, names, *a, h=h, **k):\n"
    "    h.unit_names.append(k.get('unit_digest'))\n"
    "    return h.real_update(shell, sp, names, *a, **k)\n"
    "h.update_mutated_variable_lineages = naming\n"
)
UNIT_NAMES = "__import__('cash.notebook.control_structures.helpers').notebook.control_structures.helpers.unit_names"
FILES = "import glob\nfiles = sorted(glob.glob('inputs/*.num'))"
LOOP = "rows = []\nfor f in files:\n    text = open(f).read()\n    value = int(text)\n    rows.append(value)"
REPORT = "print('SUM', sum(rows), len(rows))"


def test_restores_of_two_file_states_get_two_names(nb_runner):
    inputs = nb_runner.work_dir / "inputs"
    inputs.mkdir()
    for i in range(60):  # enough to run as one unit
        (inputs / f"part_{i:03d}.num").write_text(str(i), encoding="utf-8")
    nb_runner.create_notebook([SETUP, FILES, LOOP, REPORT])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({NAMES!r})")
    try:
        nb_runner.run_cells([3, 4])  # computed, state A
        nb_runner.run_cells([3, 4])  # restored, state A
        assert "SUM 1770 60" in nb_runner.get_output(4)

        (inputs / "part_007.num").write_text("9", encoding="utf-8")  # the same size
        nb_runner.run_cells([3, 4])  # computed, state B
        assert "SUM 1772 60" in nb_runner.get_output(4)
        nb_runner.run_cells([3, 4])  # restored, state B
        assert "SUM 1772 60" in nb_runner.get_output(4)

        names = eval(nb_runner.peek(UNIT_NAMES))
        assert len(names) == 4 and all(names), names
        computed_a, restored_a, computed_b, restored_b = names
        assert restored_a == computed_a
        assert restored_b == computed_b
        assert restored_a != restored_b
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")


def _restored_name(nb_runner) -> str:
    """The name the loop takes when it is restored in a fresh kernel."""
    nb_runner.restart()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({NAMES!r})")
    try:
        nb_runner.run_cells([3, 4])
        names = eval(nb_runner.peek(UNIT_NAMES))
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")
    assert len(names) == 1 and names[0], names
    return names[0]


@pytest.mark.fresh_kernel
def test_after_a_restart_restores_of_two_file_states_get_two_names(nb_runner):
    inputs = nb_runner.work_dir / "inputs"
    inputs.mkdir()
    for i in range(60):
        (inputs / f"part_{i:03d}.num").write_text(str(i), encoding="utf-8")
    nb_runner.create_notebook([SETUP, FILES, LOOP, REPORT])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({NAMES!r})")
    try:
        nb_runner.run_cells([3, 4])
        computed_a = eval(nb_runner.peek(UNIT_NAMES))
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")
    restored_a = _restored_name(nb_runner)
    assert [restored_a] == computed_a
    assert "SUM 1770 60" in nb_runner.get_output(4)

    (inputs / "part_007.num").write_text("9", encoding="utf-8")
    nb_runner.run_cells([3, 4])
    assert "SUM 1772 60" in nb_runner.get_output(4)
    restored_b = _restored_name(nb_runner)
    assert "SUM 1772 60" in nb_runner.get_output(4)
    assert restored_a != restored_b
