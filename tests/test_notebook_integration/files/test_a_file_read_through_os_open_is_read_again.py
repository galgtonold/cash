"""A cell whose helper reads a file through ``os.open`` reads it again after
the file changes.

``os.open`` raises the ``open`` audit event with flags in place of a mode,
and the native file hook leaves the interpreter's own calls to that event,
so the read was seen by neither: a rerun after the file changed showed the
old bytes, with no warning.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.files]

HELPER = (
    "import os, time\n"
    "def read_raw(p):\n"
    "    fd = os.open(p, os.O_RDONLY)\n"
    "    data = os.read(fd, 100)\n"
    "    os.close(fd)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return data"
)


def test_a_rerun_after_an_edit_shows_the_new_bytes(nb_runner):
    data = nb_runner.work_dir / "data.bin"
    data.write_bytes(b"one")
    nb_runner.create_notebook(["import cash\n%cash_on", HELPER, "v = read_raw('data.bin')\nprint('V', v)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "V b'one'" in nb_runner.get_output(3), nb_runner.get_output(3)

    data.write_bytes(b"twotwo")
    nb_runner.run_cell(3)
    assert "V b'twotwo'" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert nb_runner.peek("v") == "b'twotwo'"
