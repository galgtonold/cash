"""A helper folder added with ``sys.path.insert`` is found after a restart.

Reproduced 2/2: new day, fresh kernel, ``%cash_on``, then
straight to a cell far down -- which the quickstart says works. The repair
re-ran ``import bt`` but not the ``sys.path.insert(0, lib)`` right above it,
and stopped with ``UpstreamStateError: ... No module named 'bt'``. The insert
is a setting of ``sys``, replayed for a statement that reads ``sys``; an
import does not read it, but finding a module that is not loaded yet depends
on it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

MODULE = "def slow(x):\n    _ = sum(i * i for i in range(2_000_000))\n    return x * 3\n"


def test_jumping_below_the_import_after_a_restart(nb_runner, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "btsyspath.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            f"import sys\nsys.path.insert(0, {str(lib)!r})\nimport btsyspath\nX = 5",
            "y = btsyspath.slow(X)",
            "print('Y', y)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Y 15" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "UpstreamStateError" not in raw and "No module named" not in raw, raw
    assert "Y 15" in nb_runner.get_output(4), raw
