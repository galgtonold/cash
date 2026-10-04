"""What a test imports after a restart is gone when the next test starts.

A warm kernel is cleaned between tests by purging every module, and every
``sys.path`` entry, that was not there when it was clean. That baseline is kept
in the kernel process, so a test that restarts the kernel throws it away, and
the next test took its snapshot from a kernel that had already imported the
restarted test's modules. Those were then never purged: a later test that did
``import helpers`` got an earlier test's ``helpers``.

The second ``start_kernel()`` is what the next test on the worker does.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_a_module_imported_after_a_restart_is_purged_for_the_next_test(nb_runner):
    lib = nb_runner.work_dir / "lib"
    lib.mkdir()
    (lib / "left_by_a_restart.py").write_text("VALUE = 7\n", encoding="utf-8")
    nb_runner.create_notebook(
        [
            f"import sys\nsys.path.insert(0, {str(lib)!r})\n"
            "import left_by_a_restart\nprint('VALUE', left_by_a_restart.VALUE)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.restart()
    nb_runner.run_all()
    assert "VALUE 7" in nb_runner.get_output(1), nb_runner.get_raw_output(1)

    nb_runner.start_kernel()  # what the next test's start does

    assert nb_runner.peek("'left_by_a_restart' in __import__('sys').modules") == "False"
    assert nb_runner.peek(f"{str(lib)!r} in __import__('sys').path") == "False"
