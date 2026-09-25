"""A notebook restores from the cache ``cash.configure`` moved it to.

``cash.configure(cache_dir=...)`` after ``%cash_on`` builds a new backend. The
statements were stored there, but looked up in the one ``%cash_on`` built, so a
rerun executed every statement again until the kernel restarted.
"""

import pytest

from tests._nbharness.badge import shows_cached, shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def test_a_rerun_restores_after_configure(nb_runner, tmp_path):
    moved = tmp_path / "moved"
    nb_runner.create_notebook(
        [
            f"import cash\n%cash_on\n%cash_badge print\ncash.configure(cache_dir=r'{moved}')",
            "import time",
            "x = (time.sleep(0.3), 7)[1]",
            "# @cash:no-cache\nprint('X', x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 7" in nb_runner.get_output(4)
    assert moved.is_dir(), "the setting did not move the cache"

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert shows_cached(out) and not shows_executed(out), out
