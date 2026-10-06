"""A cell magic's body is not run as notebook Python by the upstream check.

``%%script false`` is how a cell is disabled and ``%%writefile`` how a helper
module is written. cash read their bodies as Python and, on a plain Run All,
re-ran ``threshold = 0.9`` and ``RATE = 99`` in the kernel. ``files = !echo``
and ``%%bash`` cells were reported as syntax errors.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_run_all_matches_a_plain_kernel(nb_runner):
    nb_runner.create_notebook(
        [
            "threshold = 0.5\nRATE = 1",
            "%%script false --no-raise-error\nthreshold = 0.9",
            "%%writefile cfg.py\nRATE = 99",
            "files = !echo hi",
            "%%bash\necho hi",
            "y = (threshold, RATE * 2, len(files))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("y") == "(0.5, 2, 1)"
    assert "syntax error" not in nb_runner.get_output(6, filter_debug=False)
