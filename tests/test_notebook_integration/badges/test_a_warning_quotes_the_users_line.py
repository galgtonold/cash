"""A warning raised from a cached call quotes the line the user wrote.

A pandas warning quoted ``__cash_call__(fit_series, 0)(g,
...)`` as the offending source line. Python prints that line from the source
cash registers for the statement it compiled, and with calls routed through
the cache that was the rewritten statement.
"""

import pytest

pytestmark = [pytest.mark.integration]


def test_the_warning_line_is_the_users(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import time, warnings\n"
            "def fit(x):\n"
            "    warnings.warn('careful with x', UserWarning, stacklevel=2)\n"
            "    time.sleep(0.05)\n"
            "    return x * 2",
            "y = fit(21)\nprint('Y', y)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_raw_output(3)
    assert "careful with x" in out, out
    assert "__cash_call__" not in out, out
    assert "y = fit(21)" in out, out
