"""A file rewritten by a statement is checked again by the statements after it.

The statements of one cell share the answer for a file they depend on
(``CacheFreshnessChecker.forget_file_answers``): one notebook re-checked 10,000
documents per statement lookup. That sharing must end when a statement
changes the file. Here the first and last statements recorded the same state
of ``data.csv``; the statement between them rewrites it through a helper, only
on the second run.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.files]

HELPERS = {
    "open_write": "def maybe_write(flag):\n    if flag:\n        with open('data.csv', 'w') as fh:\n            fh.write('v\\n5\\n')\n    return flag",
    "to_csv": "def maybe_write(flag):\n    if flag:\n        pd.DataFrame({'v': [5]}).to_csv('data.csv', index=False)\n    return flag",
}


@pytest.mark.parametrize("helper", list(HELPERS), ids=list(HELPERS))
def test_the_statement_after_the_write_reads_the_new_file(nb_runner, helper):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import time\nimport pandas as pd\nopen('data.csv', 'w').write('v\\n1\\n')\n" + HELPERS[helper],
            "FLAG = False",
            "time.sleep(0.02)\nbefore = int(pd.read_csv('data.csv')['v'].sum())\n"
            "done = maybe_write(FLAG)\n"
            "after = int(pd.read_csv('data.csv')['v'].sum())\n"
            "print('BEFORE', before, 'AFTER', after)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert "BEFORE 1 AFTER 1" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "FLAG = True")
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    assert "BEFORE 1 AFTER 5" in nb_runner.get_output(4), nb_runner.get_output(4)
