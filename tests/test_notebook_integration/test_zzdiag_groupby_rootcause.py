"""Root-cause discriminator for the CAS-86 groupby channel.

Same structure as test_groupby_across_cells_tail_mutation_matches_plain but
the mutated row sits INSIDE compute_hash's head-5 sample window. Outcomes:

- in-sample PASSES while out-of-sample fails  -> sampling is the broken link
  (CAS-86 attribution correct).
- in-sample ALSO FAILS -> the df->g derived-object edge is missing and
  sampling is irrelevant (channel belongs to the CAS-89 aliasing family).
"""

import pytest

pytestmark = [pytest.mark.timeout(120)]


def _run_variant(nb_runner, row):
    cells = [
        "import pandas as pd\ndf = pd.DataFrame({'k': [0, 1] * 500, 'v': list(range(1000))})",
        "g = df.groupby('k')",
        f"df.iloc[{row}, 1] = 0",
        "print('sums=', g['v'].sum().to_dict())",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    nb_runner.set_cell_source(3, f"df.iloc[{row}, 1] = 5")
    nb_runner.run_all()
    truth_edit = nb_runner.get_output(4).strip()
    nb_runner.shutdown()

    nb_runner.set_cell_source(3, f"df.iloc[{row}, 1] = 0")
    nb_runner.start_kernel(with_cash=True)
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, f"df.iloc[{row}, 1] = 5")
    nb_runner.run_all()
    got_edit = nb_runner.get_output(4).strip()
    return truth_edit, got_edit


def test_mutation_inside_sample_window(nb_runner):
    truth, got = _run_variant(nb_runner, row=2)  # row 2 is within head(5)
    assert truth in got, (
        f"IN-SAMPLE mutation ALSO stale -> df->g derived-object edge missing "
        f"(aliasing, not sampling). plain={truth!r} cash={got!r}"
    )


def test_mutation_outside_sample_window(nb_runner):
    truth, got = _run_variant(nb_runner, row=999)  # reproduces the CAS-86 repro
    assert truth in got, (
        f"out-of-sample mutation stale (expected, matches CAS-86 repro). "
        f"plain={truth!r} cash={got!r}"
    )
