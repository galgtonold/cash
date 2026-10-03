"""A repair rebuilds an intermediate the notebook deletes; it never reuses a stale one.

A cell builds a temporary frame, writes a column into it, aggregates it and
deletes it::

    m = clean.merge(network, on='id')
    m['kw'] = m['kwh'] * 4
    feeder = m.groupby('feeder')['kw'].sum()
    del m

An upstream repair re-runs the statements a result needs, but a ``del`` is not
one of them, so after a repair from below the old ``m`` stays in the kernel.
After the next edit to ``clean``, the repair re-ran ``m['kw'] = ...`` on that
old ``m`` and left the merge above it out: ``feeder`` was aggregated from the
cleaning the user had just replaced, under a badge saying it was refreshed.

The name collision is part of the shape: the cleaning cell binds ``d`` and a
cell below rebinds it, which is what makes the repair treat the whole chain as
depending on an edit. The test checks the value a from-the-top run gives.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

DATA = (
    "import pandas as pd\n"
    "raw = pd.DataFrame({'id': [1, 2, 3], 'kwh': [1.0, 2.0, 4.0]})\n"
    "network = pd.DataFrame({'id': [1, 2, 3], 'feeder': ['A', 'A', 'B']})"
)


def cleaning(scale: int) -> str:
    return f"d = raw['kwh'] * 0\nclean = raw.assign(kwh=raw['kwh'] * {scale} + d)"


AGGREGATE = "m = clean.merge(network, on='id')\nm['kw'] = m['kwh'] * 4\nfeeder = m.groupby('feeder')['kw'].sum()\ndel m"
PEAK = "d = feeder.max()"
REPORT = "print('peak', d)"


def test_an_intermediate_deleted_below_is_rebuilt_after_a_second_edit(nb_runner):
    nb_runner.create_notebook([DATA, cleaning(1), AGGREGATE, PEAK, REPORT])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "peak 16.0" in nb_runner.get_output(5), nb_runner.get_output(5)

    # First edit of the cleaning; the report repairs the chain below it, which
    # leaves the deleted `m` behind in the kernel.
    nb_runner.set_cell_source(2, cleaning(10)).run_cell(2)
    nb_runner.run_cell(5)
    assert "peak 160.0" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Second edit; the report must rebuild `m` from the new `clean`.
    nb_runner.set_cell_source(2, cleaning(100)).run_cell(2)
    nb_runner.run_cell(5)
    assert "peak 1600.0" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert nb_runner.peek("float(d)") == "1600.0"
    assert nb_runner.peek("feeder.to_dict()") == "{'A': 1200.0, 'B': 1600.0}"
