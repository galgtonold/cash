"""After a restart, a cleaning cell rebuilt for a cell below it runs every write it needs.

Round 25's r25s2 built a weekly pack from till exports. After a kernel restart
they ran only the export cell; cash rebuilt the cleaning cell but left out
``sales["refund"] = is_refund.astype(int)``, and the head-office summary showed
0 refunds for the three stores that book refunds as negative quantities (51, 62
and 119 in a plain run). No warning.

The planner gave each scheduled statement a producer of its inputs, and took
ANY earlier scheduled producer as enough: ``sales['timestamp'] = ...`` stood in
for the ``sales['refund'] = ...`` between it and the statement reading ``sales``.
This shape did not reach that state here (the tester's 3-million-row data did);
it guards the rebuild, and ``test_notebook/test_latest_producer_is_scheduled.py``
pins the planner step.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

DATA = """import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 400_000
store = rng.choice(['S1', 'S2', 'S3'], n)
day = rng.integers(0, 14, n)
q = np.where((store == 'S2') & (day % 3 == 0), -2, 3)
flag = np.where((store == 'S3') & (day % 4 == 0), 1, 0)
ts = pd.Timestamp('2026-04-06 10:00') + pd.to_timedelta(day, unit='D') + pd.to_timedelta(np.arange(n) % 3600, unit='s')
pd.DataFrame({'store': store, 'timestamp': ts.astype(str), 'quantity': q, 'price': 2.0,
              'refund': flag}).to_csv('sales.csv', index=False)
pd.DataFrame([{'store': s, 'week': w, 'staffed_hours': 40.0}
              for s in ['S1', 'S2', 'S3'] for w in (1, 2)]).to_csv('staffing.csv', index=False)"""

CELLS = [
    "import cash\n%cash_on",
    DATA,
    "import numpy as np\nimport pandas as pd\nraw = pd.read_csv('sales.csv')",
    "staffing = pd.read_csv('staffing.csv')",
    "sales = raw.drop_duplicates()\n"
    "sales['timestamp'] = pd.to_datetime(sales['timestamp'])\n"
    "is_refund = (sales['refund'] == 1) | (sales['quantity'] < 0)\n"
    "sales['refund'] = is_refund.astype(int)\n"
    "sales['quantity'] = sales['quantity'].abs()\n"
    "sales['net_units'] = np.where(is_refund, -sales['quantity'], sales['quantity'])\n"
    "sales['net_sales'] = sales['net_units'] * sales['price']\n"
    "sales['week'] = (sales['timestamp'] - pd.Timestamp('2026-04-06')).dt.days // 7 + 1",
    "weekly = sales.groupby(['store', 'week'], as_index=False).agg("
    "net_sales=('net_sales', 'sum'), refunds=('refund', 'sum'))",
    "weekly = weekly.merge(staffing, on=['store', 'week'], how='left')\n"
    "weekly['sales_per_hour'] = weekly['net_sales'] / weekly['staffed_hours']",
    "LAST_WEEK = int(staffing.week.max())\n"
    "summary = weekly[weekly.week == LAST_WEEK][['store', 'refunds']]\n"
    "print('REFUNDS', summary.set_index('store')['refunds'].to_dict())",
]


def _refunds(runner):
    return next(line for line in runner.get_output(8).splitlines() if line.startswith("REFUNDS"))


def test_after_a_restart_the_export_cell_sees_the_refund_column(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = _refunds(nb_runner)
    assert "'S2': 0" not in want and "'S3': 0" not in want, want

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(8)

    assert _refunds(nb_runner) == want
