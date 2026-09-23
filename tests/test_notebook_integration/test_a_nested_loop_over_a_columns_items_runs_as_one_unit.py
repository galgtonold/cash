"""A long, cheap inner loop over ``frame["col"].items()`` runs as one unit.

Building a gene_id -> symbol dict took 243 s under cash and
3.8 s without it (64x), on every cold run:

    for r, a in raw.items():
        for gid, s in a.var["symbol"].items():
            sym.setdefault(gid, s)

200,000 statements went through the per-statement machinery. A long loop of
cheap statements is meant to run as one unit, and the switch depends on its
iteration count -- which was only read for ``.items()`` on a plain name, not
on ``a.var["symbol"]``, so it read as unknown, and an unknown count never
switches.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport pandas as pd, time",
    "class Run:\n    def __init__(self, n, k):\n"
    "        self.var = pd.DataFrame({'symbol': [f's{i}' for i in range(n)]},\n"
    "                                index=[f'g{k}_{i}' for i in range(n)])\n"
    "raw = {k: Run(8000, k) for k in range(3)}",
    "t0 = time.perf_counter()\nsym = {}\n"
    "for r, a in raw.items():\n"
    "    for gid, s in a.var['symbol'].items():\n"
    "        sym.setdefault(gid, s)\n"
    "print('SYM', len(sym), 'TOOK', round(time.perf_counter() - t0, 2))",
]


def test_the_inner_loop_does_not_pay_per_statement(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "SYM 24000" in out, nb_runner.get_raw_output(3)
    took = float(out.split("TOOK")[1].split()[0])
    # plain Python: ~0.01 s; per-statement machinery: ~1 ms x 24,000
    assert took < 5.0, f"the 24,000-iteration inner loop took {took}s:\n" + nb_runner.get_raw_output(3)
