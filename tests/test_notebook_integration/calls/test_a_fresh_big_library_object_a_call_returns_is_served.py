"""A call returning a fresh library object holding many parts
(more than _INSIDE_MAX_OBJECTS) -- is it still served on later runs?"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.parametrize("n", [20_000, 60_000])
def test_fresh_big_namespace_is_served(nb_runner, n):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nimport time, argparse",
            f"def mk():\n    time.sleep(2.0)\n    return argparse.Namespace(rows=[{{'i': i}} for i in range({n})])",
            "t = time.time()\nw = mk()\nprint('T', round(time.time() - t, 1), len(w.rows))",
        ]
    )
    nb_runner.start_kernel()
    seen = []
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append(nb_runner.get_raw_output(3).strip())
    print(n, "runs:", seen)
    assert all(float(s.split()[1]) < 1.5 for s in seen[1:]), seen
