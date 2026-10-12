"""What a cell showed is not handed to the next statement computing it when
computing it again would give something else, in a kernel.

``g(df)`` shown, then ``kept = g(df)``, where ``g`` draws from a generator a
global holds: ``kept`` got the shown draw again and the generator was not
moved on, so every later draw differed from plain Python. And
``every_other(_arr)`` shown, then ``v = every_other(_arr); v[0] = 99``: ``v``
was handed a copy, because every ``_`` name was taken for IPython's output
history, so the write did not reach ``_arr``.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"


@pytest.mark.parametrize(
    ("made", "draw"),
    [
        ("np.random.default_rng(0)", "float(src.random())"),
        ("random.Random(0)", "src.random()"),
        ("itertools.count(1)", "next(src)"),
    ],
    ids=["numpy-generator", "random-Random", "itertools-count"],
)
def test_the_next_call_draws_again(nb_runner, made, draw):
    setup = (
        "import time, random, itertools\n"
        "import numpy as np, pandas as pd\n"
        "df = pd.DataFrame({'a': np.arange(5.0)})\n"
        f"src = {made}\n"
        "def g(d):\n"
        "    time.sleep(0.6)\n"
        f"    return d * {draw}"
    )
    plain = f"ref = {made}\nfirst, second, third = [(lambda src: {draw})(ref) for _ in range(3)]"
    nb_runner.create_notebook([ON, setup, "g(df)", "kept = g(df)", f"after = {draw}", plain])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("kept.a.tolist() == (df.a * second).tolist()") == "True"
    assert nb_runner.peek("after == third") == "True"


def test_a_view_of_an_underscore_named_variable_still_writes_through(nb_runner):
    setup = "import time\nimport numpy as np\n_arr = np.arange(10.0)\ndef every_other(a):\n    time.sleep(0.6)\n    return a[::2]"
    nb_runner.create_notebook([ON, setup, "every_other(_arr)", "v = every_other(_arr)\nv[0] = 99.0"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("_arr[:3].tolist()") == "[99.0, 1.0, 2.0]"
