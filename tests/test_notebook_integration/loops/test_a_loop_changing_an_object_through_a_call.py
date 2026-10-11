"""A loop that changes an object through a method or a function.

``for epoch in range(6): tr.step(0.5)``, ``for v in range(N): st.push(v)``
and ``for i in range(N): tag(df, i)`` change an object an earlier cell made,
and bind nothing that holds it. A restored loop would bring back only the
loop variable, and the object would keep the state it was made with. On a
second run of the notebook, after an edit of the loop and after a restart,
the object holds what plain Python gives.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

TRAINER = (
    "import time\nclass Trainer:\n    def __init__(self):\n        self.w = 0.0; self.history = []\n"
    "    def step(self, lr):\n        time.sleep(0.1)\n        self.w += lr\n        self.history.append(self.w)\n"
    "tr = Trainer()"
)


@pytest.mark.fresh_kernel
def test_a_training_loop(nb_runner):
    nb_runner.create_notebook([TRAINER, "for epoch in range(6):\n    tr.step(0.5)", "print(tr.w, len(tr.history))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3.0 6"
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3.0 6"
    nb_runner.set_cell_source(2, "for epoch in range(8):\n    tr.step(0.5)")
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "4.0 8"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "4.0 8"


CASES = {
    "a method": (
        "class Stats:\n    def __init__(self):\n        self.n = 0\n    def push(self, v):\n        self.n += 1\nst = Stats()",
        "for v in range(300000):\n    st.push(v)",
        "print(st.n)",
        "300000",
    ),
    "a function on a dict": (
        "st = {'n': 0}\ndef push(st, v):\n    st['n'] += 1",
        "for v in range(300000):\n    push(st, v)",
        "print(st['n'])",
        "300000",
    ),
    "a function on a frame": (
        "import pandas as pd\ndf = pd.DataFrame({'a': range(2000)})\ndef tag(d, i):\n    d.loc[i, 'flag'] = i % 3",
        "for i in range(2000):\n    tag(df, i)",
        "print(float(df['flag'].sum()))",
        "1999.0",
    ),
}


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", list(CASES))
def test_a_loop_run_as_one_unit(nb_runner, case):
    make, loop, read, expected = CASES[case]
    nb_runner.create_notebook([make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == expected
