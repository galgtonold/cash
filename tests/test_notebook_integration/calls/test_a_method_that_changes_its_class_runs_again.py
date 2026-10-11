"""A method or ``__call__`` that changes a class attribute through
``type(self)`` (``type(self).fitted += 1``, a common fit or call counter)
runs again on every Run All, called by name or handed to ``s.apply``, as in
a plain kernel. It was served from the cache and the class counter stayed 0.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ROWS = 20
CELLS = [
    f"import time, pandas as pd, numpy as np\ns = pd.Series(np.arange({ROWS}))",
    "class Model:\n    fitted = 0\n    def fit(self, v):\n        time.sleep(0.3); type(self).fitted += 1; return v * 2\nm = Model()",
    "y1 = m.fit(3)",
    "class Counted:\n    n = 0\n    def __call__(self, v):\n        time.sleep(0.015); type(self).n += 1; return v * 2\n"
    "cnt = Counted()",
    "r2 = s.apply(cnt)",
    "class Noted:\n    seen = []\n    def note(self, v):\n        time.sleep(0.3); self.__class__.seen.append(v); return v\n"
    "Noted.seen = []\nnt = Noted()",
    "y3 = nt.note(5)",
    "print(Model.fitted, Counted.n, len(Noted.seen))",
]
STATE = "(Model.fitted, Counted.n, len(Noted.seen))"
WANT = repr((1, ROWS, 1))


def test_a_class_counter_counts_again_on_every_run_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
