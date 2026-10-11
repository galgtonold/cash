"""A list held through a ``__slots__`` object, a ``UserDict``, an object
array or a pandas ``attrs``, and handed back by a function of the user's
module, is the very list again on every Run All and after a restart.

The share check could not look inside those holders, so ``ds =
mylib.Slotted(rows)`` was stored and restored holding a copy, and a later
``rows.append(3)`` was not seen through ``ds`` or through what
``mylib.rows_of(ds)`` returned: ``t is rows`` was False and its length stale,
where a plain kernel gives True. The check looks inside them now; a pandas
object, which copies its ``attrs`` apart on every copy, is not stored when its
``attrs`` hold an object something else holds too.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

MYLIB = """import time
class Dataset:
    def __init__(self, rows):
        self.rows = rows
class Slotted:
    __slots__ = ("rows",)
    def __init__(self, rows):
        self.rows = rows
def rows_of(ds):
    time.sleep(0.15)
    return ds.rows
def item(c, k):
    time.sleep(0.15)
    return c[k]
"""

CELLS = [
    "import mylib, collections, numpy as np, pandas as pd\n"
    "r0 = [1, 2]; r1 = [1, 2]; r2 = [1, 2]; r3 = [1, 2]; r4 = [1, 2]; r5 = [1, 2]",
    "t0 = mylib.rows_of(mylib.Dataset(r0))",
    "t1 = mylib.item({'train': r1}, 'train')",
    "ds = mylib.Slotted(r2)\nt2 = mylib.rows_of(ds)",
    "ud = collections.UserDict(train=r3)\nt3 = mylib.item(ud, 'train')",
    "arr = np.empty(1, dtype=object); arr[0] = r4\nt4 = mylib.item(arr, 0)",
    "df = pd.DataFrame({'a': [1]}); df.attrs['rows'] = r5\nt5 = mylib.item(df.attrs, 'rows')",
    "for r in (r0, r1, r2, r3, r4, r5): r.append(3)",
    "T = (t0, t1, t2, t3, t4, t5); R = (r0, r1, r2, r3, r4, r5)\n"
    "print([len(t) for t in T], [t is r for t, r in zip(T, R)])",
]
WANT = "[3, 3, 3, 3, 3, 3] [True, True, True, True, True, True]"


def test_every_holder_keeps_the_very_list(nb_runner):
    (nb_runner.work_dir / "mylib.py").write_text(MYLIB, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(len(CELLS)).strip() == WANT, "positive control: the first run"
    nb_runner.run_all()
    assert nb_runner.get_output(len(CELLS)).strip() == WANT, "the second Run All restored a copy"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(len(CELLS)).strip() == WANT, "Run All after a restart restored a copy"
