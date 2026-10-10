"""A frame with text and object columns passed to your own function is read
from its buffers, and a change the function makes to it is still seen.

``y = f(df)`` fingerprints ``df`` before and after the call, and keys the
call on it. A text column (pandas' ``str`` type, Arrow-backed) was turned
into Python strings and pickled one by one, and so was an object column of
floats or bools: 2.7 s against 0.10 s plain for a 1M-row frame with text
columns. Both are now read from their buffers -- the Arrow buffers of the
text, the typed array of the floats -- and still hold every value: a callee
that writes into either column in place is seen, and after a restart the
frame is the one the callee left.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(300)]

pytest.importorskip("pyarrow")

SETUP = "import cash\n%load_ext cash\n%cash_on"
DEFS = (
    "import time\nimport numpy as np, pandas as pd\n"
    "df = pd.DataFrame({'s': pd.array([f'name{i}' for i in range(20_000)], dtype='string[pyarrow]'),\n"
    "                   'o': pd.Series([float(i) for i in range(20_000)], dtype=object)})\n"
    "def rename_first(d):\n    time.sleep(0.2)\n    d.loc[0, 's'] = 'renamed'\n    return len(d)\n"
    "def bump_first(d):\n    time.sleep(0.2)\n    d.loc[0, 'o'] = 99.5\n    return len(d)\n"
)
# Installed afresh on every use, over the real function, with a fresh count:
# cash's modules outlive a test in a reused kernel.
COUNT = (
    "import cash.content_hashers as ch\n"
    "ch._object_items_bytes = getattr(ch, 'real_items', ch._object_items_bytes)\n"
    "ch.real_items = ch._object_items_bytes\n"
    "ch.pickled_items = []\n"
    "def counting(items, ch=ch):\n"
    "    ch.pickled_items.append(len(items))\n"
    "    return ch.real_items(items)\n"
    "ch._object_items_bytes = counting\n"
)
PICKLED = "sum(__import__('cash.content_hashers').content_hashers.pickled_items)"


def test_text_and_float_columns_are_not_pickled_and_writes_are_seen(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            DEFS,
            "n1 = rename_first(df)",
            "n2 = bump_first(df)",
            "print('FIRST', df.loc[0, 's'], df.loc[0, 'o'], n1, n2)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cells([3, 4, 5])
    assert "FIRST renamed 99.5 20000 20000" in nb_runner.get_output(5)
    assert nb_runner.peek(PICKLED) == "0", "the text or float column was pickled item by item"

    nb_runner.restart()
    nb_runner.run_all()
    assert "FIRST renamed 99.5 20000 20000" in nb_runner.get_output(5), "a restart lost the callee's writes"
