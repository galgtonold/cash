"""A frame changed through a column view pandas hands the user's callback
is not served from the cache as if it were unchanged.

``df.apply(bump, raw=True)`` gives ``bump`` read-only views of the columns
pandas took itself, and ``col.flags.writeable = True`` (the usual answer to
"assignment destination is read-only") writes straight into the frame.
Only views the user's own code asked for were recorded, so the frame memo
trusted the frame: ``n = prep(df)`` was stored without ``df``, and the
second Run All served it from the cache and never changed ``df``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = (
    "import cash\n%cash_on\nimport pandas as pd, numpy as np, time\n"
    "def bump(col):\n    col.flags.writeable = True\n    col += 1.0\n    return 0\n"
    "def prep(df):\n    time.sleep(0.5)\n    df.apply(bump, raw=True)\n    return len(df)"
)
MAKE = "df = pd.DataFrame({'a': np.arange(200_000, dtype=float), 'b': np.ones(200_000)})"
USE = "print('R', df.sum().sum())"
EXPECT = "R 20000500000.0"


def test_after_a_restart_the_frame_is_changed_again(nb_runner):
    # Restarted, the frame is built afresh (too quick to be kept on disk) and
    # the call that changed it last time must change it again, not be served.
    # (Within one kernel the second Run All gets the RAM tier's frame, which
    # is read-only underneath: making a view of it writable raises there.)
    nb_runner.create_notebook([SETUP, MAKE, "n = prep(df)", USE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), nb_runner.get_raw_output(3) + nb_runner.get_raw_output(4)
    nb_runner.restart()
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), (
        "after a restart the change made through the view was skipped:\n"
        + nb_runner.get_raw_output(3)
        + nb_runner.get_raw_output(4)
    )
