"""What cash costs a process that is not a Jupyter kernel.

cash-bench runs each agent script as one cell in a fresh IPython shell, and
cash's start was about 0.35 s of every such process. Part of it was work for
a notebook that is not there: importing ipykernel (and jupyter_client, zmq)
to look for a kernel connection file, the HTTP client to ask servers about
it, and minifying the HTML badge's stylesheet; and, at the first table
stored, a 16 MB self-test only big tables need.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

from cash.backends import frame_sharing

SCRIPT = """
import io, sys
import pandas  # some pandas versions load urllib.request themselves
from IPython.core.interactiveshell import InteractiveShell
sh = InteractiveShell.instance()
WATCHED = ("ipykernel", "jupyter_client", "urllib.request", "http.client")
before = {m for m in WATCHED if m in sys.modules}  # what the shell itself loaded
who = []


class _Who:  # where a watched module gets imported from, for the failure message
    def find_spec(self, name, path, target=None):
        if name in WATCHED:
            import traceback

            who.append(name + " <- " + " | ".join(f"{f.filename}:{f.lineno}" for f in traceback.extract_stack()[-12:-1]))


sys.meta_path.insert(0, _Who())
out, sys.stdout = sys.stdout, io.StringIO()
sh.run_cell("import cash\\n%cash_on\\n%cash_badge print")
sh.run_cell("import pandas as pd\\ndf = pd.DataFrame({'a': range(1000)})\\nprint(df.a.sum())")
sys.stdout = out
print("\\n".join(who))
print(sorted(m for m in WATCHED if m in sys.modules and m not in before))
"""


def test_a_script_shell_loads_no_kernel_or_http_modules(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=tmp_path, capture_output=True, text=True, timeout=120, check=True
    )
    assert done.stdout.strip().splitlines()[-1] == "[]", done.stdout + done.stderr


def test_a_small_table_is_copied_without_the_self_test(monkeypatch):
    ran = []
    monkeypatch.setattr(frame_sharing, "_CHECKED_OK", [])
    monkeypatch.setattr(frame_sharing, "_checked_self_test", lambda: ran.append(1) or True)
    assert frame_sharing.share_checked(pd.DataFrame({"a": np.arange(1000.0)})) is None
    assert ran == []
    if not frame_sharing.enabled():
        return  # tables are always copied here: there is nothing to ask
    # Control: a table big enough to keep uncopied still asks first.
    big = pd.DataFrame({"a": np.arange(float(frame_sharing.CHECKED_MIN_BYTES // 8 + 1))})
    frame_sharing.share_checked(big)
    assert ran == [1]
