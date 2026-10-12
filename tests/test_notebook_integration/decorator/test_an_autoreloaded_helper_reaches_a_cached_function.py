"""``%autoreload 2`` in a kernel that uses only ``@cash.cache``: an edit of a
helper the cached function calls is run, not served from the old entry.

Autoreload patches the old function object in place (``old.__code__ =
new.__code__``), so the helper is the same object before and after the
edit; the cached function must still be keyed by the code it runs now.
"""

import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

V1 = "import cash\n\ndef helper(v):\n    return v * 10\n\n@cash.cache\ndef run(x):\n    return helper(x)\n"


def test_the_cached_function_runs_the_edited_helper(nb_runner):
    name = f"arhelp_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    path.write_text(V1, encoding="utf-8", newline="\n")
    nb_runner.create_notebook(
        [
            "%load_ext autoreload\n%autoreload 2",
            f"import {name} as mylib",
            "print(mylib.run(3), mylib.run.__wrapped__(3))",
        ]
    )
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    assert nb_runner.get_output(3).split() == ["30", "30"]

    before = os.stat(path).st_mtime
    path.write_text(V1.replace("v * 10", "v * 20"), encoding="utf-8", newline="\n")
    os.utime(path, (before + 2, before + 2))
    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).split() == ["60", "60"], nb_runner.get_raw_output(3)

    nb_runner.run_all()
    assert nb_runner.get_output(3).split() == ["60", "60"], nb_runner.get_raw_output(3)
