"""Code in a cell writes to the ``sys.stdout`` plain gives it.

cash took a statement's output by putting a ``StringIO`` (or, for a loop, a
writer passing it on) in ``sys.stdout``, so code saw a different stream than
plain: ``type(sys.stdout).__name__`` was ``StringIO`` and ``sys.stdout.name``
raised. The stream now stays in place, its writes taken on the instance.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__"
    "._last_cell_metrics['statements']]"
)

HELPER = (
    "import sys, time\n"
    "def f(n):\n"
    "    time.sleep(0.4)  # slow enough to store\n"
    "    print('in f', type(sys.stdout).__name__, sys.stdout.name)\n"
    "    return n"
)


def test_the_stream_is_the_one_plain_gives(nb_runner):
    nb_runner.create_notebook(
        [
            "%cash_badge off",
            HELPER,
            "print('cell', type(sys.stdout).__name__, sys.stdout.name)",
            "for i in range(3000):\n    if i == 2999:\n        print('loop', type(sys.stdout).__name__, sys.stdout.name)",
            "a = f(1)",
        ]
    )
    nb_runner.start_kernel()
    plain = ast.literal_eval(nb_runner.peek("type(sys.stdout).__name__ + ' ' + sys.stdout.name"))
    assert plain == "OutStream stdout", plain  # ipykernel's, outside any statement

    for run in ("computed", "served"):
        nb_runner.run_all()
        assert nb_runner.get_output(3).strip() == f"cell {plain}"
        assert nb_runner.get_output(4).strip() == f"loop {plain}"
        assert nb_runner.get_output(5).strip() == f"in f {plain}"
        status = ast.literal_eval(nb_runner.peek(_STATUSES))
        assert status == (["COMPUTED"] if run == "computed" else ["RESTORED"]), status
