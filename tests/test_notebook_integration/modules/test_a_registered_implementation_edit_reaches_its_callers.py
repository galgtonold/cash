"""Editing a ``singledispatch`` implementation in a local module reaches the cells that use it.

``@fmt.register def _(x: int)`` in ``helper.py`` is part of what ``fmt``
does, but the module's per-name keying saw only the definition of ``fmt``:
after an edit to the implementation, the cell calling ``helper.fmt`` through
a notebook function kept the old result.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

DISPATCH = (
    "from functools import singledispatch\n\n"
    "@singledispatch\ndef fmt(x):\n    return 'obj'\n\n"
    "@fmt.register\ndef _(x: int):\n    return x + {n}\n"
)


def test_the_cell_calling_the_dispatcher_recomputes(nb_runner, tmp_path):
    mod = tmp_path / "dispatchhelper.py"
    mod.write_text(DISPATCH.format(n=1), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import dispatchhelper, time\ndef compute(x):\n    time.sleep(0.3)\n    return dispatchhelper.fmt(x)",
            "r = compute(3)\nprint('R', r)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 4" in nb_runner.get_output(2)

    mod.write_text(DISPATCH.format(n=100), encoding="utf-8")
    nb_runner.run_cell(2)

    assert nb_runner.peek("r") == "103", nb_runner.get_raw_output(2)
