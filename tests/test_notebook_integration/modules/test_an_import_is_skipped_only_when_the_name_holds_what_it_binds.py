"""An import is skipped as redundant only when the name already holds what it would bind.

Found in the full sweep: `language/test_arrays_and_structs.py` failed with a
TypeError on ``array('i', [...])`` whenever its reused kernel had run
``import array`` for the test before. The badge read ``SKIPPED: from array
import array``: an import-only statement was skipped when every name it binds
was present in the namespace, whatever that name held. The module ``array``
stood in for the class ``array.array``.

The same happens in one notebook: ``import array`` in one cell, ``from array
import array`` in a later one.
"""

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.parametrize(
    "first, second, use",
    [
        ("import array", "from array import array", "print('OUT', array('i', [1, 2]).tolist())"),
        ("from os import path", "import os.path as path", "print('OUT', path.basename('a/b'))"),
        ("import json as j", "import math as j", "print('OUT', j.sqrt(4))"),
    ],
    ids=["module_then_class", "same_module_both_ways", "alias_rebound"],
)
def test_the_second_import_binds_its_own_object(nb_runner, first, second, use):
    nb_runner.create_notebook(["import cash\n%cash_on", first, second, use])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT" in nb_runner.get_output(4), nb_runner.get_output(4)


def test_a_repeated_import_is_still_skipped(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "from array import array",
            "from array import (array)",
            "print('OUT', array('i', [3]).tolist())",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT [3]" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert "SKIPPED" in nb_runner.get_raw_output(3) or "OUT" not in nb_runner.get_output(3)
