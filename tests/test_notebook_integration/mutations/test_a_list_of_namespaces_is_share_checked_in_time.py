"""A list of many ``SimpleNamespace`` records is share-checked in time.

Its share check walked every record one Python step at a time: a list of
1.5 million records ran past the check's time (twice what the cell took),
then the heap was searched for holders until that time ran out too, and the
cell ended in NOTEBOOK-SHARE-UNCHECKED, 8 s more than the 1.5 s it took to
build them. Each record that only its list holds, told by reference count,
needs no walk, as for records of a class of the notebook's.

numpy is imported first: with it loaded, the walks for array views and
borrowed memory read the same records, and took them a record at a time
past the check's time too.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

N = 300_000


def test_a_list_of_namespaces_ends_without_the_unchecked_warning(nb_runner):
    nb_runner.create_notebook(
        [
            # numpy loaded, as a warm kernel may have it from an earlier
            # test: the walks for views and borrowed memory run only then.
            "import numpy\nimport cash\n%cash_on",
            "import types",
            f"records = [types.SimpleNamespace(i=i, x=i * 0.5) for i in range({N})]",
            "first = records[0]",
            "first.x = 99.0",
            "print('x', records[0].x, len(records))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "NOTEBOOK-SHARE-UNCHECKED" not in nb_runner.get_raw_output(3)
    assert "NOTEBOOK-SHARE-UNCHECKED" not in nb_runner.get_raw_output(4)
    assert f"x 99.0 {N}" in nb_runner.get_output(6)
