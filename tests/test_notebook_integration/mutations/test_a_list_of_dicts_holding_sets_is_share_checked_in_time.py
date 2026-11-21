"""A list of many dicts each holding a small set is share-checked in time.

Its share check walked every dict and every set one Python step at a time:
a list of 1.5 million ``{"i": i, "tags": {i % 7, i % 11}}`` records ran past
the check's time (twice what the cell took), then the heap was searched for
holders until that time ran out too, and the cell ended in
NOTEBOOK-SHARE-UNCHECKED, 11 s more than the 2.4 s it took to build them.
A set of values that only its dict holds needs no walk, as the dict needs
none; one a variable holds too is still found, and a change through it
reaches the list.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

N = 300_000


def test_a_list_of_dicts_holding_sets_ends_without_the_unchecked_warning(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            f"records = [{{'i': i, 'tags': {{i % 7, i % 11}}}} for i in range({N})]",
            "tags = records[3]['tags']",
            "tags.add(-1)",
            "print('tags', sorted(records[3]['tags']), len(records))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "NOTEBOOK-SHARE-UNCHECKED" not in nb_runner.get_raw_output(2)
    assert "NOTEBOOK-SHARE-UNCHECKED" not in nb_runner.get_raw_output(3)
    assert f"tags [-1, 3] {N}" in nb_runner.get_output(5)
