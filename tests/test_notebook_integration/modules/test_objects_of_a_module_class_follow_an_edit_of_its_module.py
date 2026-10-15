"""A cell calling into objects of a local module's class follows an edit of
that module, however the objects were made.

``orders = pickle.load(f)`` holds ``shop.Order`` objects and
``t = [o.total() for o in orders]`` runs ``Order.total`` and ``shop.rate()``.
Neither the cell nor the lineage of ``orders`` (the pickle file's) names
``shop``, so after ``rate`` was edited cash served the old ``t``. With
``%autoreload 2`` an object made before the edit also kept the old class
under cash, where a plain kernel updates the class in place.
"""

import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SHOP = (
    "import time\n"
    "def rate():\n"
    "    time.sleep(0.1)\n"
    "    return {rate}\n"
    "class Order:\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def total(self):\n"
    "        return self.v * {factor}\n"
)


def _shop(nb_runner, rate=2, factor="rate()"):
    name = f"shop_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    path.write_text(SHOP.format(rate=rate, factor=factor), encoding="utf-8")
    return name, path


def _edit(path, rate=2, factor="rate()"):
    before = os.stat(path).st_mtime
    path.write_text(SHOP.format(rate=rate, factor=factor), encoding="utf-8")
    os.utime(path, (before + 2, before + 2))


def _pickle_cells(name):
    return [
        f"import pickle, {name}\nwith open('orders.pkl', 'wb') as f:\n"
        f"    pickle.dump([{name}.Order(i) for i in range(3)], f)",
        "with open('orders.pkl', 'rb') as f:\n    orders = pickle.load(f)",
        "t = [o.total() for o in orders]",
    ]


def test_a_reader_of_unpickled_objects_follows_the_edit(nb_runner):
    name, path = _shop(nb_runner)
    nb_runner.create_notebook(_pickle_cells(name))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "[0, 2, 4]"

    _edit(path, rate=5)
    nb_runner.run_cell(3)
    assert nb_runner.peek("t") == "[0, 5, 10]", nb_runner.get_raw_output(3)


@pytest.mark.fresh_kernel
def test_after_a_restart_the_edit_is_not_served_the_old_result(nb_runner):
    """Guard: the restarted kernel computes ``t`` from the edited module and
    stores it under a key of its own, with the module's part in it."""
    name, path = _shop(nb_runner)
    nb_runner.create_notebook(_pickle_cells(name))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "[0, 2, 4]"

    _edit(path, rate=5)
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "[0, 5, 10]", nb_runner.get_raw_output(3)


def test_with_autoreload_objects_made_before_the_edit_run_the_edited_method(nb_runner):
    """Objects a cell above would not rebuild (their lineage is the pickle
    file's): with ``%autoreload 2`` they run the edited method, as in a
    plain kernel, where cash's reload left them the old class."""
    name, path = _shop(nb_runner, factor=2)
    nb_runner.create_notebook(
        [
            f"%load_ext autoreload\n%autoreload 2\nimport pickle, {name}",
            f"with open('orders.pkl', 'wb') as f:\n    pickle.dump([{name}.Order(i) for i in range(3)], f)",
            "with open('orders.pkl', 'rb') as f:\n    orders = pickle.load(f)",
            "t = [o.total() for o in orders]",
            f"same = type(orders[0]) is {name}.Order",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "[0, 2, 4]"

    _edit(path, factor=3)
    nb_runner.run_cell(4)
    nb_runner.run_cell(5)
    assert nb_runner.peek("t") == "[0, 3, 6]", nb_runner.get_raw_output(4)
    assert nb_runner.peek("same") == "True"
