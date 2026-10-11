"""A change made through one name reaches the cells that read another name for
the same object.

``data = raw`` then ``raw.append(...)``, ``config = {'features': features}``
then ``features.append(...)``: two names, one object. A cell reading the
other name must not be served from the cache after the change is edited and
the notebook runs again, and a first Run All must not rebuild a value and lose
what was changed through its other name. Each notebook is compared with what
plain Python prints.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SLOW = "import time\ndef slow(v):\n    time.sleep(0.2)\n    return repr(v)"

#: name -> (cells, cell to edit, its new source, what the last cell prints
#: before, and after the edit and a second Run All)
EDITED = {
    "an alias changed in a later cell": (
        [SLOW, "raw = [1, 2, 3]", "data = raw", "raw.append(4)", "r = slow(data)\nprint(r)"],
        4,
        "raw.append(5)",
        "[1, 2, 3, 4]",
        "[1, 2, 3, 5]",
    ),
    "an alias changed in a loop": (
        [SLOW, "raw = [1]\nscale = 1", "data = raw", "for i in range(2):\n    raw.append(i * scale)", "r = slow(data)\nprint(r)"],
        2,
        "raw = [1]\nscale = 2",
        "[1, 0, 1]",
        "[1, 0, 2]",
    ),
    "a list a dict holds, changed through its own name": (
        [
            SLOW,
            "features = ['age']\nconfig = {'features': features}\nfeatures.append('zip')",
            "r = slow(config)\nprint(r)",
        ],
        2,
        "features = ['age']\nconfig = {'features': features}\nfeatures.append('city')",
        "{'features': ['age', 'zip']}",
        "{'features': ['age', 'city']}",
    ),
    "a list changed through the dict that holds it": (
        [
            SLOW,
            "features = ['age']\nconfig = {'features': features}\nconfig['features'].append('zip')",
            "r = slow(features)\nprint(r)",
        ],
        2,
        "features = ['age']\nconfig = {'features': features}\nconfig['features'].append('city')",
        "['age', 'zip']",
        "['age', 'city']",
    ),
    "a list an attribute holds": (
        [SLOW, "class B: pass\nb = B()\nx = [1]", "b.ref = x", "b.ref.append(99)", "r = slow(x)\nprint(r)"],
        4,
        "b.ref.append(98)",
        "[1, 99]",
        "[1, 98]",
    ),
    "a list a tuple holds": (
        [SLOW, "lst = [1]", "t = (lst,)", "t[0].append(3)", "r = slow(lst)\nprint(r)"],
        4,
        "t[0].append(4)",
        "[1, 3]",
        "[1, 4]",
    ),
}


@pytest.mark.parametrize("name", list(EDITED))
def test_an_edited_change_reaches_the_other_name(nb_runner, name):
    cells, edit_at, edited, before, after = EDITED[name]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(len(cells)).strip() == before
    nb_runner.set_cell_source(edit_at, edited)
    nb_runner.run_all()
    assert nb_runner.get_output(len(cells)).strip() == after


#: name -> (cells, what the last cell prints)
FIRST_RUN = {
    "an alias change kept when the list changes again": (
        ["history = []\nlog = history", "log.append('start')", "history.append('step 1')", "print(history, log is history)"],
        "['start', 'step 1'] True",
    ),
    "an alias change kept when a loop changes the list": (
        [
            "history = []\nlog = history",
            "log.append('start')",
            "for i in range(2):\n    history.append(i)",
            "print(history, log is history)",
        ],
        "['start', 0, 1] True",
    ),
    "a list taken out of a dict": (
        [
            "data = {'train': [1, 2], 'test': [3]}",
            "train = data['train']\ntrain.append(10)",
            "data['test'].append(99)",
            "print(data, train is data['train'])",
        ],
        "{'train': [1, 2, 10], 'test': [3, 99]} True",
    ),
}


@pytest.mark.parametrize("name", list(FIRST_RUN))
def test_a_run_all_keeps_a_change_made_through_the_other_name(nb_runner, name):
    cells, expected = FIRST_RUN[name]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(len(cells)).strip() == expected, "first Run All"
    nb_runner.run_all()
    assert nb_runner.get_output(len(cells)).strip() == expected, "second Run All"
