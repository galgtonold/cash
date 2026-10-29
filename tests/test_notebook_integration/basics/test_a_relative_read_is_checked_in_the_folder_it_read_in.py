"""A file a loop or a ``with`` block read by a relative path is checked in
the folder it was read in, not the one the notebook moved to since.

Cell 1 changed into ``dir1`` and read ``a.txt``; cell 2 changed into
``dir2``. The cell below reading the value checked cell 1's files, took
``a.txt`` in ``dir2`` (none, or another file) for a change, and ran cell 1's
read again there: ``FileNotFoundError``, where plain Python prints the value.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(60)]


@pytest.mark.parametrize(
    "read",
    [
        "with open('a.txt') as f:\n    a = f.read()",
        "a = ''\nfor name in ['a.txt']:\n    with open(name) as f:\n        a += f.read()",
    ],
    ids=["a with block", "a loop"],
)
def test_a_folder_changed_below_does_not_re_read_it(nb_runner, tmp_path, read):
    dir1, dir2 = tmp_path / "dir1", tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "a.txt").write_text("from dir1", encoding="utf-8")
    (dir2 / "b.txt").write_text("from dir2", encoding="utf-8")
    one, two = (str(d).replace("\\", "/") for d in (dir1, dir2))
    nb_runner.create_notebook(
        [
            f"import os\nos.chdir('{one}')\n{read}",
            f"os.chdir('{two}')\nwith open('b.txt') as f:\n    b = f.read()",
            "print(f'a={a}, b={b}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "a=from dir1, b=from dir2" in nb_runner.get_output(3)
    nb_runner.run_cell(3)
    assert "a=from dir1, b=from dir2" in nb_runner.get_output(3)
