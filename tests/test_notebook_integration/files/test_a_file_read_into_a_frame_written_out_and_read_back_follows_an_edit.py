"""An edit of a file read into a frame reaches the frame read back from its CSV.

The sentiment notebook (E10) reads review files into a frame in a loop,
writes the frame to ``movie_data.csv`` and reads it back as ``df``. After
one review is edited, a restart and Run All gives a ``df`` with the edit:
the unpacking is guarded by an existence check, so the loop reads the
edited file, and the CSV is written from it.

Cash ran nothing above a cell reading ``df``: its read of the CSV was
current, and the writer was taken for one that writes back what its input
was read from (a counter, never re-fired), by the files of the ``df`` read
back from the CSV below it, not of the frame it wrote from.

When the unpacking is not guarded, Run All unpacks the archive again over
the edit. Cash keeps a change made from outside to a file a cell above
wrote, as it does for a counter file (`test_cells_rewriting_one_file_run_once`):
it re-runs what read the file, and ``df`` has the edit, as ``rows`` already
had; the edit is not written over.
"""

from __future__ import annotations

import io
import tarfile

import pytest

pytest.importorskip("pandas")

pytestmark = pytest.mark.files

FILES = 40

#: ``filter=`` came in 3.10.12; before it, ``extractall`` does what ``'fully_trusted'`` does.
EXTRACT_ARGS = "filter='fully_trusted'" if hasattr(tarfile, "fully_trusted_filter") else ""

UNPACK = f"with tarfile.open('data.tar') as tar:\n    tar.extractall({EXTRACT_ARGS})"
GUARDED = "if not os.path.isdir('aclImdb'):\n" + "\n".join("    " + line for line in UNPACK.splitlines())


def _cells(unpack: str) -> list[str]:
    return [
        "import os, tarfile\nimport pandas as pd\nimport cash\n%cash_on\n%cash_badge print",
        unpack,
        "rows = []\n"
        "for label in ('pos', 'neg'):\n"
        "    path = os.path.join('aclImdb', 'train', label)\n"
        "    for name in sorted(os.listdir(path)):\n"
        "        with open(os.path.join(path, name), 'r', encoding='utf-8') as fh:\n"
        "            rows.append((fh.read(), label))\n"
        "df = pd.DataFrame(rows, columns=['review', 'sentiment'])",
        "df = df.sample(frac=1, random_state=0).reset_index(drop=True)",
        "df.to_csv('movie_data.csv', index=False, encoding='utf-8')",
        "df = pd.read_csv('movie_data.csv', encoding='utf-8')\ndf = df.rename(columns={'0': 'review'})",
        "print('SHAPE', df.shape)",
        "print('EDITED', int(df['review'].str.contains('EDITED').sum()))",
    ]


def _archive(folder) -> None:
    with tarfile.open(folder / "data.tar", "w") as tar:
        for label in ("pos", "neg"):
            for i in range(FILES):
                data = f"review {label} {i}".encode()
                info = tarfile.TarInfo(f"aclImdb/train/{label}/{i}_7.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


@pytest.mark.timeout(150)
@pytest.mark.parametrize(("unpack", "edited"), [(GUARDED, 1), (UNPACK, 1)], ids=["guarded", "unguarded"])
def test_a_cell_reading_the_frame_after_a_review_is_edited_sees_the_edit(nb_runner, unpack, edited):
    work = nb_runner.work_dir
    _archive(work)
    nb_runner.create_notebook(_cells(unpack))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "EDITED 0" in nb_runner.get_output(8)

    (work / "aclImdb" / "train" / "neg" / "17_7.txt").write_text("EDITED review", encoding="utf-8")
    nb_runner.run_cells([8])

    out = nb_runner.get_output(8)
    assert f"EDITED {edited}" in out, out
    assert "EDITED" in (work / "movie_data.csv").read_text(encoding="utf-8"), "the CSV was not written again"


@pytest.mark.timeout(150)
@pytest.mark.fresh_kernel
def test_after_a_restart_the_frame_read_back_sees_the_edit(nb_runner):
    work = nb_runner.work_dir
    _archive(work)
    nb_runner.create_notebook(_cells(GUARDED))
    nb_runner.start_kernel()
    nb_runner.run_all()

    (work / "aclImdb" / "train" / "neg" / "17_7.txt").write_text("EDITED review", encoding="utf-8")
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cells([1, 8])

    out = nb_runner.get_output(8)
    assert "EDITED 1" in out, out
