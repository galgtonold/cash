"""Which statements only delete files.

A replay of a chart cell's drawing pulls the cell's deletions along
(``FileWriterScheduler._whole_cell_writers``): without
``for old in OUT.glob('*.png'): old.unlink()`` a replay after ``N`` went
from 3 to 2 left the third chart behind.
"""

import pytest

from cash.analysis.file_effects import statement_only_deletes_files


@pytest.mark.parametrize(
    "code",
    [
        "for old in OUT.glob('*.png'):\n    old.unlink()",
        "for old in OUT.glob('*.png'):\n    old.unlink(missing_ok=True)",
        "for f in glob.glob('out/*.png'):\n    os.remove(f)",
        "os.unlink(path)",
        "shutil.rmtree(OUT)",
        "with open(p) as fh:\n    os.remove(fh.name)",
    ],
)
def test_a_deletion(code):
    assert statement_only_deletes_files(code)


@pytest.mark.parametrize(
    "code",
    [
        "x = 1",
        "(OUT / 'a.txt').write_text('x')",
        "for f in fs:\n    os.remove(f)\n    open(f, 'w').write('x')",
        "os.rename(a, b)",
        "for f in fs:\n    os.remove(f)\n    f.touch()",
    ],
)
def test_not_only_a_deletion(code):
    """Control: a statement that also writes, or writes another way, replays
    by the rules for its writes."""
    assert not statement_only_deletes_files(code)
