"""A ``@cash.pure`` function that only makes an argument, or a waiver on a
print helper's line, does not let a statement whose helper writes a file be
served from the cache: the next Run All writes the files again, as a plain
kernel does."""

import pytest

pytestmark = [pytest.mark.integration]

CELLS = [
    """import os, gzip, shutil, time
import cash
@cash.pure
def normalize(v):
    return v / 2
def export(v, p):
    time.sleep(0.3)
    f = gzip.open(p, 'wt'); f.write(str(v)); f.close()
    return v
def log(msg):
    print(msg)  # @cash:assume-safe
def export_logged(v, p):
    log('exporting')
    return export(v, p)
""",
    "shutil.rmtree('out', ignore_errors=True); os.makedirs('out')",
    "s = export(normalize(4), 'out/a.gz')",
    "t = export_logged(4, 'out/b.gz')",
    "print(sorted(os.listdir('out')))",
]


def test_both_files_are_written_again(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().endswith("['a.gz', 'b.gz']")
