"""The single-unit policy refuses a loop that changes files, not one that reads.

A loop run as one unit is one statement under one file tracker, so every file
it reads becomes a dependency of its entry. A loop that writes, moves or
removes files keeps the per-iteration path.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.control_structures import single_unit_policy


def _body(code: str) -> list[ast.stmt]:
    return ast.parse(f"for f in files:\n    {code}").body[0].body


@pytest.mark.parametrize(
    "code",
    [
        "df.to_csv(f)",
        "df.to_parquet(f)",
        "df.to_excel(f)",
        "np.save(f, a)",
        "np.savez(f, a=a)",
        "np.savetxt(f, a)",
        "fh.write(s)",
        "open(f, 'w').write(s)",
        "with open(f, 'a') as fh:\n        fh.write(s)",
        "pathlib.Path(f).write_text(s)",
        "os.remove(f)",
        "Path(f).unlink()",
        "os.rename(f, f + '.done')",
        "shutil.move(f, dest)",
        "pickle.dump(obj, open(f, 'wb'))",
    ],
)
def test_a_body_that_changes_files_writes_files(code):
    assert single_unit_policy.writes_files(_body(code))


@pytest.mark.parametrize(
    "code",
    [
        "d = pd.read_csv(f)",
        "d = pd.read_parquet(f)",
        "d = pd.read_excel(f)",
        "text = open(f).read()",
        "with open(f) as fh:\n        text = fh.read()",
        "arr = np.load(f)",
        "arr = np.loadtxt(f)",
        "cfg = json.load(open(f))",
        "names = os.listdir(f)",
        "hits = glob.glob(f + '/*.csv')",
        "text = Path(f).read_text()",
    ],
)
def test_a_body_that_only_reads_files_does_not(code):
    assert not single_unit_policy.writes_files(_body(code))


def _loop(body: str, n: int = 100) -> ast.For:
    return ast.parse(f"for f in range({n}):\n    {body}").body[0]


def test_a_long_reading_loop_runs_as_one_unit():
    node = _loop("d = pd.read_csv(f)\n    d['k'] = f\n    parts.append(d)")
    assert single_unit_policy.should_run_as_single_unit(node, range(100), {})


def test_a_long_writing_loop_stays_per_iteration():
    # Control: the same loop without the write clears every other threshold.
    assert single_unit_policy.should_run_as_single_unit(_loop("d = f * 2\n    e = d + 1\n    g = e"), range(100), {})
    node = _loop("d = f * 2\n    e = d + 1\n    open(str(f), 'w').write(str(e))")
    assert not single_unit_policy.should_run_as_single_unit(node, range(100), {})
