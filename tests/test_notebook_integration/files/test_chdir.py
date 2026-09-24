"""File dependencies read through relative paths after ``os.chdir``.

Integration tests for file dependency tracking with os.chdir().

Issue: When os.chdir() is called before file operations, the same relative
path gets tracked as different absolute paths, causing cache misses.

os.chdir + relative-path reads must re-resolve against the live cwd.

A relative-path read (``pd.read_csv('data.csv')``) recorded only the realpath
resolved at first execution, frozen to that cwd. After editing an ``os.chdir``
cell to point at a different directory holding a DIFFERENT file with the same
relative name, even run_all served the old directory's data (the frozen realpath
still existed and was unmodified). Tracking the relative path too lets the
freshness check re-resolve it against the current cwd and catch the collision.

Verifier repro: chdir cell edited to a directory holding a DIFFERENT data.csv.

Minimal repro for the probe finding `chdir-relative-path-stale-collision`:
the file-dependency for a relative-path read is frozen to the realpath
resolved at FIRST execution (file_tracker._track_path), and cwd is not an
input to the cache decision, so after editing the os.chdir cell to point at
a directory holding a different file with the same relative name, run_all
serves the OLD directory's data.

Phase 1 ground-truths the expected values with a plain (with_cash=False)
kernel; phase 2 shows the cash kernel diverging on run_all.
"""

import pandas as pd
import pytest


@pytest.mark.files
def test_chdir_doesnt_break_file_caching(nb_runner, tmp_path):
    """
    Reproduces the bug: os.chdir() before pd.read_csv() breaks caching.

    Scenario:
    1. Cell 1: os.chdir() to change working directory
    2. Cell 2: pd.read_csv('data.csv') - uses relative path
    3. Re-execute cell 2 → should be cache HIT but currently is MISS

    The bug: os.path.abspath('data.csv') produces different absolute paths
    depending on the current working directory at execution time.
    """
    # Create test data in a subdirectory
    data_dir = tmp_path / "data_subdir"
    data_dir.mkdir()
    csv_path = data_dir / "test_data.csv"
    str(csv_path).replace("\\", "/")

    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv_path, index=False)

    # Create notebook that changes directory then reads file
    data_dir_str = str(data_dir).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import os; import pandas as pd",
            f"os.chdir(r'{data_dir_str}')",
            "df = pd.read_csv('test_data.csv'); print('Loaded CSV')",
        ]
    )

    nb_runner.start_kernel()

    # Enable cash debug
    nb_runner._run_async(nb_runner.client.kc._async_execute_interactive("%cash_debug on", store_history=False))

    nb_runner.run_all()

    output1 = nb_runner.get_output(3)
    assert "Loaded CSV" in output1  # First run should execute

    print("\n[DEBUG] First execution - checking debug logs")

    # Second run of cell 3 - should restore from cache (chdir persists in kernel)
    nb_runner.run_cell(3)
    nb_runner.get_output(3)

    # Check if it was restored from cache
    raw_output = nb_runner.get_raw_output(3)
    print(f"\n[DEBUG] Second execution raw output:\n{raw_output}")

    # The critical test: check for cache hit indicators in debug output
    # "Loaded CSV" appearing is OK - it's from replayed stdout, not re-execution
    # We need to check the badge or debug markers
    #
    # Looking for signs of cache hit:
    # - "CACHE_HIT_DEBUG" in debug output
    # - "Restored" or "RESTORED" in badge
    # - "Skipped" for the df assignment (already executed optimization)
    assert "CACHE_HIT_DEBUG" in raw_output or "Restored" in raw_output or "RESTORED" in raw_output.upper(), (
        f"Expected cache hit (CACHE_HIT_DEBUG or Restored marker), but not found. Raw output: {raw_output[:500]}"
    )


@pytest.mark.files
def test_chdir_in_same_notebook_multiple_cells(nb_runner, tmp_path):
    """
    Test that chdir in cell 1, then file operations in cells 2 and 3 work.

    When re-executing cell 3, it should find the cache from the first run.
    """
    data_dir = tmp_path / "project" / "data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "sales.csv"

    pd.DataFrame({"product": ["A", "B", "C"], "sales": [100, 200, 300]}).to_csv(csv_path, index=False)

    data_dir_str = str(data_dir).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import os; import pandas as pd",
            f"os.chdir(r'{data_dir_str}')",
            "df = pd.read_csv('sales.csv')",
            "print(f'Loaded {len(df)} rows')",
            "total = df['sales'].sum()",
            "print(f'Total sales: {total}')",
        ]
    )

    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(4)
    assert "Loaded 3 rows" in output1

    output2 = nb_runner.get_output(6)
    assert "Total sales: 600" in output2

    # Re-run cells 3-6
    nb_runner.run_cells([3, 4, 5, 6])

    # Outputs should be identical (from cache)
    assert nb_runner.get_output(4) == output1
    assert nb_runner.get_output(6) == output2


@pytest.mark.files
def test_absolute_path_unaffected_by_chdir(nb_runner, tmp_path):
    """
    Control test: absolute paths should work correctly regardless of chdir.

    This should pass even with the bug, proving the issue is with relative paths.
    """
    csv_path = tmp_path / "absolute_test.csv"
    csv_path_str = str(csv_path).replace("\\", "/")

    pd.DataFrame({"x": [10, 20], "y": [30, 40]}).to_csv(csv_path, index=False)

    # Use absolute path - should be unaffected by chdir
    nb_runner.create_notebook(
        [
            "import os; import pandas as pd",
            f"os.chdir(r'{str(tmp_path).replace(chr(92), '/')}')",  # Change to tmp_path
            f"df = pd.read_csv(r'{csv_path_str}')",  # But use absolute path
            "print(df.to_string())",
        ]
    )

    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(4)
    assert "10" in output1 and "30" in output1

    # Change directory to somewhere else
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    nb_runner.set_cell_source(2, f"os.chdir(r'{str(other_dir).replace(chr(92), '/')}')")
    nb_runner.run_cells([2, 3, 4])

    output2 = nb_runner.get_output(4)
    # Should still get cache hit because absolute path is the same
    assert output1 == output2


@pytest.mark.files
def test_relative_path_from_notebook_dir(nb_runner, tmp_path):
    """
    Test the desired behavior: relative paths should be resolved relative
    to the notebook's directory when using realpath.

    This test verifies that file dependencies are properly tracked even when
    the notebook and data file are in a subdirectory.
    """
    # Create notebook directory
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()

    # Create data file in the notebook directory
    csv_path = notebook_dir / "local_data.csv"
    pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_csv(csv_path, index=False)

    # Create notebook that uses relative path
    # The notebook's work_dir will be notebook_dir, so relative paths resolve there
    nb_runner.work_dir = notebook_dir
    nb_runner.create_notebook(
        [
            "import pandas as pd",
            "df = pd.read_csv('local_data.csv')",  # Relative to notebook dir
            "print(df.to_string())",
        ]
    )

    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(3)
    assert "1" in output1 and "3" in output1

    # Re-run cells 2-3 - should hit cache
    nb_runner.run_cells([2, 3])
    output2 = nb_runner.get_output(3)
    assert output1 == output2


# The file backend keeps an absolute cache_dir: a relative one would point
# somewhere else once the notebook calls os.chdir(), and writes to disk would
# fail silently.
@pytest.mark.files
class TestDiskPromotionAfterChdir:
    """Test that disk caching works correctly after os.chdir() in notebooks."""

    def test_computed_items_promoted_to_disk(self, nb_runner, tmp_path):
        """Verify that expensive computations are cached to RAM+DISK even after chdir.

        This simulates the real-world scenario:
        1. Cash is initialized at import time (CWD = project root)
        2. Notebook calls os.chdir() to a different directory
        3. An expensive computation runs
        4. The result should be stored in RAM+DISK, not just RAM
        """
        other_dir = str(tmp_path / "workdir").replace("\\", "/")
        str(tmp_path / "check_cash_dir").replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: chdir to a different directory
                f"import os\nos.makedirs('{other_dir}', exist_ok=True)\nos.chdir('{other_dir}')\nprint(f'CWD: {{os.getcwd()}}')",
                # Cell 2: Import time to simulate expensive operation
                "import time",
                # Cell 3: Simulate expensive computation (use sleep to ensure > 1s threshold)
                "time.sleep(1.5)\nresult = sum(range(1000000))\nprint(f'Result: {result}')",
                # Cell 4: Check if FileBackend has absolute path
                (
                    "import cash\n"
                    "b = cash._active.default_cash().backend\n"
                    "fb = b.backends[1]\n"
                    "print(f'FileBackend path absolute: {os.path.isabs(fb.cache_dir)}')\n"
                    "print(f'FileBackend path: {fb.cache_dir}')\n"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Check cell 4 output - FileBackend should have absolute path
        output = nb_runner.get_output(4)
        assert "FileBackend path absolute: True" in output, f"Expected absolute path, got: {output}"

    def test_cached_data_persists_after_chdir(self, nb_runner, tmp_path):
        """Verify cache data is written to the correct location after chdir."""
        other_dir = str(tmp_path / "workdir").replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: chdir
                f"import os\nos.makedirs('{other_dir}', exist_ok=True)\nos.chdir('{other_dir}')\nprint('OK')",
                # Cell 2: Store something via cash
                "x = list(range(100))\nprint(f'x has {{len(x)}} items')",
                # Cell 3: Check that the .cash directory has files at the correct location
                (
                    "import cash, os, glob\n"
                    "fb = cash._active.default_cash().backend.backends[1]\n"
                    "files = glob.glob(os.path.join(fb.cache_dir, '*'))\n"
                    "print(f'Cache dir: {fb.cache_dir}')\n"
                    "print(f'Cache files: {len(files)}')\n"
                    "# The cache dir should NOT be under the chdir target\n"
                    f'print(f\'Wrong location exists: {{os.path.exists(os.path.join("{other_dir}", ".cash"))}}\')\n'
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        # The wrong .cash location should not exist
        assert "Wrong location exists: False" in output3, f"Cache dir created in wrong location: {output3}"


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_chdir_relative_path_different_file_invalidates(nb_runner, tmp_path):
    dira = tmp_path / "dira"
    dirb = tmp_path / "dirb"
    dira.mkdir()
    dirb.mkdir()
    (dira / "data.csv").write_text("v\n1\n2\n3\n", encoding="utf-8")
    (dirb / "data.csv").write_text("v\n100\n200\n300\n400\n", encoding="utf-8")
    pa = str(dira).replace("\\", "/")
    pb = str(dirb).replace("\\", "/")

    nb_runner.create_notebook(
        [
            "import os\nimport pandas as pd",
            f"os.chdir('{pa}')",
            "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)

    # Point the chdir cell at a DIFFERENT directory whose data.csv differs.
    nb_runner.set_cell_source(2, f"os.chdir('{pb}')")
    nb_runner.run_all()
    assert "vals = [100, 200, 300, 400]" in nb_runner.get_output(3), (
        f"reader served the OLD directory's data after chdir edit: {nb_runner.get_output(3)!r}"
    )


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_relative_read_same_cwd_stays_cached(nb_runner, tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "data.csv").write_text("v\n7\n8\n", encoding="utf-8")
    p = str(d).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import os\nimport pandas as pd",
            f"os.chdir('{p}')",
            "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [7, 8]" in nb_runner.get_output(3)

    # No chdir change, no file change → second run restores from cache.
    nb_runner.run_all()
    assert "vals = [7, 8]" in nb_runner.get_output(3)


def _p(path) -> str:
    return str(path).replace("\\", "/")


@pytest.mark.timeout(90)
def test_chdir_collision_run_all_serves_old_dirs_file(nb_runner, tmp_path):
    dira = tmp_path / "dira"
    dirb = tmp_path / "dirb"
    dira.mkdir()
    dirb.mkdir()
    pd.DataFrame({"v": [1, 2, 3]}).to_csv(dira / "data.csv", index=False)
    pd.DataFrame({"v": [100, 200, 300, 400]}).to_csv(dirb / "data.csv", index=False)

    cells = [
        "import os\nimport pandas as pd",
        f"os.chdir(r'{_p(dira)}')",
        "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
    ]
    chdir_b = f"os.chdir(r'{_p(dirb)}')"

    # ---- Phase 1: ground truth (plain kernel, no cash) -------------------
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, chdir_b)
    nb_runner.run_all()
    truth = nb_runner.get_output(3)
    assert "vals = [100, 200, 300, 400]" in truth, (
        f"HARNESS SANITY: plain kernel did not see dirb's data.csv: {truth!r}"
    )
    nb_runner.shutdown()

    # ---- Phase 2: same sequence with cash --------------------------------
    nb_runner.create_notebook(cells)  # reset cell 2 back to dira
    nb_runner.start_kernel(with_cash=True)
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, chdir_b)
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "vals = [100, 200, 300, 400]" in out, (
        f"run_all after editing the chdir cell to a directory holding a "
        f"DIFFERENT data.csv served the OLD directory's data (ground truth "
        f"= [100, 200, 300, 400]). Got: {out!r}"
    )
