"""Integration tests for disk promotion after os.chdir().

Reproduces issue where notebooks calling os.chdir() caused FileBackend
to silently fail because cache_dir was stored as a relative path.
"""
import pytest


pytestmark = pytest.mark.files


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
        other_dir = str(tmp_path / "workdir").replace('\\', '/')
        str(tmp_path / "check_cash_dir").replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: chdir to a different directory
            f"import os\nos.makedirs('{other_dir}', exist_ok=True)\nos.chdir('{other_dir}')\nprint(f'CWD: {{os.getcwd()}}')",
            # Cell 2: Import time to simulate expensive operation
            "import time",
            # Cell 3: Simulate expensive computation (use sleep to ensure > 1s threshold)
            "time.sleep(1.5)\nresult = sum(range(1000000))\nprint(f'Result: {result}')",
            # Cell 4: Check if FileBackend has absolute path
            (
                "import cash\n"
                "b = cash._global_cash.backend\n"
                "fb = b.backends[1]\n"
                "print(f'FileBackend path absolute: {os.path.isabs(fb.cache_dir)}')\n"
                "print(f'FileBackend path: {fb.cache_dir}')\n"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Check cell 4 output - FileBackend should have absolute path
        output = nb_runner.get_output(4)
        assert "FileBackend path absolute: True" in output, f"Expected absolute path, got: {output}"

    def test_cached_data_persists_after_chdir(self, nb_runner, tmp_path):
        """Verify cache data is written to the correct location after chdir."""
        other_dir = str(tmp_path / "workdir").replace('\\', '/')

        nb_runner.create_notebook([
            # Cell 1: chdir
            f"import os\nos.makedirs('{other_dir}', exist_ok=True)\nos.chdir('{other_dir}')\nprint('OK')",
            # Cell 2: Store something via cash
            "x = list(range(100))\nprint(f'x has {{len(x)}} items')",
            # Cell 3: Check that the .cash directory has files at the correct location
            (
                "import cash, os, glob\n"
                "fb = cash._global_cash.backend.backends[1]\n"
                "files = glob.glob(os.path.join(fb.cache_dir, '*'))\n"
                "print(f'Cache dir: {fb.cache_dir}')\n"
                "print(f'Cache files: {len(files)}')\n"
                "# The cache dir should NOT be under the chdir target\n"
                f"print(f'Wrong location exists: {{os.path.exists(os.path.join(\"{other_dir}\", \".cash\"))}}')\n"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output3 = nb_runner.get_output(3)
        # The wrong .cash location should not exist
        assert "Wrong location exists: False" in output3, f"Cache dir created in wrong location: {output3}"
