"""Files read before a kernel restart, changed or not after it."""

import pytest


@pytest.mark.stress
@pytest.mark.files
class TestFileChangeWithRestart:
    """File changes combined with kernel restarts."""

    def test_file_change_after_restart(self, nb_runner, tmp_path):
        """Change file after restart — should detect stale cache."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("val\n1\n2\n3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "s = df['val'].sum()\nprint(f's = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 6" in nb_runner.get_output(2)

        # Restart + change file
        nb_runner.shutdown()
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 60" in nb_runner.get_output(2)

    def test_file_change_and_code_edit_after_restart(self, nb_runner, tmp_path):
        """File changes + code edit + kernel restart — triple stress."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n5\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "result = df['x'].iloc[0] * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        csv_path.write_text("x\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['x'].iloc[0] * 3\nprint(f'result = {result}')")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(2)


@pytest.mark.files
@pytest.mark.restore
class TestDiskPersistenceRestore:
    """Test disk persistence and restore after kernel restart."""

    @pytest.mark.timeout(60)
    def test_simple_values_persist_across_restart(self, nb_runner, tmp_path):
        """
        Test that simple values cached to disk can be restored after restart.
        Uses FileBackend to ensure disk persistence.
        """
        cache_dir = tmp_path / "cache"
        cache_dir_str = str(cache_dir).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Setup with file backend
                f"""import time
from cash import Cash
from cash.backends import FileBackend
from cash.notebook.ipython.magics import CashMagics

backend = FileBackend(cache_dir='{cache_dir_str}')
ip = get_ipython()
cash = Cash(backend=backend, register_magic=False)
magics = CashMagics(ip, cash)
ip.register_magics(magics)
%cash_on
time.sleep(0.1)  # Ensure disk writes complete""",
                # Cell 2: Slow computation (persists to disk)
                """# @cash: persist
import time
time.sleep(0.3)
x = 42
print(f'x = {x}')""",
                # Cell 3: Depends on x
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "x = 42" in out2, f"Initial run cell 2: {out2}"
        out3 = nb_runner.get_output(3)
        assert "y = 84" in out3, f"Initial run cell 3: {out3}"
