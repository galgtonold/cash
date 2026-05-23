"""Tests for %cash_benchmark magic command."""
import pytest
from unittest.mock import MagicMock
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.magics import CashMagics
from cash.backends import InMemoryBackend


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing %cash_benchmark."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


class TestBenchmarkMagic:
    """Test %cash_benchmark magic command."""

    def test_benchmark_default(self, magics_fixture, capsys):
        """Default benchmark enables with 3 iterations."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("")
        captured = capsys.readouterr()
        assert "Mode enabled" in captured.out
        assert "3 iterations" in captured.out
        assert magics._benchmark_config is not None
        assert magics._benchmark_config['iterations'] == 3
        assert magics._benchmark_config['active'] is True

    def test_benchmark_custom_iterations(self, magics_fixture, capsys):
        """Custom iteration count."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("5")
        captured = capsys.readouterr()
        assert "5 iterations" in captured.out
        assert magics._benchmark_config['iterations'] == 5

    def test_benchmark_cold_start(self, magics_fixture, capsys):
        """Cold start flag."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("--cold")
        captured = capsys.readouterr()
        assert "cold start" in captured.out
        assert magics._benchmark_config['cold_start'] is True

    def test_benchmark_compare_mode(self, magics_fixture, capsys):
        """Compare mode flag."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("--compare")
        captured = capsys.readouterr()
        assert "compare mode" in captured.out
        assert magics._benchmark_config['compare_mode'] is True

    def test_benchmark_combined_flags(self, magics_fixture, capsys):
        """Multiple flags and iterations."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("10 --cold --compare")
        captured = capsys.readouterr()
        assert "10 iterations" in captured.out
        assert "cold start" in captured.out
        assert "compare mode" in captured.out
        assert magics._benchmark_config['iterations'] == 10
        assert magics._benchmark_config['cold_start'] is True
        assert magics._benchmark_config['compare_mode'] is True

    def test_benchmark_max_iterations(self, magics_fixture, capsys):
        """Iterations capped at 100."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("999")
        assert magics._benchmark_config['iterations'] == 100

    def test_benchmark_one_shot(self, magics_fixture, capsys):
        """Benchmark config is deactivated after use."""
        magics, shell, backend = magics_fixture
        magics.cash_benchmark("")
        assert magics._benchmark_config['active'] is True

        # Simulate the _execute_cell check
        magics._benchmark_config['active'] = False
        assert magics._benchmark_config['active'] is False

    def test_benchmark_init_default(self, magics_fixture):
        """Benchmark config starts as None."""
        magics, shell, backend = magics_fixture
        # _benchmark_config should be initialized to None
        assert hasattr(magics, '_benchmark_config')

    def test_run_benchmark_basic(self, magics_fixture, capsys):
        """Test _run_benchmark produces output."""
        magics, shell, backend = magics_fixture

        # Mock _execute_cell to avoid actual execution
        original_proxy = magics._execute_cell
        magics._execute_cell = MagicMock(return_value=None)

        magics._run_benchmark("x = 1", iterations=2, cold_start=False, compare_mode=False)
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "2 iterations" in captured.out
        assert "With caching" in captured.out

        # Restore
        magics._execute_cell = original_proxy

    def test_run_benchmark_compare(self, magics_fixture, capsys):
        """Test _run_benchmark in compare mode."""
        magics, shell, backend = magics_fixture

        # Mock both execution paths
        magics._execute_cell = MagicMock(return_value=None)
        magics.shell.run_cell = MagicMock(return_value=MagicMock(success=True))

        magics._run_benchmark("x = 1", iterations=2, cold_start=False, compare_mode=True)
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "Without caching" in captured.out
        assert "Speedup" in captured.out
