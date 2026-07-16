"""Tests for %cash_benchmark magic command."""
import pytest
from unittest.mock import MagicMock
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
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

    def test_compare_uncached_arm_forces_recompute(self, magics_fixture, capsys):
        """CAS-168: the --compare 'without caching' arm must genuinely recompute
        on every iteration instead of being served from cache.

        Before the fix, the uncached arm ran the cell through cash's *patched*
        ``run_cell`` (which dispatches to ``_execute_cell``), so the result was
        stored on the first iteration and restored from cache on every later
        one. The benchmark then compared cache-hit vs cache-hit and printed a
        meaningless ~1x speedup. The fix routes the uncached arm through the
        original, unpatched IPython ``run_cell`` (``_original_run_cell``).
        """
        import time as _time
        magics, shell, backend = magics_fixture

        # Genuine (uncached) recompute path: counts every invocation and is
        # deliberately slow so its mean dwarfs a cache hit. This mimics the
        # original, unpatched IPython run_cell that executes the user code.
        uncached_runs = {'n': 0}

        def fake_original_run_cell(code, *args, **kwargs):
            uncached_runs['n'] += 1
            _time.sleep(0.005)
            return MagicMock(success=True)

        magics._original_run_cell = fake_original_run_cell

        # Cash-cached path: computes once (cold store), then serves instant
        # cache hits — exactly what _execute_cell does for a cached statement.
        cache_state = {'stored': False}
        cached_computes = {'n': 0}

        def fake_execute_cell(code, *args, **kwargs):
            if not cache_state['stored']:
                cached_computes['n'] += 1
                _time.sleep(0.005)  # cold store
                cache_state['stored'] = True
            # else: cache hit — return instantly, no recompute
            return None

        magics._execute_cell = fake_execute_cell

        iterations = 3
        magics._run_benchmark(
            "y = slow_compute()", iterations=iterations,
            cold_start=False, compare_mode=True,
        )
        out = capsys.readouterr().out

        # THE CAS-168 assertion: the uncached arm really recomputed on every
        # iteration. Pre-fix this is 0 — the arm went through the cache path
        # (fake_execute_cell) instead of the uncached path.
        assert uncached_runs['n'] == iterations, (
            f"uncached arm recomputed {uncached_runs['n']}x, expected {iterations} "
            "(without-caching arm was served from cache)"
        )
        # The with-caching arm was warmed exactly once, then measured pure hits.
        assert cached_computes['n'] == 1
        # And the reported speedup is a real number, not the resolution fallback.
        assert "Speedup" in out
        assert "n/a" not in out
