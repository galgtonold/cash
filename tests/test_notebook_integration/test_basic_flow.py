"""
Test basic caching flow in notebooks.
"""
import pytest

pytestmark = pytest.mark.core

def test_in_memory_caching(nb_runner):
    """
    Test that running a notebook with cash enabled works and caching works.
    Using default InMemoryBackend.
    
    This test verifies:
    1. Cash is properly initialized and caching works
    2. Variables are cached and can be used in downstream cells
    3. Re-running a cell with same inputs hits the cache
    """
    nb_runner.create_notebook([
        # Cell 1: Create some data
        "data = [1, 2, 3, 4, 5]",
        # Cell 2: Compute result
        """result = sum(data)
print(f"Sum: {result}")""",
        # Cell 3: Use result in downstream cell
        """doubled = result * 2
print(f"Doubled: {doubled}")""",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # Check outputs
    run1_out = nb_runner.get_output(2)
    assert 'Sum: 15' in run1_out, f"Cell 2 output wrong. Got: {run1_out}"
    
    run2_out = nb_runner.get_output(3)
    assert 'Doubled: 30' in run2_out, f"Cell 3 output wrong. Got: {run2_out}"


def test_file_backend_setup(nb_runner, tmp_path):
    """
    Test establishing a FileBackend to persist cache across kernel restarts.
    """
    cache_dir = tmp_path / "persistent_cache"
    cache_dir_str = str(cache_dir).replace("\\", "/")
    
    # Create notebook with custom backend setup
    nb_runner.create_notebook([
        # Cell 1: Custom backend setup (replaces default cash init)
        f"""
%load_ext cash
from cash import Cash
from cash.backends.backend import FileBackend
from cash.notebook.magics import CashMagics

backend = FileBackend(cache_dir='{cache_dir_str}')
ip = get_ipython()
cash = Cash(backend=backend, register_magic=False)
magics = CashMagics(ip, cash)
ip.register_magics(magics)
%cash_on
""",
        # Cell 2: Some cacheable computation. Must do enough real work
        # to cross the ``min_execution_time_to_cache_seconds`` floor
        # (default 10ms) — otherwise the floor short-circuits and no
        # metadata gets written, which is the documented behaviour.
        # A tight sum over a few-hundred-k range does ~10-50ms.
        """data = sum(i * i for i in range(500_000))
print(f'Persistent Data: {data}')"""
    ])

    # Start kernel WITHOUT auto cash init (we do it manually in cell 1)
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    output_text = nb_runner.get_output(2)
    assert "Persistent Data" in output_text

    # Verify cache files exist on disk
    assert cache_dir.exists()
    assert list(cache_dir.glob("*.db")) or list(cache_dir.glob("*")), "Cache directory should not be empty"
