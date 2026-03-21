import unittest
from unittest.mock import MagicMock
from cash.core import Cash
from cash.exceptions import CashError
from cash.notebook.magics import CashMagics
from cash.ui.debugger import CacheDebugger
from cash.backends.backend import InMemoryBackend

class MockShell:
    def __init__(self):
        self.user_ns = {}
        self.magics_manager = MagicMock()
        # Mock the magics structure
        self.magics_manager.magics = {'cell': {}}
        self.events = MagicMock()
        self.ast_transformers = []
        self.run_cell = MagicMock()

class TestCacheDebugger(unittest.TestCase):
    def setUp(self):
        self.shell = MockShell()
        self.backend = InMemoryBackend()
        self.cash_instance = Cash(backend=self.backend)
        self.magics = CashMagics(self.shell, self.cash_instance)
        
        # Register the magic in the mock shell as it would be in IPython
        # IPython stores the bound method usually, or the object?
        # Actually, magics_manager.magics['cell']['cash'] usually points to the function.
        # But for class-based magics, the function is a bound method of the instance.
        self.shell.magics_manager.magics['cell']['cash'] = self.magics.cash

    def test_init_with_instance(self):
        """Test initializing with explicit Cash instance."""
        debugger = CacheDebugger(self.shell, self.cash_instance)
        self.assertEqual(debugger.cash, self.cash_instance)

    def test_init_with_module_and_magic_lookup(self):
        """Test initializing with module (simulating user error) and falling back to magic lookup."""
        import cash as cash_module
        
        # This should find self.cash_instance via self.shell -> magics -> cash magic -> __self__
        debugger = CacheDebugger(self.shell, cash_module)
        self.assertEqual(debugger.cash, self.cash_instance)

    def test_init_with_none_and_magic_lookup(self):
        """Test initializing with None and falling back to magic lookup."""
        debugger = CacheDebugger(self.shell, None)
        self.assertEqual(debugger.cash, self.cash_instance)

    def test_magic_lookup_failure(self):
        """Test failure when magic is not registered."""
        self.shell.magics_manager.magics['cell'] = {} # Clear magics
        
        with self.assertRaises(CashError):
            CacheDebugger(self.shell, None)

if __name__ == '__main__':
    unittest.main()
