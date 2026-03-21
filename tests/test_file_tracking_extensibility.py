
import unittest
import os
import sys
import tempfile
from unittest.mock import MagicMock
from cash.notebook.file_tracker import FileDependencyRegistry, FileAccessTracker
from cash.core import Cash

class TestFileTrackingExtensibility(unittest.TestCase):
    def setUp(self):
        self.mock_shell = MagicMock()
        self.mock_shell.user_ns = {}
        
        # Reset registry for each test to avoid pollution
        FileDependencyRegistry._instance = None
        
        # Test helpers
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write("data")
            self.temp_path = tf.name.replace(os.sep, '/')

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
            
    def test_registry_basics(self):
        registry = FileDependencyRegistry()
        
        # Check defaults
        handlers = registry.get_handlers_for_module('builtins')
        self.assertTrue(any(h[0] == 'open' for h in handlers))
        
        handlers = registry.get_handlers_for_module('pandas')
        self.assertTrue(any(h[0] == 'read_*' for h in handlers))

    def test_api_registration(self):
        cash = Cash()
        
        # Define a mock library and function
        mock_lib = MagicMock()
        sys.modules['mylib'] = mock_lib
        
        def read_custom(path):
            return "read"
        
        mock_lib.read_custom = read_custom
        
        # Register handler via Cash API
        def custom_handler(original, tracker):
            def wrapper(path, *args, **kwargs):
                tracker(path)
                return original(path, *args, **kwargs)
            return wrapper
            
        cash.register_file_handler('mylib', 'read_custom', custom_handler)
        
        # Verify it works
        tracker = FileAccessTracker(self.mock_shell.user_ns)
        with tracker:
            mock_lib.read_custom(self.temp_path)
            
        self.assertIn(self.temp_path, tracker.get_accessed_files())
        
        del sys.modules['mylib']

    def test_wildcard_registration(self):
        registry = FileDependencyRegistry()
        
        mock_lib = MagicMock()
        sys.modules['wildlib'] = mock_lib
        mock_lib.load_data = MagicMock(return_value="data")
        mock_lib.load_config = MagicMock(return_value="config")
        
        # Register wildcard
        def path_handler(original, tracker):
            def wrapper(path, *args, **kwargs):
                tracker(path)
                return original(path, *args, **kwargs)
            return wrapper
            
        registry.register('wildlib', 'load_*', path_handler)
        
        tracker = FileAccessTracker(self.mock_shell.user_ns)
        with tracker:
            mock_lib.load_data(self.temp_path)
            
        self.assertIn(self.temp_path, tracker.get_accessed_files())
        
        del sys.modules['wildlib']

if __name__ == '__main__':
    unittest.main()
