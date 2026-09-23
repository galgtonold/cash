import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from cash.core import Cash
from cash.tracking import file_tracker
from cash.tracking.file_tracker import FileAccessTracker, FileDependencyRegistry, file_registry


class TestFileTrackingExtensibility(unittest.TestCase):
    def setUp(self):
        self.mock_shell = MagicMock()
        self.mock_shell.user_ns = {}

        # A fresh registry for each test, so handlers do not leak between them
        self._saved_registry = file_tracker._registry
        file_tracker._registry = FileDependencyRegistry()

        # Test helpers
        with tempfile.NamedTemporaryFile(delete=False, mode="w+") as tf:
            tf.write("data")
            # Canonical form — see the note in test_notebook/test_statement_file_tracking.py: the CI
            # runners' temp dirs are spelled non-canonically (macOS /var
            # symlink, Windows 8.3 short name) and cash records the resolved
            # path, so the raw name never matches there.
            self.temp_path = os.path.realpath(tf.name).replace(os.sep, "/")

    def tearDown(self):
        file_tracker._registry = self._saved_registry
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_registry_basics(self):
        registry = file_registry()

        # Check defaults
        handlers = registry.get_handlers_for_module("sqlite3")
        self.assertTrue(any(h[0] == "connect" for h in handlers))

        handlers = registry.get_handlers_for_module("pandas")
        self.assertTrue(any(h[0] == "read_*" for h in handlers))

    def test_api_registration(self):
        cash = Cash()

        # Define a mock library and function
        mock_lib = MagicMock()
        sys.modules["mylib"] = mock_lib

        def read_custom(path):
            return "read"

        mock_lib.read_custom = read_custom

        # Register handler via Cash API
        def custom_handler(original, tracker):
            def wrapper(path, *args, **kwargs):
                tracker(path)
                return original(path, *args, **kwargs)

            return wrapper

        cash.register_file_handler("mylib", "read_custom", custom_handler)

        # Verify it works
        tracker = FileAccessTracker(self.mock_shell.user_ns)
        with tracker:
            mock_lib.read_custom(self.temp_path)

        self.assertIn(self.temp_path, tracker.get_accessed_files())

        del sys.modules["mylib"]

    def test_wildcard_registration(self):
        registry = file_registry()

        mock_lib = MagicMock()
        sys.modules["wildlib"] = mock_lib

        # Real functions, not MagicMocks: the install-once skip checks
        # getattr(fn, '_is_file_tracker_patch', False), and a MagicMock
        # returns a truthy child for that, so the patcher would skip it.
        def load_data(path):
            return "data"

        def load_config(path):
            return "config"

        mock_lib.load_data = load_data
        mock_lib.load_config = load_config

        # Register wildcard
        def path_handler(original, tracker):
            def wrapper(path, *args, **kwargs):
                tracker(path)
                return original(path, *args, **kwargs)

            return wrapper

        registry.register("wildlib", "load_*", path_handler)

        tracker = FileAccessTracker(self.mock_shell.user_ns)
        with tracker:
            mock_lib.load_data(self.temp_path)

        self.assertIn(self.temp_path, tracker.get_accessed_files())

        del sys.modules["wildlib"]


if __name__ == "__main__":
    unittest.main()
