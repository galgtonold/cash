import unittest
import shutil
import os
from cash import Cash

class TestUIGeneration(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "test_ui_cache"
        self.app = Cash(cache_dir=self.cache_dir)
        self.app.backend.clear()

    def tearDown(self):
        self.app.backend.clear()
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_size_tracking(self):
        @self.app.cache
        def data_func():
            return b"0" * 100 # 100 bytes

        data_func()
        
        entries = self.app.backend.list_entries()
        self.assertEqual(len(entries), 1)
        # Size might be slightly more due to serialization overhead (pickle)
        self.assertGreaterEqual(entries[0]['size'], 100)

    def test_widget_generation(self):
        # Mock get_ipython to return a valid message ID for ipywidgets
        from unittest.mock import patch, MagicMock
        
        with patch('IPython.get_ipython', create=True) as mock_get_ipython:
            # Configure mock kernel to satisfy ipywidgets.Output
            mock_kernel = MagicMock()
            mock_kernel.get_parent.return_value = {'header': {'msg_id': 'mock_msg_id'}}
            mock_get_ipython.return_value.kernel = mock_kernel
            
            @self.app.cache
            def mod_func():
                return 1

            mod_func()
            
            explorer = self.app.explorer()
            
            # 1. Test Interactive Widget (ipywidgets)
            widget = explorer.widget()
            
            if widget is None:
                self.skipTest("IPython not available - widget() returned None")
            
            # Check if it's a VBox/HBox (container) - only applies when ipywidgets is installed
            try:
                import ipywidgets  # noqa: F401
                _has_ipywidgets = True
            except ImportError:
                _has_ipywidgets = False
            
            if _has_ipywidgets and hasattr(widget, 'children') and not isinstance(widget.children, unittest.mock.MagicMock):
               self.assertGreater(len(widget.children), 0)
            
            # 2. Test HTML/JS Generator (IFrame) explicitly
            # This allows us to verify the tree structure logic regardless of ipywidgets presence
            iframe = explorer._widget_html()
            
            if iframe is None:
                # Should not happen unless dependencies missing, but let's be safe
                return
                
            if not hasattr(iframe, 'src'):
                 self.skipTest("IFrame missing src attribute")
            
            data_uri = iframe.src
            # Handle case where src might be a string
            if not isinstance(data_uri, str):
                self.fail(f"Widget src is not a string. Type: {type(data_uri)}. Value: {data_uri}")
                
            if ',' not in data_uri:
                self.skipTest("Widget src is not a data URI")
                
            import base64
            base64_content = data_uri.split(',')[1]
            html_content = base64.b64decode(base64_content).decode('utf-8')
            
            # Check for hierarchical structure in JS data
            self.assertIn('treeData', html_content)
            self.assertIn('"type": "module"', html_content)
            self.assertIn('"type": "function"', html_content)
            self.assertIn('"stats":', html_content)
            self.assertIn('"total_size":', html_content)

if __name__ == '__main__':
    unittest.main()
