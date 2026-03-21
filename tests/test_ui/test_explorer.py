import unittest
import shutil
import os
from cash import Cash

class TestCacheExplorer(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "test_explorer_cache"
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.app = Cash(cache_dir=self.cache_dir)

    def tearDown(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_list_entries(self):
        @self.app.cache
        def func1(x):
            return x + 1
            
        @self.app.cache(ttl=60)
        def func2(x):
            return x * 2
            
        func1(10)
        func2(20)
        
        explorer = self.app.explorer()
        entries = explorer.list_entries()
        
        self.assertEqual(len(entries), 2)
        
        func_names = sorted([e.get('func_name') for e in entries])
        expected_names = sorted([Cash._get_func_key(func1), Cash._get_func_key(func2)])
        self.assertEqual(func_names, expected_names)
                                      
        # Check metadata
        e2 = [e for e in entries if e.get('func_name').endswith('func2')][0]
        self.assertEqual(e2['ttl'], 60)
        self.assertIn('timestamp', e2)
        self.assertIn('key', e2)
        self.assertIn('source_code', e2)

    def test_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
            
        @self.app.cache
        def func1(x):
            return x
            
        func1(1)
        
        explorer = self.app.explorer()
        df = explorer.to_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn('func_name', df.columns)
        self.assertIn('timestamp_human', df.columns)

if __name__ == '__main__':
    unittest.main()
