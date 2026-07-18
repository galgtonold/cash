
import unittest
import os
import time
import tempfile
import pandas as pd
from unittest.mock import MagicMock
from cash.notebook.statement import StatementProcessor
from cash.core import Cash

class TestFileTracking(unittest.TestCase):
    def setUp(self):
        self.mock_shell = MagicMock()
        self.mock_shell.user_ns = {}
        
        # Use an in-memory backend for testing logic, but we need Cash instance
        self.cash = Cash(backend=MagicMock()) 
        # Mock backend methods to behave like a dict
        self._cache_storage = {}
        self.cash.backend.get.side_effect = lambda k: self._cache_storage.get(k, (None, None))
        self.cash.backend.set.side_effect = self._mock_set
        
        self.processor = StatementProcessor(
            self.mock_shell, 
            self.cash, 
            debug=True,
            compute_hash_fn=lambda x: str(hash(x))
        )
        
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write("data1")
            # realpath, not just a separator swap: cash records dependencies in
            # canonical form (resolve_file_dep_path -> normalize_path(realpath))
            # so that two spellings of one file are one dependency. The temp
            # directory is spelled non-canonically on both CI platforms -- macOS
            # hands out /var/... for /private/var/..., and the Windows runner's
            # TEMP is the 8.3 short name C:\Users\RUNNER~1 for runneradmin -- so
            # comparing against the raw name failed there while passing on any
            # machine whose temp dir happens to already be canonical.
            self.temp_path = os.path.realpath(tf.name).replace(os.sep, '/')

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def _mock_set(self, key, value, metadata=None, serializer=None):
        self._cache_storage[key] = (metadata, value)

    def test_open_tracking(self):
        code = f"with open('{self.temp_path}', 'r') as f: content = f.read()"
        
        # First execution: Should be a cache miss (execution)
        print("\n--- Run 1 (Cache Miss) ---")
        self.processor.process_statement(code)
        
        # Verify content matches
        self.assertEqual(self.mock_shell.user_ns.get('content'), 'data1')
        
        # Check if cache entry was created with file dependency
        cache_key = list(self._cache_storage.keys())[0]
        metadata, _ = self._cache_storage[cache_key]
        
        print(f"Metadata: {metadata}")
        # NOTE: This assertion will fail until implementation is done
        self.assertIn('file_dependencies', metadata)
        self.assertIn(self.temp_path, metadata['file_dependencies'])
        
        # Modify file
        time.sleep(1.1) # Ensure mtime changes (some systems have 1s resolution)
        with open(self.temp_path, 'w') as f:
            f.write("data2")
            
        # Second execution: Should be a cache miss due to file change
        print("\n--- Run 2 (File Modified) ---")
        # Reset mock shell content to verify re-execution updates it
        self.mock_shell.user_ns['content'] = 'old'
        
        self.processor.process_statement(code)
        
        self.assertEqual(self.mock_shell.user_ns.get('content'), 'data2')

    def test_pandas_tracking(self):
        # Create CSV
        csv_path = self.temp_path + ".csv"
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df.to_csv(csv_path, index=False)
        csv_path = csv_path.replace(os.sep, '/') # Normalize for tracking
        
        try:
            code = f"import pandas as pd; df = pd.read_csv('{csv_path}')"
            
            # Run 1
            print("\n--- Pandas Run 1 ---")
            self.processor.process_statement(code)
            
            # Check dependency
            cache_key = [k for k in self._cache_storage if 'stmt' in k][-1]
            metadata, _ = self._cache_storage[cache_key]
            # NOTE: Will fail until implemented
            self.assertIn('file_dependencies', metadata)
            self.assertIn(csv_path, metadata['file_dependencies'])
            
            # Modify CSV
            time.sleep(1.1)
            df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
            df2.to_csv(csv_path, index=False)
            
            # Run 2
            print("\n--- Pandas Run 2 (Modified) ---")
            self.processor.process_statement(code)
            
            result_df = self.mock_shell.user_ns['df']
            self.assertEqual(result_df.iloc[0]['a'], 5)
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_pathlib_tracking(self):
        try:
            from pathlib import Path  # noqa: F401
            code = f"from pathlib import Path; content = Path('{self.temp_path}').read_text()"
            
            # Run 1
            print("\\n--- Pathlib Run 1 ---")
            self.processor.process_statement(code)
            
            # Check dependency
            cache_key = list(self._cache_storage.keys())[-1]
            metadata, _ = self._cache_storage[cache_key]
            self.assertIn('file_dependencies', metadata)
            self.assertIn(self.temp_path, metadata['file_dependencies'])
            
            # Modify file
            time.sleep(1.1)
            with open(self.temp_path, 'w') as f:
                f.write("data_pathlib_2")
                
            # Run 2
            print("\\n--- Pathlib Run 2 (Modified) ---")
            # Reset variable to check update
            self.mock_shell.user_ns['content'] = 'old'
            
            self.processor.process_statement(code)
            
            self.assertEqual(self.mock_shell.user_ns.get('content'), 'data_pathlib_2')
            
        except ImportError:
            pass

    def test_numpy_tracking(self):
        try:
            import numpy as np
            txt_path = self.temp_path + ".txt"
            np.savetxt(txt_path, [1, 2, 3])
            txt_path = txt_path.replace(os.sep, '/')
            
            code = f"import numpy as np; arr = np.loadtxt('{txt_path}')"
            
            # Run 1
            print("\\n--- Numpy Run 1 ---")
            self.processor.process_statement(code)
            
            # Check dependency
            found_dep = False
            for _key, (meta, _val) in self._cache_storage.items():
                if meta and 'file_dependencies' in meta and txt_path in meta['file_dependencies']:
                    found_dep = True
                    break
            
            if not found_dep:
                print("WARNING: Numpy dependency NOT found in metadata")
            
            self.assertTrue(found_dep, "Numpy file dependency not tracked")
            
            # Modify file
            time.sleep(1.1)
            np.savetxt(txt_path, [4, 5, 6])
            
            # Run 2
            print("\\n--- Numpy Run 2 (Modified) ---")
            self.processor.process_statement(code)
            
            result_arr = self.mock_shell.user_ns['arr']
            self.assertEqual(result_arr[0], 4.0)
            
            if os.path.exists(txt_path):
                os.remove(txt_path)
                
        except ImportError:
            print("Skipping numpy test: numpy not installed")

if __name__ == '__main__':
    unittest.main()
