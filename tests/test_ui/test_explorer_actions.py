import shutil
import tempfile
import unittest

from cash import Cash


class TestExplorerActions(unittest.TestCase):
    def setUp(self):
        # A directory of this test's own: under xdist the tests of this class
        # run at the same time in different workers, and a shared cwd-relative
        # folder let one test's tearDown delete the other's files.
        self.cache_dir = tempfile.mkdtemp(prefix="cash_test_")
        self.app = Cash(cache_dir=self.cache_dir)

    def tearDown(self):
        self.app.backend.clear()
        shutil.rmtree(self.cache_dir)

    def test_clear_function(self):
        @self.app.cache
        def func_a(x):
            return x

        @self.app.cache
        def func_b(x):
            return x

        # Populate cache
        func_a(1)
        func_a(2)
        func_b(1)

        explorer = self.app.explorer()
        entries = explorer.list_entries()
        self.assertEqual(len(entries), 3)

        # Clear func_a (use module-qualified key)
        func_a_key = Cash.get_func_key(func_a)
        count = explorer.clear_function(func_a_key)
        self.assertEqual(count, 2)

        entries = explorer.list_entries()
        self.assertEqual(len(entries), 1)
        func_b_key = Cash.get_func_key(func_b)
        self.assertEqual(entries[0]["func_name"], func_b_key)


if __name__ == "__main__":
    unittest.main()
