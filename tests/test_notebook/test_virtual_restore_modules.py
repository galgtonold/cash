"""
Tests for unified cache key computation and module lineage propagation.

Verifies that:
1. compute_cache_key produces identical keys regardless of call site
2. Module lineages are propagated to variable_lineage during simulation
3. _update_virtual_lineage and _try_virtual_restore use the unified function
4. Keys match between simulation and runtime after kernel restart

This was the root cause of disk-cached entries not being restored after
kernel restart: the simulation's cache key diverged from runtime's because
module_component handling was inconsistent across the 4 call sites.
The fix unifies all cache key computation in cash.notebook.cache_key.
"""
import hashlib
import unittest
from unittest.mock import MagicMock

from cash.notebook.cache_key import compute_cache_key, CacheKeyContext
from cash.notebook.upstream import UpstreamChecker
from cash.notebook._protocols import TrackingState


class TestComputeCacheKey(unittest.TestCase):
    """Direct tests for the unified compute_cache_key function."""

    def test_basic_key_computation(self):
        """Simple statement with no modules or functions."""
        code = "y = x + 1"
        x_lineage = hashlib.sha256(b"x_val").hexdigest()

        key, source_hash, input_hashes, func_hashes, mod_hashes = compute_cache_key(
            code,
            {'x'},
            ctx=CacheKeyContext(variable_lineage={'x': x_lineage}, user_ns={'x': 42}),
        )

        self.assertTrue(key.startswith("stmt:"))
        self.assertEqual(input_hashes, [x_lineage])
        self.assertEqual(func_hashes, [])
        self.assertEqual(mod_hashes, [])

    def test_module_included_when_in_variable_lineage(self):
        """Modules in variable_lineage get included in module_component."""
        import numpy as np

        code = "result = np.mean(df)"
        df_lineage = hashlib.sha256(b"df_data").hexdigest()
        np_lineage = hashlib.sha256(b"np_module").hexdigest()

        key, _, input_hashes, _, mod_hashes = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={'np': np, 'df': [1, 2, 3]}),
        )

        # np is a module and IS in variable_lineage -> included in module_component
        self.assertEqual(input_hashes, [df_lineage])  # Only non-module
        self.assertEqual(mod_hashes, [f"np:{np_lineage}"])

    def test_module_excluded_when_not_in_variable_lineage(self):
        """Modules NOT in variable_lineage are excluded from module_component."""
        import numpy as np

        code = "result = np.mean(df)"
        df_lineage = hashlib.sha256(b"df_data").hexdigest()

        key, _, input_hashes, _, mod_hashes = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage}, user_ns={'np': np, 'df': [1, 2, 3]}),
        )

        # np is a module but NOT in variable_lineage -> excluded
        self.assertEqual(input_hashes, [df_lineage])
        self.assertEqual(mod_hashes, [])

    def test_virtual_lineage_fallback_for_non_modules(self):
        """For non-module inputs, virtual_lineage is used as fallback."""
        code = "y = x + 1"
        x_lineage = hashlib.sha256(b"x_val").hexdigest()

        key1, _, _, _, _ = compute_cache_key(
            code,
            {'x'},
            ctx=CacheKeyContext(variable_lineage={}, user_ns={}, virtual_lineage={'x': x_lineage}),
        )

        key2, _, _, _, _ = compute_cache_key(
            code,
            {'x'},
            ctx=CacheKeyContext(variable_lineage={'x': x_lineage}, user_ns={}),
        )

        # Both should produce the same key
        self.assertEqual(key1, key2)

    def test_virtual_modules_detected(self):
        """Variables in virtual_modules are treated as modules."""
        code = "result = np.mean(df)"
        df_lineage = hashlib.sha256(b"df_data").hexdigest()
        np_lineage = hashlib.sha256(b"np_module").hexdigest()

        key, _, input_hashes, _, mod_hashes = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={}, virtual_modules={'np'}),
        )

        # np is in virtual_modules AND in variable_lineage -> included
        self.assertEqual(input_hashes, [df_lineage])
        self.assertEqual(mod_hashes, [f"np:{np_lineage}"])

    def test_ipython_internals_skipped(self):
        """get_ipython and __builtins__ are always skipped."""
        code = "result = x"
        x_lineage = hashlib.sha256(b"x_val").hexdigest()

        key, _, input_hashes, _, _ = compute_cache_key(
            code,
            {'x', 'get_ipython', '__builtins__'},
            ctx=CacheKeyContext(variable_lineage={'x': x_lineage}, user_ns={'x': 42}),
        )

        self.assertEqual(input_hashes, [x_lineage])

    def test_output_modules_for_import_statements(self):
        """For import statements, output module lineages are included."""
        import numpy as np

        code = "import numpy as np"
        np_lineage = hashlib.sha256(b"np_module").hexdigest()

        key, _, _, _, mod_hashes = compute_cache_key(
            code,
            set(),
            ctx=CacheKeyContext(variable_lineage={'np': np_lineage}, user_ns={'np': np}),
            outputs={'np'},
        )

        # Output module should be included
        self.assertIn(f"out:np:{np_lineage}", mod_hashes)

    def test_multiple_modules_sorted(self):
        """Multiple module inputs should be sorted in module_component."""
        import numpy as np
        import os

        code = "result = np.array(os.listdir('.'))"
        np_lineage = hashlib.sha256(b"np_lin").hexdigest()
        os_lineage = hashlib.sha256(b"os_lin").hexdigest()

        key, _, _, _, mod_hashes = compute_cache_key(
            code,
            {'np', 'os'},
            ctx=CacheKeyContext(variable_lineage={'np': np_lineage, 'os': os_lineage}, user_ns={'np': np, 'os': os}),
        )

        # Should be sorted by variable name
        self.assertEqual(mod_hashes, [f"np:{np_lineage}", f"os:{os_lineage}"])

    def test_identical_keys_across_call_patterns(self):
        """The same inputs must produce the same key regardless of call pattern.
        
        This is THE critical invariant: runtime (via _analyze_and_hash) and
        simulation (via _update_virtual_lineage) must produce identical keys.
        """
        import numpy as np

        code = "result = np.mean(df)"
        df_lineage = hashlib.sha256(b"df_data").hexdigest()
        np_lineage = hashlib.sha256(b"np_module").hexdigest()

        # Pattern 1: Runtime call (variable_lineage has everything)
        key_runtime, _, _, _, _ = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={'np': np, 'df': [1, 2, 3]}),
        )

        # Pattern 2: Simulation call (variable_lineage has module, virtual_lineage has df)
        key_sim, _, _, _, _ = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(
                variable_lineage={'np': np_lineage},
                user_ns={'np': np, 'df': [1, 2, 3]},
                virtual_lineage={'df': df_lineage, 'np': np_lineage},
                virtual_modules={'np'},
            ),
        )

        self.assertEqual(key_runtime, key_sim,
                         "Runtime and simulation must produce identical cache keys")


class TestModuleLineagePropagation(unittest.TestCase):
    """Verify that _update_virtual_lineage propagates module lineages."""

    def setUp(self):
        self.shell = MagicMock()
        self.shell.user_ns = {}
        mock_backend = MagicMock()
        mock_backend.get.return_value = (None, None)
        mock_backend.get_metadata.return_value = {'output_lineages': {}}
        self.cash_instance = MagicMock()
        self.cash_instance.backend = mock_backend
        self.checker = UpstreamChecker(
            self.shell,
            cash_instance=self.cash_instance,
            debug=True,
        )
        self.checker.set_tracking_state(TrackingState())

    def test_import_propagates_lineage(self):
        """Simulating an import statement should set variable_lineage for the module."""
        virtual_lineage = {}
        virtual_modules = set()

        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "import pandas as pd", virtual_lineage, virtual_modules
        )

        # pd should now be in both virtual_lineage AND variable_lineage
        self.assertIn('pd', virtual_lineage)
        self.assertIn('pd', virtual_modules)
        self.assertIn('pd', self.checker.variable_lineage)
        self.assertEqual(self.checker.variable_lineage['pd'], virtual_lineage['pd'])

    def test_from_import_propagates_lineage(self):
        """'from ... import' statements should propagate lineage."""
        virtual_lineage = {}
        virtual_modules = set()

        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "from numpy import array", virtual_lineage, virtual_modules
        )

        self.assertIn('array', virtual_lineage)
        self.assertIn('array', virtual_modules)
        # array is detected as module output -> propagated
        self.assertIn('array', self.checker.variable_lineage)

    def test_non_import_does_not_propagate(self):
        """Non-import statements should not propagate to variable_lineage."""
        virtual_lineage = {'x': 'abc123'}
        virtual_modules = set()

        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "y = x + 1", virtual_lineage, virtual_modules
        )

        # y should be in virtual_lineage but NOT in variable_lineage
        self.assertIn('y', virtual_lineage)
        self.assertNotIn('y', self.checker.variable_lineage)

    def test_existing_variable_lineage_not_overwritten(self):
        """If variable_lineage already has a module, don't overwrite it."""
        existing_lineage = "existing_lineage_hash"
        self.checker.variable_lineage['pd'] = existing_lineage

        virtual_lineage = {}
        virtual_modules = set()

        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "import pandas as pd", virtual_lineage, virtual_modules
        )

        # Should NOT be overwritten
        self.assertEqual(self.checker.variable_lineage['pd'], existing_lineage)


class TestSimulationRuntimeKeyMatch(unittest.TestCase):
    """End-to-end test: simulation cache key matches runtime key.
    
    This simulates the real flow:
    1. Import statement -> propagates module lineage
    2. Downstream statement -> includes module in cache key
    3. Runtime statement -> also includes module in cache key
    4. Both keys MUST match
    """

    def setUp(self):
        self.shell = MagicMock()
        self.shell.user_ns = {}
        mock_backend = MagicMock()
        mock_backend.get.return_value = (None, None)
        mock_backend.get_metadata.return_value = {'output_lineages': {}}
        self.cash_instance = MagicMock()
        self.cash_instance.backend = mock_backend
        self.checker = UpstreamChecker(
            self.shell,
            cash_instance=self.cash_instance,
            debug=True,
        )
        self.checker.set_tracking_state(TrackingState())

    def _get_cache_key_from_calls(self):
        """Extract the cache key from backend calls."""
        calls = self.cash_instance.backend.get_metadata.call_args_list
        for call in calls:
            args, kwargs = call
            if args and isinstance(args[0], str) and args[0].startswith('stmt:'):
                return args[0]
        calls = self.cash_instance.backend.get.call_args_list
        for call in calls:
            args, kwargs = call
            if args and isinstance(args[0], str) and args[0].startswith('stmt:'):
                return args[0]
        return None

    def test_full_flow_key_match(self):
        """Simulate full import->downstream flow, verify key matches runtime."""
        code = "df['VolAdj'] = df['Close'] * np.sqrt(df['Volume'])"
        df_lineage = "abc123def456"
        np_lineage_from_import = None  # Will be set by import simulation

        # Step 1: Simulate import (sets variable_lineage['np'])
        virtual_lineage = {}
        virtual_modules = set()
        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "import numpy as np", virtual_lineage, virtual_modules
        )
        np_lineage_from_import = self.checker.variable_lineage['np']

        # Step 2: Set up df lineage
        virtual_lineage['df'] = df_lineage

        # Step 3: Reset backend call tracking
        self.cash_instance.backend.get_metadata.reset_mock()
        self.cash_instance.backend.get.reset_mock()
        self.cash_instance.backend.get_metadata.return_value = {'output_lineages': {}}

        # Step 4: Simulate downstream statement
        self.checker.simulator._virtual_lineage._update_virtual_lineage(code, virtual_lineage, virtual_modules)
        sim_key = self._get_cache_key_from_calls()

        # Step 5: Compute what runtime (_analyze_and_hash) would produce
        # At runtime, variable_lineage has both df and np
        runtime_key, _, _, _, _ = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage_from_import}, user_ns={}, virtual_modules={'np'}),
        )

        self.assertEqual(sim_key, runtime_key,
                         f"Simulation key must match runtime key.\n"
                         f"Simulation: {sim_key}\n"
                         f"Runtime: {runtime_key}")

    def test_try_virtual_restore_key_matches(self):
        """_try_virtual_restore key must match _analyze_and_hash key."""
        code = "result = np.mean(df)"
        df_lineage = hashlib.sha256(b"df_data").hexdigest()
        np_lineage = hashlib.sha256(b"np_module").hexdigest()

        # Simulate import having been processed (lineage propagated)
        self.checker.variable_lineage['np'] = np_lineage

        inputs = {'np', 'df'}
        outputs = {'result'}
        input_hashes = {'df': df_lineage, 'np': np_lineage}
        virtual_modules = {'np'}

        # Set up backend to return valid data
        self.cash_instance.backend.get.return_value = (
            {'output_lineages': {'result': 'cached_lineage'}, 'execution_time': 5.0},
            {'variables': {'result': 42}}
        )

        restored, _, _ = self.checker.simulator._virtual_lineage._try_virtual_restore(
            code, outputs, inputs, input_hashes, virtual_modules
        )

        # Get the key used for lookup
        calls = self.cash_instance.backend.get.call_args_list
        restore_key = None
        for call in calls:
            args, kwargs = call
            if args and isinstance(args[0], str) and args[0].startswith('stmt:'):
                restore_key = args[0]
                break

        # Compute the runtime key
        runtime_key, _, _, _, _ = compute_cache_key(
            code,
            inputs,
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={}, virtual_modules=virtual_modules),
        )

        self.assertEqual(restore_key, runtime_key,
                         f"Virtual restore key must match runtime key.\n"
                         f"Restore: {restore_key}\n"
                         f"Runtime: {runtime_key}")

    def test_module_in_user_ns_detected(self):
        """When a module is in user_ns (not virtual_modules), isinstance detects it."""
        import numpy as np
        self.shell.user_ns = {'np': np, 'df': [1, 2, 3]}
        np_lineage = hashlib.sha256(b"np_module").hexdigest()
        df_lineage = hashlib.sha256(b"df_data").hexdigest()

        # np is in variable_lineage (from import simulation) and user_ns
        self.checker.variable_lineage['np'] = np_lineage

        code = "result = np.mean(df)"
        virtual_lineage = {'df': df_lineage, 'np': np_lineage}
        virtual_modules = set()  # np NOT in virtual_modules

        self.cash_instance.backend.get_metadata.reset_mock()
        self.cash_instance.backend.get_metadata.return_value = {'output_lineages': {}}

        self.checker.simulator._virtual_lineage._update_virtual_lineage(code, virtual_lineage, virtual_modules)

        sim_key = self._get_cache_key_from_calls()

        # np should be detected via isinstance and included in module_component
        runtime_key, _, _, _, mod_hashes = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={'np': np, 'df': [1, 2, 3]}),
        )
        self.assertIn(f"np:{np_lineage}", mod_hashes)
        self.assertEqual(sim_key, runtime_key)

    def test_empty_user_ns_after_restart(self):
        """After kernel restart, user_ns is empty. Module detection via virtual_modules."""
        self.shell.user_ns = {}

        # Step 1: Simulate import to propagate lineage
        virtual_lineage = {}
        virtual_modules = set()
        self.checker.simulator._virtual_lineage._update_virtual_lineage(
            "import numpy as np", virtual_lineage, virtual_modules
        )
        np_lineage = self.checker.variable_lineage['np']

        # Step 2: Set up df lineage
        df_lineage = hashlib.sha256(b"df_data").hexdigest()
        virtual_lineage['df'] = df_lineage

        # Step 3: Reset mocks
        self.cash_instance.backend.get_metadata.reset_mock()
        self.cash_instance.backend.get_metadata.return_value = {'output_lineages': {}}

        # Step 4: Simulate downstream statement
        code = "result = np.mean(df)"
        self.checker.simulator._virtual_lineage._update_virtual_lineage(code, virtual_lineage, virtual_modules)
        sim_key = self._get_cache_key_from_calls()

        # Step 5: Compute runtime key
        runtime_key, _, _, _, _ = compute_cache_key(
            code,
            {'np', 'df'},
            ctx=CacheKeyContext(variable_lineage={'df': df_lineage, 'np': np_lineage}, user_ns={}, virtual_modules={'np'}),
        )

        self.assertEqual(sim_key, runtime_key,
                         "After kernel restart with empty user_ns, keys must match")


if __name__ == '__main__':
    unittest.main()
