
# =============================================================================
# RNG State Capture Tests
# =============================================================================

from cash.notebook.randomness import (
    capture_rng_state,
    restore_rng_state,
    get_used_rng_modules
)

class TestRNGStateUtilities:
    """Tests for RNG state capture and restore functions."""
    
    def test_get_used_rng_modules(self):
        """Test detection of used RNG modules."""
        code = """
import random
import numpy as np
x = random.random()
y = np.random.rand(10)
"""
        modules = get_used_rng_modules(code)
        assert 'random' in modules
        assert 'numpy.random' in modules
    
    def test_capture_restore_random(self):
        """Test capture and restore of random module state."""
        import random
        
        # Set initial state
        random.seed(42)
        state_orig = capture_rng_state()
        val1 = random.random()
        
        # Change state
        random.seed(123)
        val_different = random.random()
        assert val1 != val_different
        
        # Restore state
        restore_rng_state(state_orig)
        val2 = random.random()
        
        # Should match original sequence
        assert val1 == val2
        
    def test_capture_restore_numpy(self):
        """Test capture and restore of numpy random state."""
        import numpy as np
        
        # Set initial state
        np.random.seed(42)
        state_orig = capture_rng_state()
        val1 = np.random.rand()
        
        # Change state
        np.random.seed(123)
        val_different = np.random.rand()
        assert val1 != val_different
        
        # Restore state
        restore_rng_state(state_orig)
        val2 = np.random.rand()
        
        # Should match original sequence (floating point comparison)
        np.testing.assert_almost_equal(val1, val2)
