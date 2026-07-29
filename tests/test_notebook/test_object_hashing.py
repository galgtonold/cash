"""Tests for `cash.notebook.object_hashing.compute_hash` — hash dispatch table.

Migrated from `test_magics_coverage.py` when the functions moved out of
`CashMagics` into their own module. The companion size-estimator
(``estimate_object_size``) is exercised by ``test_size_estimator.py``.
"""
import hashlib

import pytest

from cash.notebook.object_hashing import compute_hash, identity_hash, is_identity_fallback_hash


# ============================================================================
# compute_hash
# ============================================================================

class TestComputeHash:
    def test_hash_simple_objects(self):
        h1 = compute_hash(42)
        h2 = compute_hash(42)
        h3 = compute_hash(43)
        assert h1 == h2
        assert h1 != h3

    def test_hash_string(self):
        h = compute_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_list(self):
        h = compute_hash([1, 2, 3])
        assert isinstance(h, str)

    def test_hash_dict(self):
        h = compute_hash({"a": 1, "b": 2})
        assert isinstance(h, str)

    def test_hash_dataframe(self):
        """DataFrame should use fast shape+dtypes hashing."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        h = compute_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_numpy_array(self):
        """ndarray should use shape+dtype hashing."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.array([1, 2, 3, 4, 5])
        h = compute_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_series(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        s = pd.Series([1, 2, 3], name='test')
        h = compute_hash(s)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_unpicklable_object(self):
        """Unpicklable objects should fall back to id-based hash."""
        import threading
        lock = threading.Lock()
        h = compute_hash(lock)
        assert isinstance(h, str)

    def test_unpicklable_object_hash_is_the_identity_fallback(self):
        """Pins tier 3's exact formula (`sha256(str(id(obj)))`), not merely
        "some hash came back" -- so a callee like `CallUnit._hash_args`
        (CAS-243) that has to tell a real content hash apart from this
        id-based one via `is_identity_fallback_hash` cannot silently stop
        working if this formula ever changes without `is_identity_fallback_hash`
        changing to match.
        """
        import threading
        lock = threading.Lock()
        h = compute_hash(lock)
        assert h == hashlib.sha256(str(id(lock)).encode('utf-8')).hexdigest()
        assert h == identity_hash(lock)
        assert is_identity_fallback_hash(lock, h)

    def test_a_real_content_hash_is_not_flagged_as_identity_fallback(self):
        """Positive control: an ordinary, picklable object's `compute_hash`
        result must NOT be mistaken for the identity fallback -- otherwise
        every object would be (wrongly) treated as unprovable.
        """
        assert not is_identity_fallback_hash(42, compute_hash(42))
        assert not is_identity_fallback_hash([1, 2, 3], compute_hash([1, 2, 3]))

    def test_hash_large_list_samples(self):
        """Lists with >200 items hash by length + head/tail sample, not full pickle."""
        h = compute_hash(list(range(300)))
        # Same length but different head/tail must differ.
        assert h != compute_hash([-1] + list(range(1, 300)))
        # Identical large lists must match.
        assert h == compute_hash(list(range(300)))

    def test_hash_large_dict_samples(self):
        """Dicts with >200 keys hash by length + sorted-key sample (object_hashing 53-54)."""
        big = {i: i for i in range(300)}
        h = compute_hash(big)
        assert isinstance(h, str) and len(h) == 64
        assert h == compute_hash({i: i for i in range(300)})
        # A different key in the sampled prefix changes the hash.
        alt = {i: i for i in range(1, 301)}
        assert h != compute_hash(alt)

    def test_hash_large_set_samples(self):
        """Sets with >200 items hash by length + sorted sample, not full pickle."""
        h = compute_hash(set(range(300)))
        assert isinstance(h, str) and len(h) == 64
        assert h == compute_hash(set(range(300)))


# ============================================================================
# calculate_memory_size
# ============================================================================

