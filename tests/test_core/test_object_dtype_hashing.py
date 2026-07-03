"""CAS-111: object-dtype ndarray args must hash over content, not pointer bytes.

For ``dtype=object`` arrays the buffer holds raw PyObject pointers, so the old
``tobytes()`` path hashed memory addresses: identical content in fresh objects
never collided (permanent misses, cross-process-unstable) and address reuse
could alias distinct content onto one key. These assert content-based hashing.
"""
import numpy as np

from cash import Cash


class TestObjectDtypeArrayHashing:
    def test_identical_content_fresh_objects_share_key(self):
        a1 = np.array([{'k': 1}, {'k': 22}], dtype=object)
        a2 = np.array([{'k': 1}, {'k': 22}], dtype=object)  # fresh objects
        assert Cash._try_hash_numpy(a1) == Cash._try_hash_numpy(a2)

    def test_different_content_does_not_collide(self):
        a1 = np.array([{'k': 1}, {'k': 22}], dtype=object)
        a3 = np.array([{'k': 1}, {'k': 99}], dtype=object)
        assert Cash._try_hash_numpy(a1) != Cash._try_hash_numpy(a3)

    def test_string_object_array_content_hashed(self):
        s1 = np.array(['a', 'bb', 'ccc'], dtype=object)
        s2 = np.array(['a', 'bb', 'ccc'], dtype=object)
        s3 = np.array(['a', 'bb', 'cCc'], dtype=object)
        assert Cash._try_hash_numpy(s1) == Cash._try_hash_numpy(s2)
        assert Cash._try_hash_numpy(s1) != Cash._try_hash_numpy(s3)

    def test_object_array_with_nested_set_is_order_independent(self):
        # A set element pickles in PYTHONHASHSEED-dependent order; the stable
        # canonicalisation must make the two arrays share a key.
        a1 = np.array([{1, 2, 3}], dtype=object)
        a2 = np.array([{3, 2, 1}], dtype=object)
        assert Cash._try_hash_numpy(a1) == Cash._try_hash_numpy(a2)

    def test_numeric_arrays_unaffected(self):
        n1 = np.array([1.0, 2.0, 3.0])
        n2 = np.array([1.0, 2.0, 3.0])
        n3 = np.array([1.0, 2.0, 4.0])
        assert Cash._try_hash_numpy(n1) == Cash._try_hash_numpy(n2)
        assert Cash._try_hash_numpy(n1) != Cash._try_hash_numpy(n3)

    def test_end_to_end_cache_hit_on_identical_object_array(self):
        c = Cash()
        calls = []

        @c.cache
        def total(arr):
            calls.append(1)
            return sum(d['k'] for d in arr)

        a1 = np.array([{'k': 1}, {'k': 2}], dtype=object)
        a2 = np.array([{'k': 1}, {'k': 2}], dtype=object)  # identical content
        a3 = np.array([{'k': 1}, {'k': 9}], dtype=object)
        assert total(a1) == 3
        assert total(a2) == 3          # cache hit — no recompute
        assert total(a3) == 10         # different content — recompute
        assert len(calls) == 2, f"expected 2 computes, got {len(calls)}"
