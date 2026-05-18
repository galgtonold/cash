import pytest

from benchmarks._object_generators import (
    FAMILIES,
    make_object,
    estimate_in_memory_size,
)


@pytest.mark.parametrize("family", [
    "dataframe_numeric", "series_numeric", "ndarray_dense",
    "dict_shallow", "list_flat", "bytes",
])
def test_make_object_produces_target_family(family):
    obj = make_object(family, target_bytes=10_000)
    assert obj is not None
    # Approximate size: within 5x of target on either side (generators are coarse).
    actual = estimate_in_memory_size(obj)
    assert 2_000 < actual < 50_000, (
        f"{family} produced object of {actual} bytes for target 10000"
    )


def test_make_object_unknown_family_raises():
    with pytest.raises(KeyError):
        make_object("unicorn", target_bytes=1000)


def test_sparse_family_optional():
    """Sparse family available only when scipy is installed; should either
    return a csr_matrix or raise ImportError. Test runs in both worlds."""
    try:
        import scipy.sparse  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            make_object("sparse", target_bytes=100_000)
    else:
        obj = make_object("sparse", target_bytes=100_000)
        assert type(obj).__name__ == "csr_matrix"


def test_families_list_includes_all_documented_families():
    expected = {
        "dataframe_numeric", "series_numeric", "ndarray_dense", "sparse",
        "dict_shallow", "list_flat", "bytes",
    }
    assert set(FAMILIES) == expected
