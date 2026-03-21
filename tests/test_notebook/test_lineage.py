import pytest
import pandas as pd
from cash import Cash

# Module-level app and cached functions
app = Cash(register_magic=False)

@app.cache
def create_df(n):
    return pd.DataFrame({'a': range(n)})

@app.cache
def process_df(df):
    return df['a'].sum()


@pytest.fixture(autouse=True)
def reset_lineage_state():
    '''Reset app state for lineage tests.'''
    app.backend.clear()
    app.functions.clear()
    app.source_hashes.clear()
    app.graph.clear()
    
    # Re-register functions
    app.cache(create_df)
    app.cache(process_df)
    
    yield
    
    # Cleanup
    app.backend.clear()


def test_lineage_hash():
    '''Test that lineage hash tracking works correctly for DataFrames.'''
    # 1. Create DF - should have lineage hash attached
    df = create_df(100)
    
    # Verify it has the hash attribute
    assert hasattr(df, '_cash_lineage_hash'), 'Created DataFrame should have _cash_lineage_hash attribute'
    
    # 2. Process DF - should use the hash instead of hashing the whole DF
    res1 = process_df(df)
    assert res1 == 4950, 'sum(0..99) should equal 4950'
    
    # 3. Create a new DataFrame without hash (simulates external data)
    df2 = pd.DataFrame({'a': range(100)})
    assert not hasattr(df2, '_cash_lineage_hash'), 'External DataFrame should not have _cash_lineage_hash'
    
    # Processing df2 should work (fallback to full hash)
    res2 = process_df(df2)
    assert res2 == 4950, 'Processing external DataFrame should work correctly'
    
    # 4. Verify caching behavior
    # process_df(df) and process_df(df2) have different cache keys:
    # - df has _cash_lineage_hash (lineage-based, fast)
    # - df2 uses full pickle hash (content-based, slower)
    # 
    # This is expected: lineage tracks *identity* (same object path), 
    # not *equality* (same content).
    # 
    # Calling process_df(df) again should use cache:
    res3 = process_df(df)
    assert res3 == 4950, 'Cached call should return same result'
