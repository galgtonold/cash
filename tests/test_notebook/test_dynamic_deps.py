import pytest
import os
import time
from cash import Cash, FileDataSource

# Module-level app and cached function
app = Cash(register_magic=False)

def file_resolver(filename):
    return FileDataSource(filename)

@app.cache(dynamic_depends_on=file_resolver)
def read_dynamic_file(filename):
    with open(filename, 'r') as f:
        return f.read()


@pytest.fixture(autouse=True)
def reset_and_cleanup_file():
    '''Reset app state and manage test file lifecycle.'''
    app.backend.clear()
    
    # Create test file
    with open('test_dyn.txt', 'w') as f:
        f.write('v1')
    
    yield
    
    # Cleanup
    app.backend.clear()
    if os.path.exists('test_dyn.txt'):
        os.remove('test_dyn.txt')


def test_dynamic_invalidation():
    '''Test that dynamic file dependencies trigger cache invalidation.'''
    # 1. First read - should execute and cache
    res1 = read_dynamic_file('test_dyn.txt')
    assert res1 == 'v1', 'First read should return initial content'
    
    # 2. Cached read - should use cache
    res2 = read_dynamic_file('test_dyn.txt')
    assert res2 == 'v1', 'Second read should use cached value'
    
    # 3. Modify file
    time.sleep(1.1)  # Ensure mtime change
    with open('test_dyn.txt', 'w') as f:
        f.write('v2')
        
    # 4. Should re-read because dynamic dependency changed
    res3 = read_dynamic_file('test_dyn.txt')
    assert res3 == 'v2', 'After file modification, should read new content (cache invalidated)'
