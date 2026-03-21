"""Tests for %cash_verify and %cash_repair magics."""


def test_cash_verify_empty_cache(cash_magics, mock_shell, cash_instance, capsys):
    """Test %cash_verify with empty cache."""
    cash_magics.cash_verify('')
    captured = capsys.readouterr()
    assert 'Checking cache integrity' in captured.out
    assert 'Total entries: 0' in captured.out
    assert 'All cache entries are healthy' in captured.out


def test_cash_verify_with_entries(cash_magics, mock_shell, cash_instance, capsys):
    """Test %cash_verify with cached entries."""
    backend = cash_instance.backend
    backend.set('test_key', b'test_value', {'func_name': 'test'})
    cash_magics.cash_verify('')
    captured = capsys.readouterr()
    assert 'Total entries: 1' in captured.out
    assert 'Healthy: 1' in captured.out


def test_cash_verify_lineage_check(cash_magics, mock_shell, cash_instance, capsys):
    """Test that %cash_verify reports stale lineage entries."""
    # Add lineage for a variable not in namespace
    cash_magics._tracking_state.variable_lineage['ghost_var'] = 'abc123'
    cash_magics.cash_verify('')
    captured = capsys.readouterr()
    assert 'Lineage' in captured.out
    assert '1' in captured.out  # 1 tracked variable not in namespace


def test_cash_repair_default(cash_magics, mock_shell, cash_instance, capsys):
    """Test %cash_repair default mode."""
    # Add stale lineage
    cash_magics._tracking_state.variable_lineage['stale_var'] = 'hash123'
    cash_magics._tracking_state.executed_cell_codes['stale_var'] = 'x = 1'
    
    cash_magics.cash_repair('')
    captured = capsys.readouterr()
    assert 'Repair complete' in captured.out
    assert 'stale_var' not in cash_magics._tracking_state.variable_lineage


def test_cash_repair_state(cash_magics, mock_shell, cash_instance, capsys):
    """Test %cash_repair --state clears in-memory state."""
    # Set up some state
    cash_magics._tracking_state.variable_lineage['x'] = 'hash1'
    cash_magics._tracking_state.executed_cell_codes['x'] = 'x = 1'
    mock_shell.user_ns['x'] = 1
    
    cash_magics.cash_repair('--state')
    captured = capsys.readouterr()
    assert 'State repair' in captured.out
    assert len(cash_magics._tracking_state.variable_lineage) == 0
    assert len(cash_magics._tracking_state.executed_cell_codes) == 0
    # Namespace should be untouched
    assert mock_shell.user_ns['x'] == 1


def test_cash_repair_full(cash_magics, mock_shell, cash_instance, capsys):
    """Test %cash_repair --full clears everything."""
    backend = cash_instance.backend
    backend.set('test_key', b'test_value', {'func_name': 'test'})
    cash_magics._tracking_state.variable_lineage['x'] = 'hash1'
    
    cash_magics.cash_repair('--full')
    captured = capsys.readouterr()
    assert 'Full repair' in captured.out
    assert len(cash_magics._tracking_state.variable_lineage) == 0
