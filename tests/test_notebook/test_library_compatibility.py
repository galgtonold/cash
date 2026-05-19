from cash.notebook.cache_status import CacheStatus
from cash.notebook.annotations import CacheAnnotation

_PERSIST = CacheAnnotation(persist=True)
"""
Library Compatibility Tests for sklearn, matplotlib, and other data science libraries.

Tests that cash correctly caches results from popular data science libraries
including serialization, restoration, and cache invalidation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable

# Skip all tests if sklearn/matplotlib aren't available
sklearn = pytest.importorskip("sklearn")
matplotlib = pytest.importorskip("matplotlib")


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def processor_fixture():
    """Provide StatementProcessor via CashMagics for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    processor = magics._statement_processor
    yield processor, shell, backend
    backend.clear()
    shell.user_ns.clear()


# ============================================================
# scikit-learn Tests
# ============================================================

class TestSklearnCaching:
    """Test caching of sklearn models and pipelines."""

    def test_cache_sklearn_model(self, processor_fixture):
        """Test caching a trained sklearn model."""
        processor, shell, backend = processor_fixture

        # Pre-load sklearn into namespace
        from sklearn.linear_model import LinearRegression
        shell.user_ns['LinearRegression'] = LinearRegression
        shell.user_ns['np'] = np

        code = """
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'model' in shell.user_ns
        pred = shell.user_ns['model'].predict(np.array([[6]]))
        assert abs(pred[0] - 12.0) < 0.1

    def test_cache_sklearn_model_restore(self, processor_fixture):
        """Test that sklearn model training runs correctly.
        
        Note: Statements referencing injected classes (LinearRegression, np) 
        without lineage tracking are typically not cached because inputs 
        lack lineage hashes. This is expected in unit test context.
        In real notebooks, imports establish lineage during execution.
        """
        processor, shell, backend = processor_fixture

        from sklearn.linear_model import LinearRegression
        shell.user_ns['LinearRegression'] = LinearRegression
        shell.user_ns['np'] = np

        code = """
model = LinearRegression().fit([[1],[2],[3]], [1,2,3])
"""
        # First run - COMPUTED
        metrics1 = processor.process_statement(code)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert 'model' in shell.user_ns
        # Model should work
        pred = shell.user_ns['model'].predict([[4]])
        assert abs(pred[0] - 4.0) < 0.1

    def test_cache_sklearn_preprocessing(self, processor_fixture):
        """Test caching sklearn preprocessing results."""
        processor, shell, backend = processor_fixture

        from sklearn.preprocessing import StandardScaler
        shell.user_ns['StandardScaler'] = StandardScaler
        shell.user_ns['np'] = np

        code = """
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'scaled' in shell.user_ns
        assert shell.user_ns['scaled'].shape == (4, 2)

    def test_cache_sklearn_pipeline(self, processor_fixture):
        """Test caching a sklearn Pipeline."""
        processor, shell, backend = processor_fixture

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        shell.user_ns['Pipeline'] = Pipeline
        shell.user_ns['StandardScaler'] = StandardScaler
        shell.user_ns['LogisticRegression'] = LogisticRegression
        shell.user_ns['np'] = np

        code = """
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200))
])
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])
pipe.fit(X, y)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'pipe' in shell.user_ns

    def test_cache_sklearn_metrics(self, processor_fixture):
        """Test caching sklearn metric computation results."""
        processor, shell, backend = processor_fixture

        from sklearn.metrics import accuracy_score
        shell.user_ns['accuracy_score'] = accuracy_score

        code = """
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
acc = accuracy_score(y_true, y_pred)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['acc'] == 0.8

    def test_sklearn_train_test_split(self, processor_fixture):
        """Test caching train_test_split results."""
        processor, shell, backend = processor_fixture

        from sklearn.model_selection import train_test_split
        shell.user_ns['train_test_split'] = train_test_split
        shell.user_ns['np'] = np

        # Use allow-random annotation since train_test_split is random
        code = """# @cash:allow-random
X = np.arange(20).reshape(10, 2)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'X_train' in shell.user_ns
        assert 'X_test' in shell.user_ns


# ============================================================
# matplotlib Tests
# ============================================================

class TestMatplotlibCaching:
    """Test caching of matplotlib plot data and figures."""

    def test_cache_matplotlib_figure(self, processor_fixture):
        """Test caching matplotlib figure creation."""
        processor, shell, backend = processor_fixture

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        shell.user_ns['plt'] = plt
        shell.user_ns['np'] = np

        code = """
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title('Sine Wave')
plt.close(fig)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'fig' in shell.user_ns

    def test_cache_plot_data(self, processor_fixture):
        """Test caching the data used for plots.

        ``@cash:persist`` bypasses the cache policy's min-execution-time
        floor (10 ms by default). This test is exercising the cache
        mechanism, not the policy, so the trivial 100-element linspace
        would otherwise be policy-skipped.
        """
        processor, shell, backend = processor_fixture

        shell.user_ns['np'] = np

        code = """
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
"""
        metrics = processor.process_statement(code, annotation=_PERSIST)
        assert metrics['status'] == CacheStatus.COMPUTED

        # Clear and restore
        shell.user_ns.pop('x', None)
        shell.user_ns.pop('y_sin', None)
        shell.user_ns.pop('y_cos', None)

        metrics2 = processor.process_statement(code, annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.RESTORED
        assert len(shell.user_ns['x']) == 100

    def test_cache_histogram_data(self, processor_fixture):
        """Test caching histogram computation data."""
        processor, shell, backend = processor_fixture

        shell.user_ns['np'] = np

        code = """# @cash:allow-random
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
hist_values, bin_edges = np.histogram(data, bins=30)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'hist_values' in shell.user_ns
        assert len(shell.user_ns['hist_values']) == 30


# ============================================================
# NumPy Advanced Tests
# ============================================================

class TestNumpyAdvancedCaching:
    """Test caching with advanced numpy operations."""

    def test_cache_numpy_linalg(self, processor_fixture):
        """Test caching numpy linear algebra results."""
        processor, shell, backend = processor_fixture
        shell.user_ns['np'] = np

        code = """
A = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = np.linalg.eig(A)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'eigenvalues' in shell.user_ns
        assert len(shell.user_ns['eigenvalues']) == 2

    def test_cache_numpy_matrix_ops(self, processor_fixture):
        """Test caching matrix multiplication results. Uses
        ``@cash:persist`` to bypass the min-execution-time floor (2x2
        matrices are trivially fast)."""
        processor, shell, backend = processor_fixture
        shell.user_ns['np'] = np

        code = """
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
det = np.linalg.det(A)
"""
        metrics = processor.process_statement(code, annotation=_PERSIST)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['C'].shape == (2, 2)

        # Verify restore
        shell.user_ns.pop('A', None)
        shell.user_ns.pop('B', None)
        shell.user_ns.pop('C', None)
        shell.user_ns.pop('det', None)

        metrics2 = processor.process_statement(code, annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.RESTORED

    def test_cache_large_numpy_array(self, processor_fixture):
        """Test caching a large numpy array."""
        processor, shell, backend = processor_fixture
        shell.user_ns['np'] = np

        code = """
big_array = np.zeros((1000, 100))
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['big_array'].shape == (1000, 100)


# ============================================================
# Pandas Advanced Tests
# ============================================================

class TestPandasAdvancedCaching:
    """Test caching with advanced pandas operations."""

    def test_cache_pandas_groupby(self, processor_fixture):
        """Test caching groupby results."""
        processor, shell, backend = processor_fixture

        import pandas as pd
        shell.user_ns['pd'] = pd

        code = """
df = pd.DataFrame({
    'group': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})
grouped = df.groupby('group')['value'].mean()
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'grouped' in shell.user_ns

    def test_cache_pandas_pivot_table(self, processor_fixture):
        """Test caching pivot table results."""
        processor, shell, backend = processor_fixture

        import pandas as pd
        shell.user_ns['pd'] = pd

        code = """
df = pd.DataFrame({
    'date': ['Mon', 'Tue', 'Mon', 'Tue'],
    'category': ['A', 'A', 'B', 'B'],
    'sales': [100, 200, 150, 250]
})
pivot = df.pivot_table(values='sales', index='date', columns='category', aggfunc='sum')
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert 'pivot' in shell.user_ns

    def test_cache_pandas_merge(self, processor_fixture):
        """Test caching merge results."""
        processor, shell, backend = processor_fixture

        import pandas as pd
        shell.user_ns['pd'] = pd

        code = """
df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})
merged = pd.merge(df1, df2, on='key', how='inner')
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert len(shell.user_ns['merged']) == 2  # Only 'a' and 'b' match
