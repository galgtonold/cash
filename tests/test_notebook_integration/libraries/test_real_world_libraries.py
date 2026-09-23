"""
Integration tests for real-world data science library workflows.

Tests patterns using:
1. scikit-learn: model training, pipeline, preprocessing
2. matplotlib: plotting workflows
3. Multi-library pipelines combining numpy, pandas, sklearn
"""

import pytest

pytestmark = pytest.mark.libraries


class TestSklearnPipeline:
    """Test caching with scikit-learn workflows."""

    def test_train_test_split_and_model(self, nb_runner):
        """Full sklearn workflow: split, train, predict, evaluate."""
        nb_runner.create_notebook(
            [
                # Cell 1: Create dataset
                (
                    "import numpy as np\n"
                    "np.random.seed(42)\n"
                    "X = np.random.randn(100, 3)\n"
                    "y = (X[:, 0] + X[:, 1] * 2 > 0).astype(int)\n"
                    "print(f'Dataset: {X.shape}, classes: {np.unique(y)}')"
                ),
                # Cell 2: Train/test split
                (
                    "from sklearn.model_selection import train_test_split\n"
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n"
                    "print(f'Train: {X_train.shape}, Test: {X_test.shape}')"
                ),
                # Cell 3: Train model
                (
                    "from sklearn.linear_model import LogisticRegression\n"
                    "model = LogisticRegression(random_state=42)\n"
                    "model.fit(X_train, y_train)\n"
                    "train_score = model.score(X_train, y_train)\n"
                    "print(f'Train accuracy: {train_score:.2f}')"
                ),
                # Cell 4: Evaluate
                ("test_score = model.score(X_test, y_test)\nprint(f'Test accuracy: {test_score:.2f}')"),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Dataset: (100, 3)" in nb_runner.get_output(1)
        assert "Train: (70, 3)" in nb_runner.get_output(2)
        assert "Train accuracy:" in nb_runner.get_output(3)
        assert "Test accuracy:" in nb_runner.get_output(4)

        # Second run should use cache
        nb_runner.run_all()
        assert "Train accuracy:" in nb_runner.get_output(3)

    def test_preprocessing_invalidation(self, nb_runner):
        """Changing preprocessing invalidates model results."""
        nb_runner.create_notebook(
            [
                # Cell 1: Data
                (
                    "import numpy as np\n"
                    "np.random.seed(1)\n"
                    "X = np.random.randn(60, 2)\n"
                    "y = (X[:, 0] > 0).astype(int)\n"
                    "print(f'Data ready: {X.shape}')"
                ),
                # Cell 2: Preprocess
                (
                    "from sklearn.preprocessing import StandardScaler\n"
                    "scaler = StandardScaler()\n"
                    "X_scaled = scaler.fit_transform(X)\n"
                    "print(f'Scaled mean: {X_scaled.mean(axis=0).round(4)}')"
                ),
                # Cell 3: Train
                (
                    "from sklearn.linear_model import LogisticRegression\n"
                    "model = LogisticRegression(random_state=1)\n"
                    "model.fit(X_scaled, y)\n"
                    "acc = model.score(X_scaled, y)\n"
                    "print(f'Accuracy: {acc:.2f}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output_before = nb_runner.get_output(3)
        assert "Accuracy:" in output_before

        # Change preprocessing - should invalidate model
        nb_runner.set_cell_source(
            2,
            (
                "from sklearn.preprocessing import MinMaxScaler\n"
                "scaler = MinMaxScaler()\n"
                "X_scaled = scaler.fit_transform(X)\n"
                "print(f'Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]')"
            ),
        )
        nb_runner.run_all()
        assert "Scaled range:" in nb_runner.get_output(2)
        assert "Accuracy:" in nb_runner.get_output(3)


class TestMatplotlibWorkflows:
    """Test caching with matplotlib plotting workflows."""

    def test_plot_data_computation(self, nb_runner):
        """Test that data computation for plots is cached (not the plot itself)."""
        nb_runner.create_notebook(
            [
                # Cell 1: Generate data
                (
                    "import numpy as np\n"
                    "np.random.seed(42)\n"
                    "x = np.linspace(0, 10, 100)\n"
                    "y = np.sin(x) + np.random.normal(0, 0.1, 100)\n"
                    "print(f'Data: x={len(x)}, y={len(y)}')"
                ),
                # Cell 2: Compute statistics for plot
                (
                    "y_smooth = np.convolve(y, np.ones(5)/5, mode='valid')\n"
                    "y_mean = y.mean()\n"
                    "y_std = y.std()\n"
                    "print(f'Stats: mean={y_mean:.3f}, std={y_std:.3f}, smooth_len={len(y_smooth)}')"
                ),
                # Cell 3: Summary
                ("peaks = np.sum(y > y_mean + y_std)\nprint(f'Points above 1 std: {peaks}')"),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Data: x=100, y=100" in nb_runner.get_output(1)
        assert "Stats:" in nb_runner.get_output(2)
        assert "Points above 1 std:" in nb_runner.get_output(3)

        # Re-run uses cache
        nb_runner.run_all()
        assert "Data: x=100, y=100" in nb_runner.get_output(1)


class TestMultiLibraryPipeline:
    """Test caching with multi-library data science pipelines."""

    def test_pandas_to_sklearn(self, nb_runner, tmp_path):
        """End-to-end: CSV → pandas → numpy → sklearn → results."""
        csv_path = tmp_path / "iris_subset.csv"
        csv_path.write_text(
            "sepal_length,sepal_width,petal_length,petal_width,species\n"
            "5.1,3.5,1.4,0.2,setosa\n"
            "4.9,3.0,1.4,0.2,setosa\n"
            "7.0,3.2,4.7,1.4,versicolor\n"
            "6.4,3.2,4.5,1.5,versicolor\n"
            "6.3,3.3,6.0,2.5,virginica\n"
            "5.8,2.7,5.1,1.9,virginica\n"
            "5.0,3.4,1.5,0.2,setosa\n"
            "6.7,3.1,4.4,1.4,versicolor\n"
            "7.1,3.0,5.9,2.1,virginica\n"
            "5.4,3.9,1.7,0.4,setosa\n",
            encoding="utf-8",
        )
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Load with pandas
                (
                    f"import pandas as pd\n"
                    f"df = pd.read_csv('{csv_str}')\n"
                    f"print(f'Loaded {{len(df)}} rows, columns: {{list(df.columns)}}')"
                ),
                # Cell 2: Prepare features
                (
                    "import numpy as np\n"
                    "feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n"
                    "X = df[feature_cols].values\n"
                    "species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}\n"
                    "y = df['species'].map(species_map).values\n"
                    "print(f'Features: {X.shape}, Labels: {np.unique(y)}')"
                ),
                # Cell 3: Train classifier and get feature importance
                (
                    "from sklearn.tree import DecisionTreeClassifier\n"
                    "clf = DecisionTreeClassifier(random_state=42, max_depth=3)\n"
                    "clf.fit(X, y)\n"
                    "accuracy = clf.score(X, y)\n"
                    "feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n"
                    "importances = dict(zip(feature_names, clf.feature_importances_))\n"
                    "most_important = max(importances, key=importances.get)\n"
                    "print(f'Training accuracy: {accuracy:.2f}')\n"
                    "print(f'Most important feature: {most_important}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Loaded 10 rows" in nb_runner.get_output(1)
        assert "Features: (10, 4)" in nb_runner.get_output(2)
        output3 = nb_runner.get_output(3)
        assert "Training accuracy:" in output3
        assert "Most important feature:" in output3

        # Second run should use cache
        nb_runner.run_all()
        assert "Training accuracy:" in nb_runner.get_output(3)

    def test_data_update_cascades(self, nb_runner, tmp_path):
        """Updating the CSV file should invalidate the entire pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')\nprint(f'Rows: {{len(df)}}')",
                "X = df[['a', 'b']].values\ny = df['label'].values\nprint(f'X shape: {X.shape}')",
                (
                    "from sklearn.linear_model import LogisticRegression\n"
                    "m = LogisticRegression(random_state=0)\n"
                    "m.fit(X, y)\n"
                    "print(f'Score: {m.score(X, y):.2f}')"
                ),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Rows: 4" in nb_runner.get_output(1)

        # Update CSV data
        csv_path.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n", encoding="utf-8")

        # Re-run - should detect file change and recompute
        nb_runner.run_all()
        assert "Rows: 6" in nb_runner.get_output(1)
