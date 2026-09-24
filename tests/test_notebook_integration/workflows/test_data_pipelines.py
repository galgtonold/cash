"""Multi-step data pipelines and ETL notebooks."""

import textwrap
import time

import pytest


# Real-world data science patterns, kernel restart scenarios,
# out-of-order execution, annotation directives, and complex mutation patterns.
#
# These tests focus on realistic notebook workflows that data scientists
# commonly use, including pandas transformations, numpy array operations,
# and iterative refinement patterns.
@pytest.mark.core
class TestPandasPipelinePatterns:
    """Test realistic pandas workflows across cells."""

    def test_load_transform_aggregate(self, nb_runner, tmp_path):
        """Classic ETL: load CSV, transform, aggregate."""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("product,qty,price\nA,10,1.5\nB,5,3.0\nA,8,1.5\nB,12,3.0\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
                "df['total'] = df['qty'] * df['price']",
                "summary = df.groupby('product')['total'].sum().to_dict()\nprint(summary)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "'A': 27.0" in out, f"Got: {out}"
        assert "'B': 51.0" in out, f"Got: {out}"

    def test_dataframe_filtering_chain(self, nb_runner, tmp_path):
        """Chain of DataFrame filters across cells."""
        csv_path = tmp_path / "people.csv"
        csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,NYC\nDiana,28,LA\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
                "nyc_df = df[df['city'] == 'NYC']",
                "nyc_over_30 = nyc_df[nyc_df['age'] >= 30]\nprint(nyc_over_30['name'].tolist())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Alice" in out, f"Got: {out}"
        assert "Charlie" in out, f"Got: {out}"

    def test_csv_modification_detected(self, nb_runner, tmp_path):
        """Modify the CSV file between runs — should re-compute."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
                "total = df['val'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify the CSV
        csv_path.write_text("val\n100\n200\n300\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "total = 600" in out, f"Expected total=600, got: {out}"


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDataTransformWorkflow:
    """Data transformation workflows."""

    def test_filter_and_aggregate(self, nb_runner):
        """Filter data then aggregate, edit the filter."""
        nb_runner.create_notebook(
            [
                "records = [('A', 10), ('B', 20), ('A', 30), ('B', 40)]",
                "filtered = [v for k, v in records if k == 'A']",
                "total = sum(filtered)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 40" in nb_runner.get_output(3)

        # Change filter
        nb_runner.set_cell_source(2, "filtered = [v for k, v in records if k == 'B']")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)

    def test_sort_and_rank(self, nb_runner):
        """Sort data and compute ranks, edit sorting order."""
        nb_runner.create_notebook(
            [
                "scores = [85, 92, 78, 95, 88]  # student scores",
                "ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)",
                "top = ranked[0]\nprint(f'top student={top[0]} score={top[1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top student=3 score=95" in nb_runner.get_output(3)

        # Change scores
        nb_runner.set_cell_source(1, "scores = [50, 99, 78, 60, 88]  # student scores updated")
        nb_runner.run_all()
        assert "top student=1 score=99" in nb_runner.get_output(3)


# Data pipeline chain interaction tests.
#
# Tests editing cells in multi-stage data pipelines
# where each stage transforms the data.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestPipelineChainEdits:
    """Editing multi-stage pipeline patterns."""

    def test_edit_pipeline_source(self, nb_runner):
        """Edit source data in a 3-stage pipeline."""
        nb_runner.create_notebook(
            [
                "raw = [1, -2, 3, -4, 5, -6]",
                "positives = [x for x in raw if x > 0]\ndoubled = [x * 2 for x in positives]\nresult = sum(doubled)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(2)

        # Change source
        nb_runner.set_cell_source(1, "raw = [10, -1, 20, -2]")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    def test_edit_pipeline_middle_stage(self, nb_runner):
        """Edit a middle stage of the pipeline."""
        nb_runner.create_notebook(
            [
                "data = ['  Alice  ', '  Bob  ', '  Charlie  ']",
                "cleaned = [s.strip() for s in data]\nresult = [s.lower() for s in cleaned]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['alice', 'bob', 'charlie']" in nb_runner.get_output(2)

        # Change middle stage to uppercase
        nb_runner.set_cell_source(
            2, "cleaned = [s.strip() for s in data]\nresult = [s.upper() for s in cleaned]\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = ['ALICE', 'BOB', 'CHARLIE']" in nb_runner.get_output(2)

    def test_edit_pipeline_aggregation(self, nb_runner):
        """Edit the final aggregation step."""
        nb_runner.create_notebook(
            [
                "sales = [100, 200, 150, 300, 250]",
                "above_avg = [s for s in sales if s > sum(sales)/len(sales)]\ntotal = sum(above_avg)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 550" in nb_runner.get_output(2)

        # Change sales
        nb_runner.set_cell_source(1, "sales = [500, 100, 200, 600]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "total = 1100" in out

    def test_edit_pipeline_filter_and_source(self, nb_runner):
        """Edit both source and filter in pipeline."""
        nb_runner.create_notebook(
            [
                "records = [('A', 10), ('B', 20), ('C', 30), ('D', 40)]",
                "filtered = [(k, v) for k, v in records if v > 15]\nkeys = [k for k, v in filtered]\nprint(f'keys = {keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys = ['B', 'C', 'D']" in nb_runner.get_output(2)

        # Change records
        nb_runner.set_cell_source(1, "records = [('X', 5), ('Y', 50), ('Z', 25)]")
        nb_runner.run_all()
        assert "keys = ['Y', 'Z']" in nb_runner.get_output(2)


# Complex multi-step data pipeline patterns.
#
# Tests multi-cell data transformation pipelines with edits at different stages.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataPipelineEdits:
    """Multi-step pipeline with edits at different stages."""

    def test_filter_transform_aggregate(self, nb_runner):
        """Three-stage pipeline: filter → transform → aggregate."""
        nb_runner.create_notebook(
            [
                "raw = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]",
                "filtered = [x for x in raw if x > 0]",
                "transformed = [x ** 2 for x in filtered]",
                "total = sum(transformed)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+9+25+49+81 = 165
        assert "total = 165" in nb_runner.get_output(4)

        # Edit filter stage
        nb_runner.set_cell_source(2, "filtered = [x for x in raw if x < 0]")
        nb_runner.run_all()
        # (-2)^2+(-4)^2+(-6)^2+(-8)^2+(-10)^2 = 4+16+36+64+100 = 220
        assert "total = 220" in nb_runner.get_output(4)

    def test_edit_transform_stage(self, nb_runner):
        """Edit the transformation in the middle of a pipeline."""
        nb_runner.create_notebook(
            [
                "prices = [10.0, 20.0, 30.0, 40.0]",
                "discounted = [p * 0.9 for p in prices]",
                "with_tax = [p * 1.1 for p in discounted]",
                "total = round(sum(with_tax), 2)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*.9*1.1 + 20*.9*1.1 + ... = 9.9+19.8+29.7+39.6 = 99.0
        assert "total = 99.0" in nb_runner.get_output(4)

        # Change discount rate
        nb_runner.set_cell_source(2, "discounted = [p * 0.5 for p in prices]")
        nb_runner.run_all()
        # 10*.5*1.1 + 20*.5*1.1 + 30*.5*1.1 + 40*.5*1.1 = 5.5+11.0+16.5+22.0 = 55.0
        assert "total = 55.0" in nb_runner.get_output(4)

    def test_dict_pipeline_edit(self, nb_runner):
        """Dict-based pipeline: lookup → transform → format."""
        nb_runner.create_notebook(
            [
                "inventory = {'apple': 50, 'banana': 30, 'cherry': 20}",
                "threshold = 25",
                "low_stock = {k: v for k, v in inventory.items() if v <= threshold}",
                "report = ', '.join(f'{k}:{v}' for k, v in sorted(low_stock.items()))\nprint(f'report = {report}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "report = cherry:20" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "threshold = 35")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "banana:30" in out
        assert "cherry:20" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataPipelineSteps:
    """multi-step data pipeline with intermediate transforms."""

    def test_pipeline_3_steps(self, nb_runner):
        nb_runner.create_notebook(
            [
                "raw = [' Alice:85 ', ' Bob:92 ', ' Charlie:78 ']",
                "cleaned = [s.strip() for s in raw]",
                "parsed = [{'name': s.split(':')[0], 'score': int(s.split(':')[1])} for s in cleaned]",
                "avg = sum(p['score'] for p in parsed) / len(parsed)\nprint(f'avg={round(avg, 1)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=85.0" in nb_runner.get_output(4)

    def test_pipeline_edit_input(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [1, -2, 3, -4, 5]",
                "positives = [x for x in data if x > 0]",
                "doubled = [x * 2 for x in positives]\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=[2, 6, 10]" in nb_runner.get_output(3)
        # Edit
        nb_runner.set_cell_source(1, "data = [-10, 20, -30, 40]")
        nb_runner.run_all()
        assert "doubled=[40, 80]" in nb_runner.get_output(3)

    def test_pipeline_edit_middle(self, nb_runner):
        nb_runner.create_notebook(
            [
                "numbers = [10, 20, 30, 40, 50]",
                "transformed = [x + 5 for x in numbers]",
                "total = sum(transformed)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=175" in nb_runner.get_output(3)
        # Edit middle step
        nb_runner.set_cell_source(2, "transformed = [x * 2 for x in numbers]")
        nb_runner.run_all()
        assert "total=300" in nb_runner.get_output(3)


# Multi-cell data pipeline interaction tests.
# Tests complex data transformations spanning multiple cells where
# edits at different pipeline stages propagate correctly.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMultiCellPipelineInteraction:
    """Test multi-cell pipeline patterns with cache invalidation."""

    def test_etl_pipeline_edit_extract(self, nb_runner):
        """Editing the extract stage should propagate through transform and load."""
        nb_runner.create_notebook(
            [
                "# Extract\nraw_data = [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]",
                "# Transform\nfiltered = [d for d in raw_data if d['score'] >= 90]",
                "# Load\nresult = ', '.join(d['name'] for d in filtered)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=Bob" in out

        nb_runner.set_cell_source(
            1,
            "# Extract\nraw_data = [{'name': 'Alice', 'score': 95}, {'name': 'Bob', 'score': 92}, {'name': 'Charlie', 'score': 98}]",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Alice" in out
        assert "Bob" in out
        assert "Charlie" in out

    def test_etl_pipeline_edit_transform(self, nb_runner):
        """Editing the transform stage should propagate to load only."""
        nb_runner.create_notebook(
            [
                "raw = [10, 20, 30, 40, 50]",
                "# Transform: filter\nprocessed = [x for x in raw if x > 20]",
                "# Aggregate\ntotal = sum(processed)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=120" in out

        # Change transform threshold
        nb_runner.set_cell_source(2, "# Transform: filter\nprocessed = [x for x in raw if x > 35]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=90" in out

    def test_pipeline_three_stage_edit_middle(self, nb_runner):
        """Editing the middle of a 5-cell pipeline."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 11))",
                "squared = [x**2 for x in data]",
                "filtered = [x for x in squared if x > 25]",
                "total = sum(filtered)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        # squared: 1,4,9,16,25,36,49,64,81,100 → filtered: 36,49,64,81,100 → total=330
        assert "total=330" in out

        # Change to cubed instead of squared
        nb_runner.set_cell_source(2, "squared = [x**3 for x in data]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        # cubed: 1,8,27,64,125,216,343,512,729,1000 → filtered: 27,64,...,1000 → total=3016
        assert "total=3016" in out


# Grand finale: full end-to-end data science pipeline stress test.
@pytest.mark.stress
@pytest.mark.integration
class TestFullPipeline:
    """End-to-end data science pipeline: 8 cells, multiple dependencies."""

    def test_full_data_pipeline(self, nb_runner, tmp_path):
        """Complete pipeline: config → data gen → clean → feature eng → model → eval → report."""
        csv_path = str(tmp_path / "pipeline_data.csv").replace("\\", "/")
        nb_runner.create_notebook(
            [
                # Cell 1: Configuration
                textwrap.dedent("""\
                CONFIG = {
                    'seed': 42,
                    'n_samples': 50,
                    'train_ratio': 0.8,
                    'features': ['age', 'income', 'score'],
                }
            """),
                # Cell 2: Data generation
                textwrap.dedent(f"""\
                import random, csv
                random.seed(CONFIG['seed'])
                rows = []
                for i in range(CONFIG['n_samples']):
                    age = random.randint(18, 65)
                    income = random.randint(20000, 120000)
                    score = random.randint(300, 850)
                    target = 1 if (income > 60000 and score > 600) else 0
                    rows.append({{'id': i, 'age': age, 'income': income, 'score': score, 'target': target}})
                # Write to CSV
                with open('{csv_path}', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['id', 'age', 'income', 'score', 'target'])
                    w.writeheader()
                    w.writerows(rows)
                n_generated = len(rows)
            """),
                # Cell 3: Data loading and cleaning
                textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    data = list(csv.DictReader(f))
                # Convert types
                for row in data:
                    for k in ['id', 'age', 'income', 'score', 'target']:
                        row[k] = int(row[k])
                n_loaded = len(data)
            """),
                # Cell 4: Feature engineering
                textwrap.dedent("""\
                for row in data:
                    row['income_bucket'] = 'high' if row['income'] > 80000 else 'mid' if row['income'] > 40000 else 'low'
                    row['age_group'] = 'young' if row['age'] < 30 else 'mid' if row['age'] < 50 else 'senior'
                    row['score_norm'] = round((row['score'] - 300) / 550, 3)
                feature_cols = CONFIG['features'] + ['score_norm']
            """),
                # Cell 5: Train/test split
                textwrap.dedent("""\
                import random
                random.seed(CONFIG['seed'])
                indices = list(range(len(data)))
                random.shuffle(indices)
                split = int(len(data) * CONFIG['train_ratio'])
                train_idx = indices[:split]
                test_idx = indices[split:]
                train = [data[i] for i in train_idx]
                test = [data[i] for i in test_idx]
            """),
                # Cell 6: Simple model (majority vote per income_bucket)
                textwrap.dedent("""\
                from collections import Counter
                bucket_votes = {}
                for row in train:
                    bucket = row['income_bucket']
                    if bucket not in bucket_votes:
                        bucket_votes[bucket] = []
                    bucket_votes[bucket].append(row['target'])
                model = {}
                for bucket, targets in bucket_votes.items():
                    c = Counter(targets)
                    model[bucket] = c.most_common(1)[0][0]
            """),
                # Cell 7: Evaluation
                textwrap.dedent("""\
                correct = 0
                for row in test:
                    pred = model.get(row['income_bucket'], 0)
                    if pred == row['target']:
                        correct += 1
                accuracy = round(correct / len(test) * 100, 1) if test else 0
                n_train = len(train)
                n_test = len(test)
            """),
                # Cell 8: Report
                textwrap.dedent("""\
                report = f"Pipeline Report:\\n"
                report += f"  Generated: {n_generated} samples\\n"
                report += f"  Loaded: {n_loaded} samples\\n"
                report += f"  Train: {n_train}, Test: {n_test}\\n"
                report += f"  Model rules: {model}\\n"
                report += f"  Accuracy: {accuracy}%\\n"
                report += f"  Features: {feature_cols}"
                print(report)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(8)
        assert "Pipeline Report" in out
        assert "Generated: 50" in out
        assert "Loaded: 50" in out
        assert "Train: 40" in out
        assert "Test: 10" in out
        assert "Accuracy:" in out
        assert "score_norm" in out

    def test_pipeline_config_change(self, nb_runner, tmp_path):
        """Change config upstream, verify entire pipeline updates."""
        csv_path = str(tmp_path / "pipeline2.csv").replace("\\", "/")
        nb_runner.create_notebook(
            [
                # Cell 1: Config
                "N = 30",
                # Cell 2: Generate
                textwrap.dedent(f"""\
                import random, csv
                random.seed(0)
                rows = [{{'x': random.gauss(0, 1), 'y': random.gauss(0, 1)}} for _ in range(N)]
                with open('{csv_path}', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['x', 'y'])
                    w.writeheader()
                    w.writerows(rows)
            """),
                # Cell 3: Analyze
                textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    loaded = list(csv.DictReader(f))
                count = len(loaded)
                mean_x = round(sum(float(r['x']) for r in loaded) / count, 4)
            """),
                # Cell 4: Report
                "print(f'count={count} mean_x={mean_x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "count=30" in out1

        nb_runner.set_cell_source(1, "N = 100")
        nb_runner.run_cells([1, 2, 3, 4])
        out2 = nb_runner.get_output(4)
        assert "count=100" in out2

    def test_multi_branch_pipeline(self, nb_runner):
        """Pipeline with branching and merging: 6 cells, diamond dependency."""
        nb_runner.create_notebook(
            [
                # Cell 1: Source data
                textwrap.dedent("""\
                import random
                random.seed(123)
                raw = [random.randint(1, 100) for _ in range(50)]
            """),
                # Cell 2: Branch A – statistics
                textwrap.dedent("""\
                mean_val = round(sum(raw) / len(raw), 2)
                median_val = sorted(raw)[len(raw) // 2]
                std_val = round((sum((x - mean_val)**2 for x in raw) / len(raw))**0.5, 2)
            """),
                # Cell 3: Branch B – categorization
                textwrap.dedent("""\
                categories = {'low': 0, 'mid': 0, 'high': 0}
                for v in raw:
                    if v < 33:
                        categories['low'] += 1
                    elif v < 67:
                        categories['mid'] += 1
                    else:
                        categories['high'] += 1
            """),
                # Cell 4: Branch C – top/bottom
                textwrap.dedent("""\
                top5 = sorted(raw, reverse=True)[:5]
                bottom5 = sorted(raw)[:5]
            """),
                # Cell 5: Merge all branches
                textwrap.dedent("""\
                summary = {
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'distribution': categories,
                    'top5': top5,
                    'bottom5': bottom5,
                    'total': len(raw),
                }
            """),
                # Cell 6: Report
                textwrap.dedent("""\
                print(f"Total: {summary['total']}")
                print(f"Mean: {summary['mean']}, Median: {summary['median']}")
                print(f"Distribution: {summary['distribution']}")
                print(f"Top 5: {summary['top5']}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(6)
        assert "Total: 50" in out
        assert "Mean:" in out
        assert "Distribution:" in out
        assert "Top 5:" in out


# complex multi-cell data pipelines with many dependencies.
@pytest.mark.stress
@pytest.mark.integration
class TestMultiCellPipeline:
    """Complex multi-cell workflows testing dependency chains."""

    def test_branching_pipeline(self, nb_runner):
        """Two branches from same source, merged at the end."""
        nb_runner.create_notebook(
            [
                # Cell 1: Source
                "data = list(range(1, 21))",
                # Cell 2: Branch A – evens
                "evens = [x for x in data if x % 2 == 0]",
                # Cell 3: Branch B – odds
                "odds = [x for x in data if x % 2 != 0]",
                # Cell 4: Merge
                textwrap.dedent("""\
                even_sum = sum(evens)
                odd_sum = sum(odds)
                ratio = round(even_sum / odd_sum, 3)
            """),
                "print(f'even_sum={even_sum} odd_sum={odd_sum} ratio={ratio}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "even_sum=110" in out
        assert "odd_sum=100" in out

    def test_pipeline_mid_change(self, nb_runner):
        """Change a middle cell in a 4-cell pipeline, verify downstream updates."""
        nb_runner.create_notebook(
            [
                "numbers = [1, 2, 3, 4, 5]",
                "doubled = [x * 2 for x in numbers]",
                "total = sum(doubled)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=30" in nb_runner.get_output(4)

        # Change the middle transformation
        nb_runner.set_cell_source(2, "doubled = [x ** 2 for x in numbers]")
        nb_runner.run_cells([2, 3, 4])
        assert "total=55" in nb_runner.get_output(4)  # 1+4+9+16+25

    def test_diamond_dependency(self, nb_runner):
        """Diamond data dependency: A → B,C → D."""
        nb_runner.create_notebook(
            [
                # A: source
                "base = 100",
                # B: depends on A
                "tax = base * 0.08",
                # C: depends on A
                "discount = base * 0.10",
                # D: depends on B and C
                "final = base + tax - discount",
                "print(f'final={final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 100 + 8 - 10 = 98
        assert "final=98.0" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "base = 200")
        nb_runner.run_cells([1, 2, 3, 4, 5])
        # 200 + 16 - 20 = 196
        assert "final=196.0" in nb_runner.get_output(5)


# Complex data transformations — cash caching with multi-step reshaping.
@pytest.mark.stress
class TestListTransforms:
    """Test complex list transformation patterns."""

    def test_group_by(self, nb_runner):
        """Group-by operation across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from collections import defaultdict

                records = [
                    ('Engineering', 'Alice', 90000),
                    ('Engineering', 'Bob', 85000),
                    ('Sales', 'Charlie', 70000),
                    ('Sales', 'Diana', 75000),
                    ('Engineering', 'Eve', 95000),
                    ('Marketing', 'Frank', 65000),
                ]

                grouped = defaultdict(list)
                for dept, name, salary in records:
                    grouped[dept].append((name, salary))
            """),
                textwrap.dedent("""\
                for dept in sorted(grouped.keys()):
                    members = grouped[dept]
                    avg_salary = sum(s for _, s in members) / len(members)
                    print(f"{dept}: {len(members)} employees, avg=${avg_salary:,.0f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Engineering: 3 employees" in out
        assert "Sales: 2 employees" in out
        assert "Marketing: 1 employees" in out

    def test_transform_propagation(self, nb_runner):
        """Transform propagates when input changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                raw = [1, 2, 3, 4, 5]
                step1 = [x * 2 for x in raw]
            """),
                textwrap.dedent("""\
                step2 = [x + 10 for x in step1]
                print(f"result={step2}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[12, 14, 16, 18, 20]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            raw = [10, 20, 30]
            step1 = [x * 2 for x in raw]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "result=[30, 50, 70]" in nb_runner.get_output(2)


# Multi-step data transformation pipelines — realistic ETL-like workflows
# with many intermediate variables and complex data flow.
@pytest.mark.integration
@pytest.mark.stress
class TestETLPipeline:
    """Test caching with ETL-like transformation pipelines."""

    def test_extract_transform_load(self, nb_runner, tmp_path):
        """Full ETL pipeline: extract from CSV, transform, write output."""
        input_csv = tmp_path / "raw.csv"
        output_csv = tmp_path / "clean.csv"
        input_csv.write_text(
            "id,name,value,category\n1,Alice,100,A\n2,Bob,-5,B\n3,Charlie,200,A\n4,Diana,150,B\n5,Eve,-10,A\n",
            encoding="utf-8",
        )
        in_str = str(input_csv).replace("\\", "/")
        out_str = str(output_csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                # Extract
                f"raw = pd.read_csv('{in_str}')",
                # Validate
                textwrap.dedent("""\
                valid = raw[raw['value'] > 0].copy()
                print(f"valid rows: {len(valid)}")
            """),
                # Transform (explicit reassignment so cash tracks lineage)
                textwrap.dedent("""\
                valid = valid.assign(
                    normalized=(valid['value'] - valid['value'].mean()) / valid['value'].std()
                )
            """),
                # Aggregate
                textwrap.dedent("""\
                summary = valid.groupby('category').agg(
                    count=('id', 'count'),
                    avg_value=('value', 'mean')
                ).reset_index()
                print(summary.to_string(index=False))
            """),
                # Load
                textwrap.dedent(f"""\
                valid.to_csv('{out_str}', index=False)
                print(f"saved {{len(valid)}} rows")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid rows: 3" in nb_runner.get_output(3)
        assert "saved 3 rows" in nb_runner.get_output(6)

    def test_etl_modify_filter_and_rerun(self, nb_runner, tmp_path):
        """Modify filter criteria in ETL and re-run."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                "filtered = df[df['x'] > 2]",
                textwrap.dedent("""\
                result = filtered['y'].sum()
                print(f"sum={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x>2: y=30+40+50=120
        assert "sum=120" in nb_runner.get_output(4)

        # Change filter
        nb_runner.set_cell_source(3, "filtered = df[df['x'] > 3]")
        nb_runner.run_all()
        # x>3: y=40+50=90
        assert "sum=90" in nb_runner.get_output(4)


# Complex multi-cell ETL pipeline — cash caching with realistic data transforms.
@pytest.mark.stress
class TestETLPipelineComplex:
    """Test complex ETL pipeline spanning multiple cells."""

    def test_extract_transform_load(self, nb_runner, tmp_path):
        """Full ETL pipeline: extract CSV → transform → aggregate → report."""
        data_dir = tmp_path / "etl_data"
        data_dir.mkdir()
        csv_file = data_dir / "sales.csv"
        csv_file.write_text(
            "date,product,qty,price\n"
            "2024-01-01,Widget,10,9.99\n"
            "2024-01-01,Gadget,5,24.99\n"
            "2024-01-02,Widget,8,9.99\n"
            "2024-01-02,Gadget,12,24.99\n"
            "2024-01-03,Widget,15,9.99\n",
            encoding="utf-8",
        )
        fpath = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Extract
                textwrap.dedent(f"""\
                import csv
                with open('{fpath}', 'r') as f:
                    reader = csv.DictReader(f)
                    raw_data = [row for row in reader]
                print(f"extracted={{len(raw_data)}} rows")
            """),
                # Cell 2: Transform
                textwrap.dedent("""\
                transformed = []
                for row in raw_data:
                    transformed.append({
                        'date': row['date'],
                        'product': row['product'],
                        'qty': int(row['qty']),
                        'price': float(row['price']),
                        'revenue': int(row['qty']) * float(row['price']),
                    })
                print(f"transformed={len(transformed)} rows")
            """),
                # Cell 3: Aggregate
                textwrap.dedent("""\
                from collections import defaultdict
                product_totals = defaultdict(lambda: {'qty': 0, 'revenue': 0.0})
                for row in transformed:
                    p = row['product']
                    product_totals[p]['qty'] += row['qty']
                    product_totals[p]['revenue'] += row['revenue']
                print(f"products={len(product_totals)}")
            """),
                # Cell 4: Report
                textwrap.dedent("""\
                for product, totals in sorted(product_totals.items()):
                    print(f"{product}: qty={totals['qty']}, revenue=${totals['revenue']:.2f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "extracted=5 rows" in nb_runner.get_output(1)
        assert "transformed=5 rows" in nb_runner.get_output(2)
        assert "products=2" in nb_runner.get_output(3)
        out4 = nb_runner.get_output(4)
        assert "Widget:" in out4
        assert "Gadget:" in out4

    def test_etl_pipeline_change_propagation(self, nb_runner, tmp_path):
        """ETL pipeline propagates changes when transform changes."""
        data_dir = tmp_path / "etl_data2"
        data_dir.mkdir()
        csv_file = data_dir / "data.csv"
        csv_file.write_text("name,value\nA,10\nB,20\nC,30\n", encoding="utf-8")
        fpath = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import csv
                with open('{fpath}', 'r') as f:
                    rows = list(csv.DictReader(f))
                values = [int(r['value']) for r in rows]
            """),
                textwrap.dedent("""\
                result = sum(values)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)

        # Change transform: multiply by 2
        nb_runner.set_cell_source(
            1,
            textwrap.dedent(f"""\
            import csv
            with open('{fpath}', 'r') as f:
                rows = list(csv.DictReader(f))
            values = [int(r['value']) * 2 for r in rows]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "result=120" in nb_runner.get_output(2)


# JSON/CSV data processing chains.
@pytest.mark.stress
@pytest.mark.integration
class TestJsonProcessing:
    """JSON manipulation and processing patterns."""

    def test_json_roundtrip(self, nb_runner):
        """JSON serialize/deserialize roundtrip."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import json
                data = {
                    'users': [
                        {'name': 'Alice', 'age': 30, 'scores': [95, 87, 92]},
                        {'name': 'Bob', 'age': 25, 'scores': [78, 82, 90]},
                    ],
                    'metadata': {'version': '1.0', 'count': 2}
                }
                json_str = json.dumps(data, indent=2)
                restored = json.loads(json_str)
                match = data == restored
            """),
                "print(f'match={match}')\nprint(f'users={len(restored[\"users\"])}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "match=True" in out
        assert "users=2" in out

    def test_json_nested_query(self, nb_runner):
        """Query nested JSON structure."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import json
                config = {
                    'database': {
                        'primary': {'host': 'db1.example.com', 'port': 5432},
                        'replica': {'host': 'db2.example.com', 'port': 5432},
                    },
                    'cache': {'host': 'redis.example.com', 'port': 6379},
                }

                def get_nested(d, path, default=None):
                    keys = path.split('.')
                    current = d
                    for k in keys:
                        if isinstance(current, dict) and k in current:
                            current = current[k]
                        else:
                            return default
                    return current

                primary_host = get_nested(config, 'database.primary.host')
                cache_port = get_nested(config, 'cache.port')
                missing = get_nested(config, 'database.tertiary.host', 'N/A')
            """),
                "print(f'primary={primary_host} cache_port={cache_port} missing={missing}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "primary=db1.example.com" in out
        assert "cache_port=6379" in out
        assert "missing=N/A" in out


@pytest.mark.stress
@pytest.mark.integration
class TestCsvProcessing:
    """CSV data processing patterns."""

    def test_csv_write_read(self, nb_runner, tmp_path):
        """Write CSV, read back, transform."""
        csv_path = str(tmp_path / "data" / "test.csv").replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import csv, os
                os.makedirs(os.path.dirname('{csv_path}'), exist_ok=True)
                rows = [
                    ['name', 'department', 'salary'],
                    ['Alice', 'Engineering', '95000'],
                    ['Bob', 'Marketing', '72000'],
                    ['Charlie', 'Engineering', '88000'],
                    ['Diana', 'Marketing', '78000'],
                ]
                with open('{csv_path}', 'w', newline='') as f:
                    csv.writer(f).writerows(rows)
            """),
                textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                eng_avg = sum(int(r['salary']) for r in data if r['department'] == 'Engineering') / sum(1 for r in data if r['department'] == 'Engineering')
            """),
                "print(f'count={len(data)} eng_avg={eng_avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=4" in out
        assert "eng_avg=91500" in out

    def test_csv_propagation(self, nb_runner, tmp_path):
        """CSV with upstream filter change propagation."""
        csv_path = str(tmp_path / "data" / "scores.csv").replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import csv, os
                os.makedirs(os.path.dirname('{csv_path}'), exist_ok=True)
                with open('{csv_path}', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['name', 'score'])
                    for name, score in [('A', 90), ('B', 75), ('C', 85), ('D', 60), ('E', 95)]:
                        w.writerow([name, score])
            """),
                "min_score = 80",
                textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    data = list(csv.DictReader(f))
                passing = [r['name'] for r in data if int(r['score']) >= min_score]
            """),
                "print(f'passing={passing}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "A" in out
        assert "C" in out
        assert "E" in out

        nb_runner.set_cell_source(2, "min_score = 90")
        nb_runner.run_cells([2, 3, 4])
        out2 = nb_runner.get_output(4)
        assert "A" in out2
        assert "E" in out2
        # B, C, D should no longer be in passing
        assert "B" not in out2
        assert "C" not in out2


# Complex real-world data analysis simulation tests.
#
# Tests simulating real data analysis workflows with multiple
# edit cycles, variable reuse, and result verification.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestStatisticalAnalysis:
    """Statistical analysis workflow with edits."""

    def test_mean_calculation_edit(self, nb_runner):
        """Compute mean, then edit the data source."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]  # sample data",
                "mean_val = sum(data) / len(data)\nprint(f'mean = {mean_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [100, 200, 300]  # new sample data")
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(2)

    def test_data_pipeline_multiple_stats(self, nb_runner):
        """Compute multiple statistics, edit the dataset."""
        nb_runner.create_notebook(
            [
                "nums = [4, 8, 15, 16, 23, 42]  # dataset",
                "n = len(nums)\nmean = sum(nums) / n",
                "variance = sum((x - mean) ** 2 for x in nums) / n",
                "import math\nstd = math.sqrt(variance)\nprint(f'mean={mean:.1f} std={std:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=18.0" in nb_runner.get_output(4)

        # Change dataset
        nb_runner.set_cell_source(1, "nums = [10, 10, 10, 10]  # uniform dataset")
        nb_runner.run_all()
        assert "mean=10.0" in nb_runner.get_output(4)
        assert "std=0.0" in nb_runner.get_output(4)


@pytest.mark.core
class TestNumpyPatterns:
    """Test numpy array operations and caching."""

    def test_numpy_random_with_seed(self, nb_runner):
        """Seeded random should be reproducible and cacheable."""
        nb_runner.create_notebook(
            [
                "import numpy as np\nnp.random.seed(42)",
                "vals = np.random.rand(3)\nprint([round(v, 4) for v in vals])",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(2)

        # Re-run — should produce same output (from cache or same seed)
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)

        # Both should contain the same values
        assert out1 == out2 or "0.3745" in out2, f"Inconsistent: {out1} vs {out2}"


@pytest.mark.core
class TestComplexDataStructures:
    """Test caching with complex nested data structures."""

    def test_nested_dict_of_lists(self, nb_runner):
        """Nested dict creation and access."""
        nb_runner.create_notebook(
            [
                "data = {'users': [{'name': 'Alice', 'scores': [90, 85]}, {'name': 'Bob', 'scores': [78, 92]}]}",
                "avg_scores = {u['name']: sum(u['scores'])/len(u['scores']) for u in data['users']}\nprint(avg_scores)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "'Alice': 87.5" in out, f"Got: {out}"
        assert "'Bob': 85.0" in out, f"Got: {out}"
