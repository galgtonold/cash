"""Batch 100 – Grand finale: full end-to-end data science pipeline stress test."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestFullPipeline:
    """End-to-end data science pipeline: 8 cells, multiple dependencies."""

    def test_full_data_pipeline(self, nb_runner, tmp_path):
        """Complete pipeline: config → data gen → clean → feature eng → model → eval → report."""
        csv_path = str(tmp_path / "pipeline_data.csv").replace('\\', '/')
        nb_runner.create_notebook([
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
        ])
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
        csv_path = str(tmp_path / "pipeline2.csv").replace('\\', '/')
        nb_runner.create_notebook([
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
        ])
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
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(6)
        assert "Total: 50" in out
        assert "Mean:" in out
        assert "Distribution:" in out
        assert "Top 5:" in out
