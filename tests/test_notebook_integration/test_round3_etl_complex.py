"""Batch 73: Complex multi-cell ETL pipeline — cash caching with realistic data transforms."""
import textwrap
import pytest


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
            "2024-01-03,Widget,15,9.99\n"
        )
        fpath = str(csv_file).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
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
        csv_file.write_text(
            "name,value\n"
            "A,10\n"
            "B,20\n"
            "C,30\n"
        )
        fpath = str(csv_file).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)

        # Change transform: multiply by 2
        nb_runner.set_cell_source(1, textwrap.dedent(f"""\
            import csv
            with open('{fpath}', 'r') as f:
                rows = list(csv.DictReader(f))
            values = [int(r['value']) * 2 for r in rows]
        """))
        nb_runner.run_cells([1, 2])
        assert "result=120" in nb_runner.get_output(2)


@pytest.mark.stress
class TestDataProcessingPipeline:
    """Test multi-step data processing pipelines."""

    def test_filter_map_reduce_chain(self, nb_runner):
        """Filter → Map → Reduce pipeline across cells."""
        nb_runner.create_notebook([
            # Cell 1: Generate data
            textwrap.dedent("""\
                import random
                random.seed(42)
                data = [random.randint(1, 100) for _ in range(50)]
                print(f"generated={len(data)} items")
            """),
            # Cell 2: Filter
            textwrap.dedent("""\
                filtered = [x for x in data if x > 30]
                print(f"filtered={len(filtered)} items")
            """),
            # Cell 3: Map
            textwrap.dedent("""\
                mapped = [x ** 2 for x in filtered]
                print(f"mapped={len(mapped)} items, min={min(mapped)}, max={max(mapped)}")
            """),
            # Cell 4: Reduce
            textwrap.dedent("""\
                total = sum(mapped)
                avg = total / len(mapped)
                print(f"total={total} avg={avg:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "generated=50" in nb_runner.get_output(1)
        assert "filtered=" in nb_runner.get_output(2)
        assert "mapped=" in nb_runner.get_output(3)
        assert "total=" in nb_runner.get_output(4)

    def test_pivot_aggregation(self, nb_runner):
        """Pivot table aggregation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import defaultdict

                sales = [
                    ('Q1', 'East', 100), ('Q1', 'West', 150),
                    ('Q2', 'East', 120), ('Q2', 'West', 180),
                    ('Q3', 'East', 90),  ('Q3', 'West', 200),
                    ('Q4', 'East', 130), ('Q4', 'West', 170),
                ]

                # Pivot: quarter -> region -> amount
                pivot = defaultdict(dict)
                for quarter, region, amount in sales:
                    pivot[quarter][region] = amount
            """),
            textwrap.dedent("""\
                # Summary
                for q in sorted(pivot.keys()):
                    east = pivot[q].get('East', 0)
                    west = pivot[q].get('West', 0)
                    print(f"{q}: East={east} West={west} Total={east + west}")
            """),
            textwrap.dedent("""\
                # Grand totals
                grand_east = sum(pivot[q].get('East', 0) for q in pivot)
                grand_west = sum(pivot[q].get('West', 0) for q in pivot)
                print(f"Grand: East={grand_east} West={grand_west} Total={grand_east + grand_west}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Q1:" in out2
        assert "Q4:" in out2
        out3 = nb_runner.get_output(3)
        assert "Grand:" in out3
        assert "Total=1140" in out3
