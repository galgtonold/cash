"""Complex multi-cell ETL pipeline — cash caching with realistic data transforms."""

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
        csv_file.write_text("name,value\nA,10\nB,20\nC,30\n")
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
