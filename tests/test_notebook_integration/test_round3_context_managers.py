"""Context managers & resource management — cash caching with with-statements."""

import textwrap

import pytest


@pytest.mark.stress
class TestContextManagerBasics:
    """Test context manager patterns across cells."""

    def test_custom_context_manager(self, nb_runner):
        """Custom context manager with __enter__/__exit__."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Timer:
                    def __init__(self, label):
                        self.label = label
                        self.elapsed = None
                    def __enter__(self):
                        import time
                        self._start = time.perf_counter()
                        return self
                    def __exit__(self, *args):
                        import time
                        self.elapsed = time.perf_counter() - self._start
                        return False

                with Timer('test') as t:
                    total = sum(range(100000))
                print(f"total={total} timed={t.elapsed is not None}")
            """),
                textwrap.dedent("""\
                print(f"label={t.label} elapsed_positive={t.elapsed > 0}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=4999950000" in nb_runner.get_output(1)
        assert "timed=True" in nb_runner.get_output(1)
        assert "label=test" in nb_runner.get_output(2)
        assert "elapsed_positive=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestResourcePatterns:
    """Test resource management patterns."""

    def test_exitstack(self, nb_runner):
        """ExitStack for dynamic context management."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from contextlib import ExitStack

                cleanup_log = []

                class Resource:
                    def __init__(self, name):
                        self.name = name
                    def __enter__(self):
                        cleanup_log.append(f"opened:{self.name}")
                        return self
                    def __exit__(self, *args):
                        cleanup_log.append(f"closed:{self.name}")

                with ExitStack() as stack:
                    resources = [stack.enter_context(Resource(f"r{i}")) for i in range(3)]
                    names = [r.name for r in resources]
                print(f"names={names}")
            """),
                textwrap.dedent("""\
                # Verify LIFO cleanup order
                closed = [e for e in cleanup_log if e.startswith('closed')]
                print(f"cleanup_order={closed}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['r0', 'r1', 'r2']" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "closed:r2" in out2
        # LIFO: r2 closes before r1 before r0
        assert out2.index("closed:r2") < out2.index("closed:r0")

    def test_context_manager_propagation(self, nb_runner):
        """Context manager results propagate on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from contextlib import contextmanager

                @contextmanager
                def database(name):
                    db = {'name': name, 'data': []}
                    yield db
                    # cleanup

                with database('test') as db:
                    db['data'].extend([1, 2, 3])
                    snapshot = dict(db)
                print(f"snapshot={snapshot}")
            """),
                textwrap.dedent("""\
                print(f"name={snapshot['name']} count={len(snapshot['data'])}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=test" in nb_runner.get_output(2)
        assert "count=3" in nb_runner.get_output(2)

        # Change DB name and data
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from contextlib import contextmanager

            @contextmanager
            def database(name):
                db = {'name': name, 'data': []}
                yield db

            with database('production') as db:
                db['data'].extend([10, 20, 30, 40, 50])
                snapshot = dict(db)
            print(f"snapshot={snapshot}")
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "name=production" in nb_runner.get_output(2)
        assert "count=5" in nb_runner.get_output(2)
