"""with blocks, contextlib and custom context managers across cells."""

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
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestContextManagerEdits:
    """Editing context manager patterns."""

    def test_edit_context_manager_body(self, nb_runner):
        """Edit the body of a with statement."""
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager",
                "@contextmanager\ndef tag(name):\n    print(f'<{name}>', end='')\n    yield\n    print(f'</{name}>', end='')",
                "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nwith tag('div'):\n    print('hello', end='')\nsys.stdout = old\nresult = buf.getvalue()\nprint(repr(result))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "<div>hello</div>" in out3

        # Change tag name
        nb_runner.set_cell_source(
            3,
            "import io, sys\nbuf = io.StringIO()\nold = sys.stdout\nsys.stdout = buf\nwith tag('span'):\n    print('world', end='')\nsys.stdout = old\nresult = buf.getvalue()\nprint(repr(result))",
        )
        nb_runner.run_all()
        out3b = nb_runner.get_output(3)
        assert "<span>world</span>" in out3b

    def test_edit_cm_class(self, nb_runner):
        """Edit a class-based context manager."""
        nb_runner.create_notebook(
            [
                "class Timer:\n    def __enter__(self):\n        self.msg = 'started'\n        return self\n    def __exit__(self, *args):\n        self.msg = 'stopped'",
                "with Timer() as t:\n    pass\nprint(f'msg = {t.msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = stopped" in nb_runner.get_output(2)

        # Change exit message
        nb_runner.set_cell_source(
            1,
            "class Timer:\n    def __enter__(self):\n        self.msg = 'started'\n        return self\n    def __exit__(self, *args):\n        self.msg = 'finished'",
        )
        nb_runner.run_all()
        assert "msg = finished" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextManagerClassPattern:
    """context manager class enter exit pattern."""

    def test_custom_context_manager(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Timer:\n    def __init__(self, name): self.name = name\n    def __enter__(self):\n        self.log = [f'enter:{self.name}']\n        return self\n    def __exit__(self, *args):\n        self.log.append(f'exit:{self.name}')\n        return False\nwith Timer('test') as t:\n    t.log.append('body')\nprint(f'log={t.log}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['enter:test', 'body', 'exit:test']" in nb_runner.get_output(2)

    def test_nested_context(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Scope:\n    instances = []\n    def __init__(self, name): self.name = name\n    def __enter__(self):\n        Scope.instances.append(self.name)\n        return self\n    def __exit__(self, *a):\n        Scope.instances.pop()\nwith Scope('outer') as o:\n    with Scope('inner') as i:\n        snapshot = list(Scope.instances)\nprint(f'snapshot={snapshot} after={Scope.instances}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "snapshot=['outer', 'inner']" in out
        assert "after=[]" in out

    def test_context_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Ctx:\n    def __init__(self, v): self.v = v\n    def __enter__(self): return self.v\n    def __exit__(self, *a): pass\nwith Ctx(42) as val:\n    result = val * 2\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=84" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Ctx:\n    def __init__(self, v): self.v = v\n    def __enter__(self): return self.v\n    def __exit__(self, *a): pass\nwith Ctx(100) as val:\n    result = val * 3\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=300" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestCustomContextManagerClass:
    """Test caching with context managers."""

    def test_custom_context_manager_class(self, nb_runner):
        """Custom __enter__/__exit__ context manager."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Timer:
                    def __enter__(self):
                        import time
                        self.start = time.time()
                        return self
                    def __exit__(self, *args):
                        import time
                        self.elapsed = time.time() - self.start
            """),
                textwrap.dedent("""\
                import time
                with Timer() as t:
                    time.sleep(0.01)
                print(f"elapsed={t.elapsed > 0}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elapsed=True" in nb_runner.get_output(2)


class TestResourcePatterns:
    """Resource management patterns."""

    @pytest.mark.stress
    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_temp_file_context(self, nb_runner, tmp_path):
        """Edit temp file usage in a context manager."""
        fpath = str(tmp_path / "ctx_test.txt").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"path = '{fpath}'",
                "with open(path, 'w') as f:\n    f.write('version1')",
                "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = version1" in nb_runner.get_output(3)

        # Edit write content
        nb_runner.set_cell_source(2, "with open(path, 'w') as f:\n    f.write('version2')")
        nb_runner.run_all()
        assert "content = version2" in nb_runner.get_output(3)

    @pytest.mark.stress
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

    @pytest.mark.stress
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


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextmanagerDecorator:
    """Test @contextmanager decorator across cells."""

    def test_contextmanager_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define context manager
                "from contextlib import contextmanager\n@contextmanager\ndef track_state(name):\n    state = {'entered': True, 'name': name, 'exited': False}\n    try:\n        yield state\n    finally:\n        state['exited'] = True\nprint('track_state defined')",
                # Cell 2: use context manager
                "with track_state('test') as s:\n    s['value'] = 42\n    inside_name = s['name']\nprint(f'name={inside_name}')\nprint(f'exited={s[\"exited\"]}')\nprint(f'value={s[\"value\"]}')",
                # Cell 3: reference results
                "summary = f'{inside_name}:{s[\"value\"]}'\nprint(f'summary={summary}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "name=test" in out2
        assert "exited=True" in out2
        assert "value=42" in out2
        out3 = nb_runner.get_output(3)
        assert "summary=test:42" in out3

    def test_contextmanager_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef counter():\n    c = [0]\n    yield c\n    c[0] += 1  # increment on exit\nprint('counter defined')",
                "with counter() as c:\n    c[0] = 10\nresult = c[0]\nprint(f'result={result}')",
                "doubled = result * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)
        assert "doubled=22" in nb_runner.get_output(3)

        # Edit initial value
        nb_runner.set_cell_source(2, "with counter() as c:\n    c[0] = 20\nresult = c[0]\nprint(f'result={result}')")
        nb_runner.run_cells([2, 3])
        assert "result=21" in nb_runner.get_output(2)
        assert "doubled=42" in nb_runner.get_output(3)

    def test_contextmanager_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef tag(name):\n    result = [f'<{name}>']\n    yield result\n    result.append(f'</{name}>')\nprint('tag defined')",
                "with tag('div') as parts:\n    parts.append('content')\nhtml = ''.join(parts)\nprint(f'html={html}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "html=<div>content</div>" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "html=<div>content</div>" in nb_runner.get_output(2)


@pytest.mark.stress
class TestContextmanagerAcrossCells:
    """Test context manager patterns."""

    def test_contextlib_contextmanager(self, nb_runner):
        """@contextmanager decorator across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from contextlib import contextmanager

                @contextmanager
                def timer_context(name):
                    import time
                    log = []
                    log.append(f"start:{name}")
                    start = time.time()
                    try:
                        yield log
                    finally:
                        elapsed = time.time() - start
                        log.append(f"end:{name}:{elapsed:.3f}s")
            """),
                textwrap.dedent("""\
                with timer_context("computation") as log:
                    result = sum(range(10000))
                    log.append(f"computed:{result}")
                print(f"log_count={len(log)} first={log[0]}")
                print(f"has_end={'end:computation' in log[2]}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log_count=3" in out
        assert "start:computation" in out
        assert "has_end=True" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextlibOps:
    """Test contextlib utilities across cells."""

    def test_contextmanager(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: custom context manager
                "from contextlib import contextmanager\nlog = []\n@contextmanager\ndef tracked(name):\n    log.append(f'enter:{name}')\n    yield name.upper()\n    log.append(f'exit:{name}')\n\nwith tracked('test') as val:\n    log.append(f'inside:{val}')\nlog_str = '->'.join(log)\nprint(f'flow={log_str}')",
                # Cell 2: check val from context
                "upper_val = val\nprint(f'upper_val={upper_val}')",
                # Cell 3: suppress exceptions
                "from contextlib import suppress\nresults = []\nfor x in [10, 0, 5]:\n    with suppress(ZeroDivisionError):\n        results.append(100 // x)\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "flow=enter:test->inside:TEST->exit:test" in out1
        out2 = nb_runner.get_output(2)
        assert "upper_val=TEST" in out2
        out3 = nb_runner.get_output(3)
        assert "results=[10, 20]" in out3

    def test_contextlib_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef multiplier(factor):\n    yield lambda x: x * factor\n\nwith multiplier(3) as fn:\n    result = fn(10)\nprint(f'result={result}')",
                "doubled = result * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(1)
        assert "doubled=60" in nb_runner.get_output(2)

        # Edit factor
        nb_runner.set_cell_source(
            1,
            "from contextlib import contextmanager\n@contextmanager\ndef multiplier(factor):\n    yield lambda x: x * factor\n\nwith multiplier(5) as fn:\n    result = fn(10)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=50" in nb_runner.get_output(1)
        assert "doubled=100" in nb_runner.get_output(2)

    def test_contextlib_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress\nerrors = 0\nfor v in ['1', 'x', '3', 'y']:\n    with suppress(ValueError):\n        int(v)\n        continue\n    errors += 1\nprint(f'errors={errors}')",
                "has_errors = errors > 0\nprint(f'has_errors={has_errors}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "errors=2" in nb_runner.get_output(1)
        assert "has_errors=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_errors=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextlibPatterns:
    """contextlib.suppress and contextlib patterns."""

    def test_suppress(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress\ndata = {'a': 1}",
                "with suppress(KeyError):\n    val = data['missing']\nresult = data.get('a', 0)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1" in nb_runner.get_output(2)

    def test_redirect_stdout(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import redirect_stdout\nimport io",
                "f = io.StringIO()\nwith redirect_stdout(f):\n    print('captured')\noutput = f.getvalue().strip()\nprint(f'output={output}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "output=captured" in nb_runner.get_output(2)

    def test_custom_contextmanager(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import contextmanager\n@contextmanager\ndef timer_mock():\n    log = ['enter']\n    yield log\n    log.append('exit')",
                "with timer_mock() as log:\n    log.append('work')\nprint(f'log={log}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['enter', 'work', 'exit']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextlibSuppressRedirect:
    """contextlib suppress and redirect_stdout."""

    def test_suppress(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress",
                "with suppress(KeyError):\n    d = {}\n    val = d['missing']\nresult = 'survived'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=survived" in nb_runner.get_output(2)

    def test_suppress_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress\ndata = [1, 2, 3]",
                "with suppress(IndexError):\n    val = data[10]\nresult = len(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from contextlib import suppress\ndata = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "result=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestContextlibSuppressStringIO:
    """contextlib suppress and redirect to stringio."""

    def test_suppress_errors(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress",
                "results = []\nfor key in ['a', 'b', 'c']:\n    d = {'a': 1, 'c': 3}\n    with suppress(KeyError):\n        results.append(d[key])\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 3]" in nb_runner.get_output(2)

    def test_redirect_stdout(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import redirect_stdout\nimport io",
                "buf = io.StringIO()\nwith redirect_stdout(buf):\n    print('captured line')\n    print('second line')\ncaptured = buf.getvalue()\nlines = captured.strip().split('\\n')\nprint(f'count={len(lines)} first={lines[0]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=2" in out
        assert "first=captured line" in out

    def test_suppress_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from contextlib import suppress",
                "val = 0\nwith suppress(ZeroDivisionError):\n    val = 10 // 0\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=0" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "val = 0\nwith suppress(ZeroDivisionError):\n    val = 10 // 2\nprint(f'val={val}')"
        )
        nb_runner.run_all()
        assert "val=5" in nb_runner.get_output(2)
