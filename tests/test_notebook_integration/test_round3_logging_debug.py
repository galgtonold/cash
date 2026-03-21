"""Batch 54: Logging & debug patterns — cash caching with logging, warnings, traceback."""
import textwrap
import pytest


@pytest.mark.stress
class TestLoggingPatterns:
    """Test Python logging module across cells."""

    def test_basic_logging_setup(self, nb_runner):
        """Logger setup and usage across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import logging
                import io

                # Create a string handler to capture output
                log_stream = io.StringIO()
                handler = logging.StreamHandler(log_stream)
                handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))

                logger = logging.getLogger('test_logger')
                logger.handlers = [handler]
                logger.setLevel(logging.DEBUG)
            """),
            textwrap.dedent("""\
                logger.info("started processing")
                logger.warning("something unusual")
                logger.debug("debug info")
                log_output = log_stream.getvalue()
                line_count = len(log_output.strip().split('\\n'))
                print(f"lines={line_count}")
                print(f"has_warning={'WARNING' in log_output}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=3" in nb_runner.get_output(2)
        assert "has_warning=True" in nb_runner.get_output(2)

    def test_custom_log_filter(self, nb_runner):
        """Custom log filter across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import logging
                import io

                class LevelFilter(logging.Filter):
                    def __init__(self, min_level):
                        super().__init__()
                        self.min_level = min_level
                    def filter(self, record):
                        return record.levelno >= self.min_level

                stream = io.StringIO()
                handler = logging.StreamHandler(stream)
                handler.addFilter(LevelFilter(logging.WARNING))
                handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))

                logger = logging.getLogger('filtered_logger')
                logger.handlers = [handler]
                logger.setLevel(logging.DEBUG)
            """),
            textwrap.dedent("""\
                logger.debug("hidden")
                logger.info("also hidden")
                logger.warning("visible1")
                logger.error("visible2")
                output = stream.getvalue()
                lines = [l for l in output.strip().split('\\n') if l]
                print(f"visible_count={len(lines)}")
                print(f"first={lines[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "visible_count=2" in out
        assert "WARNING:visible1" in out


@pytest.mark.stress
class TestWarningsPatterns:
    """Test warnings module."""

    def test_custom_warning(self, nb_runner):
        """Custom warning class across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import warnings

                class DeprecationWarning2(UserWarning):
                    pass

                def old_api(x):
                    warnings.warn("old_api is deprecated, use new_api", DeprecationWarning2, stacklevel=2)
                    return x * 2

                def new_api(x):
                    return x * 3
            """),
            textwrap.dedent("""\
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = old_api(5)
                    warn_count = len(w)
                    warn_msg = str(w[0].message) if w else "none"
                print(f"result={result} warns={warn_count} msg={warn_msg}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result=10" in out
        assert "warns=1" in out
        assert "deprecated" in out


@pytest.mark.stress
class TestContextManagerPatterns:
    """Test context manager patterns."""

    def test_contextlib_contextmanager(self, nb_runner):
        """@contextmanager decorator across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log_count=3" in out
        assert "start:computation" in out
        assert "has_end=True" in out

    def test_nested_context_managers(self, nb_runner):
        """Nested context managers across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class TrackingCM:
                    history = []
                    def __init__(self, name):
                        self.name = name
                    def __enter__(self):
                        TrackingCM.history.append(f"enter:{self.name}")
                        return self
                    def __exit__(self, *args):
                        TrackingCM.history.append(f"exit:{self.name}")
                        return False
            """),
            textwrap.dedent("""\
                with TrackingCM("outer") as o:
                    with TrackingCM("inner") as i:
                        TrackingCM.history.append("work")
                print(f"history={TrackingCM.history}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "enter:outer" in out
        assert "enter:inner" in out
        assert "exit:inner" in out
        assert "exit:outer" in out

    def test_suppress_context(self, nb_runner):
        """contextlib.suppress across cells."""
        nb_runner.create_notebook([
            "from contextlib import suppress",
            textwrap.dedent("""\
                results = []
                for val in ['10', 'abc', '20', None, '30']:
                    with suppress(TypeError, ValueError):
                        results.append(int(val))
                print(f"results={results}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, 20, 30]" in nb_runner.get_output(2)
