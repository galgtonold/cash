"""
Tests for Control Structure Processing

These tests verify that:
- For loops are decomposed per-iteration with each body statement going
  through the statement processor separately.
- While, if, with, try are processed as single cacheable units.
- Break/continue loops fall back to single-unit execution.
- Iteration context hashing differentiates cache keys per iteration.
"""

import pytest
import ast
from unittest.mock import MagicMock
from cash.notebook.cache_status import CacheStatus


class TestControlStructureDetection:
    """Test detection of control structures from AST nodes."""

    def test_for_loop_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("for i in range(3): print(i)").body[0]
        assert is_control_structure(node) is True

    def test_while_loop_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("while True: break").body[0]
        assert is_control_structure(node) is True

    def test_if_statement_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("if x > 0: print('positive')").body[0]
        assert is_control_structure(node) is True

    def test_with_statement_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("with open('f') as f: pass").body[0]
        assert is_control_structure(node) is True

    def test_try_statement_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("try: pass\nexcept: pass").body[0]
        assert is_control_structure(node) is True

    def test_assignment_not_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("x = 1").body[0]
        assert is_control_structure(node) is False

    def test_expression_not_detected(self):
        from cash.notebook.control_structures import is_control_structure
        node = ast.parse("print('hello')").body[0]
        assert is_control_structure(node) is False


class TestControlStructureProcessor:
    """Test the ControlStructureProcessor class."""

    @pytest.fixture
    def mock_shell(self):
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    @pytest.fixture
    def mock_statement_processor(self):
        processor = MagicMock()
        processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': '',
            'stderr': '',
            'outputs': []
        })
        processor.variable_lineage = {}
        processor.vars_with_mutation_lineage = set()
        processor.compute_hash = MagicMock(return_value='fakehash')
        return processor

    @pytest.fixture
    def control_processor(self, mock_shell, mock_statement_processor):
        from cash.notebook.control_structures import ControlStructureProcessor
        return ControlStructureProcessor(
            mock_shell,
            mock_statement_processor,
            debug=True
        )

    # ---- For loops: per-iteration ----

    def test_for_loop_per_iteration(self, control_processor, mock_shell, mock_statement_processor):
        """For loop should be decomposed per-iteration, one call per body statement per iteration."""
        code = "for i in range(3): x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        assert result.total_iterations == 3  # 3 iterations
        # 1 body statement × 3 iterations = 3 calls
        assert mock_statement_processor.process_statement.call_count == 3

    def test_for_loop_body_gets_iteration_context(self, control_processor, mock_shell, mock_statement_processor):
        """Each body statement should include __iteration_context__ in code."""
        code = "for i in range(2):\n    x = i\n    y = i * 2"
        node = ast.parse(code).body[0]

        control_processor.process(node)

        # 2 body statements × 2 iterations = 4 calls
        assert mock_statement_processor.process_statement.call_count == 4
        # Every call should have the iteration context marker
        for call in mock_statement_processor.process_statement.call_args_list:
            passed_code = call[0][0]
            assert '# __iteration_context__:' in passed_code

    def test_for_loop_body_statements_not_full_loop(self, control_processor, mock_shell, mock_statement_processor):
        """Body statements should be individual statements, not the full loop code."""
        code = "for i in range(2):\n    x = i * 2"
        node = ast.parse(code).body[0]

        control_processor.process(node)

        for call in mock_statement_processor.process_statement.call_args_list:
            passed_code = call[0][0]
            # Should contain the body statement, not the full for-loop header
            assert 'x = i * 2' in passed_code
            assert 'for i in range' not in passed_code

    def test_for_loop_different_context_per_iteration(self, control_processor, mock_shell, mock_statement_processor):
        """Different iterations should have different context hashes."""
        code = "for i in [1, 2]: x = i"
        node = ast.parse(code).body[0]

        control_processor.process(node)

        calls = mock_statement_processor.process_statement.call_args_list
        ctx_hashes = set()
        for call in calls:
            passed_code = call[0][0]
            # Extract the context hash
            line = passed_code.split('\n')[0]
            ctx_hash = line.split(': ')[1]
            ctx_hashes.add(ctx_hash)

        # Should have 2 different context hashes
        assert len(ctx_hashes) == 2

    def test_for_loop_cached_all(self, control_processor, mock_shell, mock_statement_processor):
        """When all iterations are cached, report all cached."""
        mock_statement_processor.process_statement.return_value = {
            'status': CacheStatus.RESTORED,
            'execution_time': 0.0,
            'stdout': '', 'stderr': '', 'outputs': []
        }

        code = "for i in range(3): x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        assert result.total_iterations == 3
        assert result.cached_iterations == 3
        assert result.computed_iterations == 0

    def test_for_loop_mixed_cache(self, control_processor, mock_shell, mock_statement_processor):
        """When some iterations compute and some restore, track correctly."""
        # First 2 calls COMPUTED, last one RESTORED
        mock_statement_processor.process_statement.side_effect = [
            {'status': CacheStatus.COMPUTED, 'execution_time': 0.01,
             'stdout': '', 'stderr': '', 'outputs': []},
            {'status': CacheStatus.RESTORED, 'execution_time': 0.0,
             'stdout': '', 'stderr': '', 'outputs': []},
            {'status': CacheStatus.COMPUTED, 'execution_time': 0.01,
             'stdout': '', 'stderr': '', 'outputs': []},
        ]

        # 1 body statement × 3 iterations = 3 calls
        code = "for i in range(3): x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        # iterations 1 and 3 computed, iteration 2 cached
        assert result.total_iterations == 3
        assert result.computed_iterations == 2
        assert result.cached_iterations == 1

    def test_for_loop_loop_vars_in_metrics(self, control_processor, mock_shell, mock_statement_processor):
        """Metrics should include loop_vars with the iteration variable values."""
        code = "for i in [10, 20]: x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        # Check that at least one metric has loop_vars
        loop_vars_found = False
        for m in result.metrics:
            if 'loop_vars' in m:
                loop_vars_found = True
                assert 'i' in m['loop_vars']
        assert loop_vars_found

    # ---- Break/continue: single unit fallback ----

    def test_for_loop_with_break_single_unit(self, control_processor, mock_shell, mock_statement_processor):
        """For loops with break should be executed as a single unit."""
        code = "for i in range(10):\n    if i > 5:\n        break\n    x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # Single unit: 1 call to statement processor
        assert mock_statement_processor.process_statement.call_count == 1
        assert result.total_iterations == 1

    def test_for_loop_with_continue_single_unit(self, control_processor, mock_shell, mock_statement_processor):
        """For loops with continue should be executed as a single unit."""
        code = "for i in range(5):\n    if i % 2 == 0:\n        continue\n    x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert mock_statement_processor.process_statement.call_count == 1
        assert result.total_iterations == 1

    # ---- Nested for loops ----

    def test_nested_for_loops(self, control_processor, mock_shell, mock_statement_processor):
        """Nested for loops should both be decomposed per-iteration."""
        code = "for i in range(2):\n    for j in range(2):\n        x = i + j"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # Outer: 2 iterations, inner: 2 iterations each = 4 body statement calls
        assert result.total_iterations == 2
        assert mock_statement_processor.process_statement.call_count == 4

    # ---- Error handling ----

    def test_error_result_reported(self, control_processor, mock_shell, mock_statement_processor):
        """When statement processor returns ERROR, report failure."""
        mock_statement_processor.process_statement.return_value = {
            'status': CacheStatus.ERROR,
            'execution_time': 0.0,
            'error': RuntimeError("test error"),
            'stdout': '', 'stderr': '', 'outputs': []
        }

        code = "for i in range(3): x = i"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is False

    # ---- If statement: single unit ----

    def test_if_statement_per_statement(self, control_processor, mock_shell, mock_statement_processor):
        """If statement should be processed per-statement (each body statement individually)."""
        mock_shell.user_ns['condition'] = True

        code = "if condition:\n    x = 1\nelse:\n    x = 0"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # If statement executes each body statement individually
        assert mock_statement_processor.process_statement.call_count == 1
        passed_code = mock_statement_processor.process_statement.call_args[0][0]
        # Should contain the body statement with control context prefix
        assert 'x = 1' in passed_code
        assert 'control_context' in passed_code

    # ---- While loop: single unit ----

    def test_while_loop_single_unit(self, control_processor, mock_shell, mock_statement_processor):
        """While loop should be processed as a single unit."""
        code = "while count < 3: count = count + 1"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        assert result.total_iterations == 1
        assert mock_statement_processor.process_statement.call_count == 1

    # ---- With statement: single unit ----

    def test_with_statement_single_unit(self, control_processor, mock_shell, mock_statement_processor):
        """With statement should be processed as a single unit."""
        code = "with open('f') as f:\n    data = f.read()"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        assert mock_statement_processor.process_statement.call_count == 1

    # ---- Try statement: per-statement ----

    def test_try_statement_single_unit(self, control_processor, mock_shell, mock_statement_processor):
        """Try statement with successful try body processes each body statement individually."""
        code = "try:\n    x = 1\nexcept:\n    x = 0"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # Try body has 1 statement (x = 1) which succeeds → only 1 call
        assert mock_statement_processor.process_statement.call_count == 1

    def test_try_statement_two_body_stmts(self, control_processor, mock_shell, mock_statement_processor):
        """Try with two body statements → two calls to statement_processor."""
        code = "try:\n    x = 1\n    y = 2\nexcept:\n    x = 0"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        assert mock_statement_processor.process_statement.call_count == 2

    def test_try_with_else_body(self, control_processor, mock_shell, mock_statement_processor):
        """Try/else — else body runs when no exception, adds calls."""
        code = "try:\n    x = 1\nexcept:\n    x = 0\nelse:\n    y = 2"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # try body (1 stmt) + else body (1 stmt) = 2 calls
        assert mock_statement_processor.process_statement.call_count == 2

    def test_try_with_finally_body(self, control_processor, mock_shell, mock_statement_processor):
        """Try/finally — finally always runs."""
        code = "try:\n    x = 1\nexcept:\n    x = 0\nfinally:\n    z = 3"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # try body (1 stmt) + finally body (1 stmt) = 2 calls
        assert mock_statement_processor.process_statement.call_count == 2

    def test_try_except_error_routes_to_handler(self, control_processor, mock_shell, mock_statement_processor):
        """When try body raises, should route to matching except handler."""
        # First call raises ValueError; second call (in handler) succeeds
        mock_statement_processor.process_statement.side_effect = [
            ValueError("test error"),
            {'status': CacheStatus.COMPUTED, 'execution_time': 0.01, 'stdout': '', 'stderr': '', 'outputs': []},
        ]

        code = "try:\n    x = int('abc')\nexcept ValueError:\n    x = -1"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        assert result.success is True
        # 1 call for try body (raises) + 1 call for handler body
        assert mock_statement_processor.process_statement.call_count == 2

    def test_try_control_context_in_code(self, control_processor, mock_shell, mock_statement_processor):
        """Body statements should include control_context in the code."""
        code = "try:\n    x = 1\nexcept:\n    x = 0"
        node = ast.parse(code).body[0]

        control_processor.process(node)

        for call in mock_statement_processor.process_statement.call_args_list:
            passed_code = call[0][0]
            assert '# control_context:' in passed_code

    def test_try_metrics_tagged_with_control_type(self, control_processor, mock_shell, mock_statement_processor):
        """Metrics from try processing should have control_type = 'try'."""
        code = "try:\n    x = 1\nexcept:\n    x = 0"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)

        for m in result.metrics:
            assert m.get('control_type') == 'try'


class TestIterationContextHashing:
    """Test that iteration context hashing differentiates cache keys."""

    def test_context_hash_different_values(self):
        from cash.notebook.control_structures import compute_context_hash
        h1 = compute_context_hash({'i': 0})
        h2 = compute_context_hash({'i': 1})
        assert h1 != h2

    def test_context_hash_same_values(self):
        from cash.notebook.control_structures import compute_context_hash
        h1 = compute_context_hash({'i': 5})
        h2 = compute_context_hash({'i': 5})
        assert h1 == h2

    def test_context_hash_multiple_variables(self):
        from cash.notebook.control_structures import compute_context_hash
        h1 = compute_context_hash({'i': 0, 'j': 0})
        h2 = compute_context_hash({'i': 0, 'j': 1})
        assert h1 != h2


class TestTargetExtraction:
    """Test extraction of target variable names from for loop targets."""

    def test_simple_name_target(self):
        from cash.notebook.control_structures import extract_target_names
        target = ast.parse("for i in range(3): pass").body[0].target
        assert extract_target_names(target) == ['i']

    def test_tuple_target(self):
        from cash.notebook.control_structures import extract_target_names
        target = ast.parse("for x, y, z in items: pass").body[0].target
        assert extract_target_names(target) == ['x', 'y', 'z']

    def test_nested_tuple_target(self):
        from cash.notebook.control_structures import extract_target_names
        target = ast.parse("for (a, b), c in items: pass").body[0].target
        assert set(extract_target_names(target)) == {'a', 'b', 'c'}


class TestOutputFlushing:
    """Test that output is flushed immediately during control structure processing."""

    @pytest.fixture
    def mock_shell(self):
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    @pytest.fixture
    def mock_statement_processor(self):
        processor = MagicMock()
        processor.variable_lineage = {}
        processor.vars_with_mutation_lineage = set()
        processor.compute_hash = MagicMock(return_value='fakehash')
        return processor

    @pytest.fixture
    def control_processor(self, mock_shell, mock_statement_processor):
        from cash.notebook.control_structures import ControlStructureProcessor
        return ControlStructureProcessor(
            mock_shell,
            mock_statement_processor,
            debug=False
        )

    def test_for_loop_flushes_stdout_per_iteration(self, control_processor, mock_shell, mock_statement_processor, capsys):
        """For loop body statements should flush stdout immediately, not after all iterations."""
        # Each process() call returns metrics with stdout
        call_count = [0]
        def mock_process(code, ttl=None, silent=True):
            call_count[0] += 1
            return {
                'status': CacheStatus.COMPUTED,
                'execution_time': 0.01,
                'stdout': f'iteration {call_count[0]}\n',
                'stderr': '',
                'outputs': [],
                'code': code,
            }
        mock_statement_processor.process_statement = mock_process

        code = "for i in range(3):\n    print(i)"
        node = ast.parse(code).body[0]
        mock_shell.user_ns['range'] = range

        result = control_processor.process(node)

        assert result.success is True
        # All output should have been printed (flushed) during processing
        captured = capsys.readouterr()
        assert 'iteration 1' in captured.out
        assert 'iteration 2' in captured.out
        assert 'iteration 3' in captured.out

    def test_for_loop_metrics_marked_as_flushed(self, control_processor, mock_shell, mock_statement_processor):
        """Metrics from for loop body statements should have _output_flushed=True."""
        mock_statement_processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': 'hello\n',
            'stderr': '',
            'outputs': [],
            'code': 'print("hello")',
        })

        code = "for i in range(2):\n    print('hello')"
        node = ast.parse(code).body[0]
        mock_shell.user_ns['range'] = range

        result = control_processor.process(node)

        assert result.success is True
        for m in result.metrics:
            assert m.get('_output_flushed') is True

    def test_for_loop_flushes_stderr(self, control_processor, mock_shell, mock_statement_processor, capsys):
        """For loop should also flush stderr immediately."""
        mock_statement_processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': '',
            'stderr': 'warning\n',
            'outputs': [],
            'code': 'import warnings; warnings.warn("test")',
        })

        code = "for i in range(2):\n    pass"
        node = ast.parse(code).body[0]
        mock_shell.user_ns['range'] = range

        control_processor.process(node)

        captured = capsys.readouterr()
        # stderr should contain warnings from both iterations
        assert captured.err.count('warning') == 2

    def test_if_statement_flushes_stdout(self, control_processor, mock_shell, mock_statement_processor, capsys):
        """If statement body statements should flush stdout immediately."""
        mock_statement_processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': 'branch output\n',
            'stderr': '',
            'outputs': [],
            'code': 'print("branch output")',
        })

        code = "if True:\n    print('branch output')"
        node = ast.parse(code).body[0]
        mock_shell.user_ns['True'] = True

        result = control_processor.process(node)

        assert result.success is True
        captured = capsys.readouterr()
        assert 'branch output' in captured.out
        for m in result.metrics:
            assert m.get('_output_flushed') is True

    def test_flush_metrics_output_helper(self):
        """flush_metrics_output should print stdout/stderr and mark as flushed."""
        from cash.notebook.control_structure_helpers import flush_metrics_output

        metrics = {'stdout': 'hello\n', 'stderr': 'warn\n'}
        import io
        import sys
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            flush_metrics_output(metrics)
            assert sys.stdout.getvalue() == 'hello\n'
            assert sys.stderr.getvalue() == 'warn\n'
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        assert metrics['_output_flushed'] is True

    def test_flush_no_output_still_marks_flushed(self):
        """Metrics with no stdout/stderr should still be marked as flushed."""
        from cash.notebook.control_structure_helpers import flush_metrics_output

        metrics = {'stdout': '', 'stderr': ''}
        flush_metrics_output(metrics)
        assert metrics['_output_flushed'] is True


class TestTeeWriter:
    """Tests for _TeeWriter and _tee_output used by single-unit loop streaming."""

    def test_tee_writer_writes_to_both_streams(self):
        """_TeeWriter should write to both the real stream and the buffer."""
        from io import StringIO
        from cash.notebook.statement_processor import _TeeWriter

        real = StringIO()
        chunks = []
        tee = _TeeWriter(real, chunks)
        tee.write("hello world")
        assert real.getvalue() == "hello world"
        assert tee.getvalue() == "hello world"

    def test_tee_writer_forwards_flush(self):
        """_TeeWriter.flush should flush the real stream."""
        from io import StringIO
        from cash.notebook.statement_processor import _TeeWriter

        real = StringIO()
        chunks = []
        tee = _TeeWriter(real, chunks)
        tee.write("data")
        tee.flush()
        # StringIO doesn't need flushing, but the call shouldn't error
        assert real.getvalue() == "data"
        assert tee.getvalue() == "data"

    def test_tee_writer_getattr_delegation(self):
        """_TeeWriter should forward unknown attributes to the real stream."""
        from io import StringIO
        from cash.notebook.statement_processor import _TeeWriter

        real = StringIO()
        chunks = []
        tee = _TeeWriter(real, chunks)
        # StringIO has 'encoding' attribute — forwarded via __getattr__
        assert hasattr(tee, 'readable')

    def test_tee_writer_batched_flush(self):
        """_TeeWriter should NOT flush on every write — only after the interval."""
        from io import StringIO
        from cash.notebook.statement_processor import _TeeWriter

        flush_count = [0]
        real = StringIO()
        original_flush = real.flush
        def counting_flush():
            flush_count[0] += 1
            original_flush()
        real.flush = counting_flush

        chunks = []
        tee = _TeeWriter(real, chunks)
        # Write many times rapidly — should NOT flush each time
        for i in range(1000):
            tee.write(f"line {i}\n")
        # Should have far fewer flushes than writes
        assert flush_count[0] < 100, f"Too many flushes: {flush_count[0]} for 1000 writes"
        # But all data should be recorded
        assert tee.getvalue().count('\n') == 1000

    def test_tee_output_captures_stdout(self):
        """_tee_output should record stdout while letting it through."""
        import sys
        from io import StringIO
        from cash.notebook.statement_processor import _tee_output

        old_stdout = sys.stdout
        sys.stdout = StringIO()  # Controlled real stream
        try:
            with _tee_output() as teed:
                print("hello")
            assert teed.stdout == "hello\n"
            # The "real" stdout (our StringIO) should also have it
            assert "hello\n" in sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

    def test_tee_output_captures_stderr(self):
        """_tee_output should record stderr while letting it through."""
        import sys
        from io import StringIO
        from cash.notebook.statement_processor import _tee_output

        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            with _tee_output() as teed:
                print("warning", file=sys.stderr)
            assert teed.stderr == "warning\n"
            assert "warning\n" in sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

    def test_tee_output_restores_streams_on_exception(self):
        """_tee_output should restore sys.stdout/stderr even on exception."""
        import sys
        from cash.notebook.statement_processor import _tee_output

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            with _tee_output():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert sys.stdout is orig_stdout
        assert sys.stderr is orig_stderr

    def test_tee_output_has_empty_outputs_list(self):
        """_tee_output's TeedOutput should have an empty outputs list (no rich display capture)."""
        from cash.notebook.statement_processor import _tee_output

        with _tee_output() as teed:
            pass
        assert teed.outputs == []


class TestSingleUnitStreamOutput:
    """Tests for stream_output in _execute_as_single_unit via statement_processor.process_statement()."""

    @pytest.fixture
    def mock_shell(self):
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    @pytest.fixture
    def mock_statement_processor(self):
        processor = MagicMock()
        processor.variable_lineage = {}
        processor.vars_with_mutation_lineage = set()
        processor.compute_hash = MagicMock(return_value='fakehash')
        return processor

    @pytest.fixture
    def control_processor(self, mock_shell, mock_statement_processor):
        from cash.notebook.control_structures import ControlStructureProcessor
        return ControlStructureProcessor(
            mock_shell,
            mock_statement_processor,
            debug=False
        )

    def test_single_unit_passes_stream_output(self, control_processor, mock_shell, mock_statement_processor):
        """_execute_as_single_unit should pass stream_output=True to process()."""
        mock_statement_processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.5,
            'stdout': 'progress\n',
            'stderr': '',
            'outputs': [],
            'code': 'for i in range(100): pass',
            '_output_flushed': True,
        })

        code = "while True:\n    break"
        node = ast.parse(code).body[0]

        control_processor.process(node)

        # Verify process was called with stream_output=True
        call_kwargs = mock_statement_processor.process_statement.call_args
        assert call_kwargs[1].get('stream_output') is True or (len(call_kwargs[0]) > 3 and call_kwargs[0][3] is True)

    def test_single_unit_metrics_output_flushed(self, control_processor, mock_shell, mock_statement_processor):
        """Metrics from single-unit execution should have _output_flushed when stream_output=True."""
        mock_statement_processor.process_statement = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.5,
            'stdout': 'output\n',
            'stderr': '',
            'outputs': [],
            'code': 'while True: break',
            '_output_flushed': True,
        })

        code = "while True:\n    break"
        node = ast.parse(code).body[0]

        result = control_processor.process(node)
        assert result.success is True
        # The metrics should have _output_flushed from the process call
        assert result.metrics[0].get('_output_flushed') is True
