from cash.analysis.code_analyzer import CodeAnalyzer


class TestForbidden:
    def test_forbidden_detection(self):
        # Case 1: time.time() with no imports in code (assuming user_ns or builtins?)
        # Actually my implementation relies on resolving Names.
        # If 'time' is not in user_ns and not imported, it won't resolve.
        # But usually in a notebook 'import time' is executed first.

        # Test 1: Import inside code
        import textwrap

        code1 = textwrap.dedent("""
        import time
        t = time.time()
        """)
        reasons = CodeAnalyzer.scan_for_forbidden_functions(code1, {})
        assert "time.time" in reasons

        # Test 2: Import from inside code
        code2 = textwrap.dedent("""
        from datetime import datetime
        n = datetime.now()
        """)
        reasons = CodeAnalyzer.scan_for_forbidden_functions(code2, {})
        assert "datetime.now" in reasons

        # Test 3: Alias
        code3 = textwrap.dedent("""
        import time as t
        x = t.monotonic()
        """)
        reasons = CodeAnalyzer.scan_for_forbidden_functions(code3, {})
        assert "time.monotonic" in reasons

        # Test 4: Existing user_ns
        import time

        user_ns = {"time": time}
        code4 = "time.perf_counter()"
        reasons = CodeAnalyzer.scan_for_forbidden_functions(code4, user_ns)
        assert "time.perf_counter" in reasons

        # Test 5: Safe code
        code5 = "x = 1 + 1"
        reasons = CodeAnalyzer.scan_for_forbidden_functions(code5, {})
        assert len(reasons) == 0
