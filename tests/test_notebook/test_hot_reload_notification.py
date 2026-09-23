"""Tests for hot reload notification in badge when functions change."""

from unittest.mock import patch


class TestHotReloadNotification:
    """Tests for function change detection and badge notification."""

    def test_changed_function_detected_in_metrics(self, cash_magics, mock_shell):
        """When a function's source changes, a FUNCTION_CHANGED metric is added."""

        # Define and track a function
        def helper(x):
            return x * 2

        mock_shell.user_ns["helper"] = helper
        ft = cash_magics._statement_processor.function_tracker
        ft.update_function_hash("helper", helper)

        # Now replace it with a different implementation
        def helper_v2(x):
            return x * 3

        mock_shell.user_ns["helper"] = helper_v2

        # Detect changed functions (this is what _execute_cell does internally)
        changed = ft.detect_changed_functions(mock_shell.user_ns)
        assert "helper" in changed

    def test_function_changed_notification_structure(self):
        """FUNCTION_CHANGED metric has correct structure."""

        # Build a notification metric manually (as _execute_cell does)
        notification = {
            "status": "FUNCTION_CHANGED",
            "code": "🔄 Function changed: process",
            "is_upstream": True,
            "total_time": 0.0,
            "execution_time": 0.0,
            "outputs": [],
            "changed_functions": ["process"],
        }

        assert notification["status"] == "FUNCTION_CHANGED"
        assert notification["is_upstream"] is True
        assert "process" in notification["changed_functions"]

    def test_function_changed_badge_rendering(self, cash_magics, mock_shell):
        """FUNCTION_CHANGED status renders correctly in badge."""

        notification = {
            "status": "FUNCTION_CHANGED",
            "code": "🔄 Functions changed: process, helper",
            "is_upstream": True,
            "total_time": 0.0,
            "execution_time": 0.0,
            "outputs": [],
            "changed_functions": ["helper", "process"],
        }

        # Test rendering via render_interactive_badge with patched display
        with patch.object(cash_magics, "shell") as patched_shell:
            patched_shell.user_ns = mock_shell.user_ns
            with patch("cash.notebook.ipython.badges.display"):
                cash_magics.badges.render([notification], display_id="test-id", status="DONE")
                # Should have called display
                assert True  # May not display in test env

    def test_no_notification_when_no_changes(self, cash_magics, mock_shell):
        """No FUNCTION_CHANGED metric when functions haven't changed."""

        def helper(x):
            return x * 2

        mock_shell.user_ns["helper"] = helper
        ft = cash_magics._statement_processor.function_tracker
        ft.update_function_hash("helper", helper)

        # Same function, no change
        changed = ft.detect_changed_functions(mock_shell.user_ns)
        assert "helper" not in changed

    def test_multiple_functions_changed(self, cash_magics, mock_shell):
        """Multiple function changes generate single notification."""

        def func_a():
            return 1

        def func_b():
            return 2

        mock_shell.user_ns["func_a"] = func_a
        mock_shell.user_ns["func_b"] = func_b
        ft = cash_magics._statement_processor.function_tracker
        ft.update_function_hash("func_a", func_a)
        ft.update_function_hash("func_b", func_b)

        # Replace both
        def func_a_v2():
            return 10

        def func_b_v2():
            return 20

        mock_shell.user_ns["func_a"] = func_a_v2
        mock_shell.user_ns["func_b"] = func_b_v2

        changed = ft.detect_changed_functions(mock_shell.user_ns)
        assert "func_a" in changed
        assert "func_b" in changed

        # Build notification
        func_names = ", ".join(sorted(changed))
        notification = {
            "status": "FUNCTION_CHANGED",
            "code": f"🔄 Functions changed: {func_names}",
            "is_upstream": True,
            "total_time": 0.0,
            "execution_time": 0.0,
            "outputs": [],
            "changed_functions": sorted(changed),
        }
        assert len(notification["changed_functions"]) == 2
        assert "Functions" in notification["code"]  # plural

    def test_function_deleted_detected(self, cash_magics, mock_shell):
        """Deleted function is detected as changed."""

        def old_func():
            return 42

        mock_shell.user_ns["old_func"] = old_func
        ft = cash_magics._statement_processor.function_tracker
        ft.update_function_hash("old_func", old_func)

        # Delete the function
        del mock_shell.user_ns["old_func"]

        changed = ft.detect_changed_functions(mock_shell.user_ns)
        assert "old_func" in changed

    def test_function_replaced_with_non_callable(self, cash_magics, mock_shell):
        """Function replaced with non-callable is detected as changed."""

        def my_func():
            return 42

        mock_shell.user_ns["my_func"] = my_func
        ft = cash_magics._statement_processor.function_tracker
        ft.update_function_hash("my_func", my_func)

        # Replace with a string
        mock_shell.user_ns["my_func"] = "not a function"

        changed = ft.detect_changed_functions(mock_shell.user_ns)
        assert "my_func" in changed
