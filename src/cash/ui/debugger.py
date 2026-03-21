"""Cache debugging utilities — replay, inspect hashes, and trace misses."""

from __future__ import annotations

__all__ = ["CacheDebugger"]

import ast
import hashlib
import logging
import pickle
import types
from typing import Any

from ..exceptions import CashError
from ..notebook.analysis import CodeAnalyzer

logger = logging.getLogger(__name__)

class CacheDebugger:
    def __init__(self, shell, cash_instance=None):
        self.shell = shell

        # Handle case where user passes the module instead of instance
        if isinstance(cash_instance, types.ModuleType):
            cash_instance = None

        if cash_instance is None and hasattr(shell, 'magics_manager') and shell.magics_manager:
            magics = shell.magics_manager.magics.get('cell', {}).get('cash')
            # 'magics' is likely a bound method: CashMagics.cash
            if magics and hasattr(magics, '__self__'):
                cash_magics = magics.__self__
                if hasattr(cash_magics, '_cash_instance'):
                    cash_instance = cash_magics._cash_instance
            # Fallback: maybe it's the object itself (unlikely for standard magics)
            elif magics and hasattr(magics, '_cash_instance'):
                cash_instance = magics._cash_instance

        if cash_instance is None:
            raise CashError("Could not find Cash instance. Please pass the 'cash' instance (not the module) or ensure %cash_enable has been run.")

        self.cash = cash_instance

    @staticmethod
    def _unwrap_run_cell_magic(code: str) -> str:
        """Extract inner code from get_ipython().run_cell_magic('cash', '', '<code>')."""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if (isinstance(call.func, ast.Attribute) and
                            call.func.attr == 'run_cell_magic' and
                            len(call.args) >= 3 and
                            isinstance(call.args[0], ast.Constant) and
                            call.args[0].value == 'cash' and
                            isinstance(call.args[2], ast.Constant)):
                        return call.args[2].value
        except Exception as exc:
            logger.debug("[DEBUGGER] Failed to parse run_cell_magic code: %s", exc)
        return code

    @staticmethod
    def _unwrap_run_line_magic(code: str) -> str:
        """Reconstruct %magic_name arg from get_ipython().run_line_magic(...)."""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if (isinstance(call.func, ast.Attribute) and
                            call.func.attr == 'run_line_magic' and
                            len(call.args) >= 2):
                        return f"%{call.args[0].value} {call.args[1].value}"
        except Exception as exc:
            logger.debug("[DEBUGGER] Failed to parse run_line_magic code: %s", exc)
        return code

    def _preprocess_cell_code(self, cell_code: str) -> str:
        """Strip IPython magic lines and unwrap get_ipython() transform wrappers."""
        code = '\n'.join(
            line for line in cell_code.split('\n')
            if not line.lstrip().startswith(('%', '%%'))
        )
        if 'get_ipython' in code and 'run_cell_magic' in code:
            code = self._unwrap_run_cell_magic(code)
        if 'get_ipython' in code and 'run_line_magic' in code:
            code = self._unwrap_run_line_magic(code)
        return code

    def _hash_input_var(
        self,
        var_name: str,
        val: Any,
        input_details: dict[str, str],
        input_hashes: list[str],
    ) -> None:
        """Compute hash for one input variable and populate input_details/input_hashes."""
        # Skip modules, get_ipython, and IPython magic functions
        if isinstance(val, types.ModuleType) or var_name == 'get_ipython':
            return
        if callable(val) and (var_name.startswith('_') or hasattr(val, '__self__')):
            return

        try:
            if hasattr(val, '_cash_lineage_hash'):
                h = val._cash_lineage_hash
                input_details[var_name] = f"Hash (from metadata): {h[:8]}..."
            else:
                h = hashlib.sha256(pickle.dumps(val)).hexdigest()
                input_details[var_name] = f"Hash (computed): {h[:8]}..."
            input_hashes.append(h)
        except (AttributeError, TypeError, RecursionError, pickle.PicklingError) as e:
            input_details[var_name] = f"Skipped (unpicklable): {type(e).__name__}"
        except Exception as e:
            input_details[var_name] = f"Error hashing: {e}"
            input_hashes.append(f"ERROR:{e}")

    def debug(self, cell_code: str) -> list[dict[str, Any]]:
        """
        Analyze a cell's code and return cache status for each statement.
        Does NOT execute the code.
        """
        cleaned_code = self._preprocess_cell_code(cell_code)

        try:
            tree = ast.parse(cleaned_code)
        except SyntaxError as e:
            return [{'error': f"Syntax Error: {e}"}]

        results = []
        user_ns = self.shell.user_ns

        for node in tree.body:
            try:
                stmt_code = ast.unparse(node)
            except Exception as exc:
                logger.debug("[DEBUGGER] Failed to unparse AST node: %s", exc)
                continue

            inputs, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
            source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()

            input_details: dict[str, str] = {}
            input_hashes: list[str] = []
            for var_name in sorted(inputs):
                if var_name in user_ns:
                    self._hash_input_var(var_name, user_ns[var_name], input_details, input_hashes)
                else:
                    input_details[var_name] = "Missing from namespace"

            combined_hash = hashlib.sha256(
                f"{source_hash}:{':'.join(input_hashes)}".encode()
            ).hexdigest()
            cache_key = f"stmt:{combined_hash}"

            metadata, cached_data = self.cash.backend.get(cache_key)
            status = "HIT" if cached_data else "MISS"
            execution_time = metadata.get('execution_time') if metadata else None
            memory_size = metadata.get('memory_size') if metadata else None

            results.append({
                'code': stmt_code,
                'inputs': list(inputs),
                'input_details': input_details,
                'cache_key': cache_key,
                'status': status,
                'outputs': list(outputs),
                'execution_time': execution_time,
                'memory_size': memory_size,
            })

        return results

