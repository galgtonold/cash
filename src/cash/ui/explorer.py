"""Cache content explorer — inspect, search, and manage stored entries."""

from __future__ import annotations

__all__ = ["CacheExplorer"]

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if TYPE_CHECKING:
    from ..core import Cash

logger = logging.getLogger(__name__)

class CacheExplorer:
    """Inspect, search, and manage entries stored in a `Cash` backend.

    Lets you list every cached entry with timestamps and source code,
    preview cached values, or clear entries for a specific function —
    all without touching the backend directly. Useful in notebooks
    for answering "what's actually in my cache?" and in scripts for
    targeted cache management.

    Typically obtained via `Cash.explorer()` rather than constructed
    directly:

    ```python
    import cash
    c = cash.Cash()
    explorer = c.explorer()
    explorer.list_entries()         # all entries with metadata
    explorer.to_dataframe()         # same, as a pandas DataFrame
    explorer.get_preview(key)       # peek at a stored value
    explorer.clear_function("my_module.my_func")  # surgical clear
    ```

    Status: experimental. Import via `cash.experimental.CacheExplorer`
    if instantiating outside `Cash.explorer()`.
    """

    def __init__(self, cash_app: Cash) -> None:
        """
        Args:
            cash_app: The `Cash` instance whose backend will be inspected.
        """
        self.app = cash_app

    def list_entries(self) -> list[dict[str, Any]]:
        """
        List all entries in the cache with metadata.
        """
        entries = self.app.backend.list_entries()

        # Enrich with human readable info
        for entry in entries:
            if 'timestamp' in entry:
                entry['timestamp_human'] = datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

            # Add source code if available in memory
            func_name = entry.get('func_name')
            if func_name and func_name in self.app.functions:
                import inspect
                try:
                    entry['source_code'] = inspect.getsource(self.app.functions[func_name])
                except (TypeError, OSError):
                    entry['source_code'] = "Source not available"
            else:
                entry['source_code'] = "Function not loaded"

        return entries

    def clear_function(self, func_name: str) -> int:
        """
        Clear all cache entries for a specific function.
        Returns the number of deleted entries.
        """
        count = 0
        entries = self.list_entries()
        for entry in entries:
            if entry.get('func_name') == func_name:
                self.app.backend.delete(entry['key'])
                count += 1
        return count


    def get_preview(self, key: str) -> str:
        """
        Get a string preview of the cached value.
        """
        metadata, value_bytes = self.app.backend.get(key)
        if value_bytes is None:
            return "Value not found in cache."

        try:
            # Deserialize
            from ..backends.serialization import PickleSerializer
            # We need to import other serializers if used, but for now assume Pickle or try to load class
            # The metadata has 'serializer_cls', but it's a string representation or class?
            # In core.py: 'serializer_cls': type(serializer) -> <class '...'>
            # We need to instantiate it.

            # Simple fallback: use PickleSerializer if default
            # Or try to find the class.
            # For now, let's just use PickleSerializer as it's the default.
            serializer = PickleSerializer()
            value = serializer.deserialize(value_bytes)

            if hasattr(value, 'head'): # DataFrame/Series
                return str(value.head())
            return str(value)[:1000]
        except Exception as e:
            return f"Error previewing data: {e}"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return cache entries as a pandas DataFrame.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install cash-lib[pandas]")
        entries = self.list_entries()
        if not entries:
            return pd.DataFrame()

        df = pd.DataFrame(entries)

        # Reorder columns for better readability
        cols = ['func_name', 'timestamp_human', 'ttl', 'key', 'args_hash', 'state_hash']
        # Add other columns that might exist
        existing_cols = [c for c in cols if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in cols and c != 'source_code']

        return df[existing_cols + remaining_cols]

    def _widget_html(self, height: str = "600px") -> Any | None:
        """
        Return an interactive widget for Jupyter with hierarchical view (HTML/JS version).
        """
        try:
            import base64
            import json

            from IPython.display import IFrame
        except ImportError:
            logger.warning("IPython is required for the widget.")
            return None

        entries = self.list_entries()

        # Process entries into hierarchy: Module -> Function -> Entries
        hierarchy = {}
        for entry in entries:
            func_name = entry.get('func_name', 'Unknown')
            # Heuristic to extract module: assume func_name is "module.submodule.function"
            parts = func_name.rsplit('.', 1)
            if len(parts) == 2:
                module, func = parts
            else:
                module = "Global"
                func = func_name

            if module not in hierarchy:
                hierarchy[module] = {}
            if func not in hierarchy[module]:
                hierarchy[module][func] = {'entries': [], 'total_size': 0}

            hierarchy[module][func]['entries'].append(entry)
            hierarchy[module][func]['total_size'] += entry.get('size', 0)

        # Convert to list for JSON
        tree_data = []
        for module, funcs in hierarchy.items():
            module_node = {
                'name': module,
                'type': 'module',
                'children': []
            }
            for func, data in funcs.items():
                func_node = {
                    'name': func,
                    'type': 'function',
                    'stats': {
                        'count': len(data['entries']),
                        'total_size': data['total_size']
                    },
                    'entries': data['entries']
                }
                module_node['children'].append(func_node)
            tree_data.append(module_node)

        # Prepare data for JS
        data = json.dumps(tree_data, default=str)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; overflow: hidden; }}
                .container {{ display: flex; width: 100%; height: 100%; }}
                .sidebar {{ width: 30%; border-right: 1px solid #ccc; overflow-y: auto; background: #f8f9fa; display: flex; flex-direction: column; }}
                .main-view {{ width: 70%; display: flex; flex-direction: column; height: 100%; }}

                .tree-item {{ padding: 8px 12px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
                .tree-item:hover {{ background: #e9ecef; }}
                .tree-item.active {{ background: #e7f1ff; color: #007bff; }}
                .tree-module {{ font-weight: bold; color: #333; background: #eee; border-bottom: 1px solid #ddd; }}
                .tree-func {{ padding-left: 24px; font-size: 14px; }}

                .tab-bar {{ display: flex; border-bottom: 1px solid #ddd; background: #f1f1f1; }}
                .tab {{ padding: 10px 20px; cursor: pointer; border-right: 1px solid #ddd; background: #f1f1f1; }}
                .tab.active {{ background: white; border-bottom: 2px solid #007bff; font-weight: bold; }}

                .tab-content {{ flex: 1; overflow-y: auto; padding: 20px; display: none; }}
                .tab-content.active {{ display: block; }}

                .stat-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .stat-title {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }}

                .entry-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
                .entry-table th {{ text-align: left; padding: 8px; border-bottom: 2px solid #ddd; background: #f9f9f9; position: sticky; top: 0; }}
                .entry-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                .entry-table tr:hover {{ background: #f5f5f5; cursor: pointer; }}
                .entry-table tr.active {{ background: #e7f1ff; }}

                .detail-panel {{ border-top: 1px solid #ddd; height: 40%; overflow-y: auto; padding: 20px; background: white; display: none; }}
                .detail-panel.visible {{ display: block; }}

                .meta-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 10px; margin-bottom: 20px; }}
                .meta-label {{ font-weight: bold; color: #555; }}
                .meta-value {{ font-family: monospace; word-break: break-all; }}
                .source-code {{ background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: monospace; white-space: pre; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="sidebar" id="sidebar"></div>
                <div class="main-view">
                    <div class="tab-bar">
                        <div class="tab active" onclick="switchTab('overview')">Overview</div>
                        <div class="tab" onclick="switchTab('entries')">Entries</div>
                    </div>

                    <div id="overview-tab" class="tab-content active">
                        <div style="color: #999; text-align: center; margin-top: 50px;">Select a function to view stats</div>
                    </div>

                    <div id="entries-tab" class="tab-content" style="padding: 0; display: flex; flex-direction: column;">
                        <div style="flex: 1; overflow-y: auto;">
                            <table class="entry-table">
                                <thead>
                                    <tr>
                                        <th>Key</th>
                                        <th>Time</th>
                                        <th>Size</th>
                                    </tr>
                                </thead>
                                <tbody id="entry-table-body"></tbody>
                            </table>
                        </div>
                        <div id="detail-panel" class="detail-panel">
                            <div style="color: #999; text-align: center;">Select an entry to view details</div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const treeData = {data};
                let selectedFunc = null;
                let selectedEntry = null;

                function formatBytes(bytes, decimals = 2) {{
                    if (bytes === 0) return '0 Bytes';
                    const k = 1024;
                    const dm = decimals < 0 ? 0 : decimals;
                    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
                }}

                function renderTree() {{
                    const sidebar = document.getElementById('sidebar');
                    sidebar.innerHTML = '';

                    treeData.forEach(module => {{
                        const modDiv = document.createElement('div');
                        modDiv.className = 'tree-item tree-module';
                        modDiv.textContent = module.name;
                        sidebar.appendChild(modDiv);

                        module.children.forEach(func => {{
                            const funcDiv = document.createElement('div');
                            funcDiv.className = 'tree-item tree-func';
                            funcDiv.textContent = func.name;
                            funcDiv.onclick = () => selectFunction(func, funcDiv);
                            sidebar.appendChild(funcDiv);
                        }});
                    }});
                }}

                function selectFunction(func, element) {{
                    selectedFunc = func;

                    // Update active state
                    document.querySelectorAll('.tree-item').forEach(el => el.classList.remove('active'));
                    element.classList.add('active');

                    // Render Overview
                    const overview = document.getElementById('overview-tab');
                    overview.innerHTML = `
                        <h2>${{func.name}}</h2>
                        <div class="stat-card">
                            <div class="stat-title">Cached Entries</div>
                            <div class="stat-value">${{func.stats.count}}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-title">Total Size</div>
                            <div class="stat-value">${{formatBytes(func.stats.total_size)}}</div>
                        </div>
                    `;

                    // Render Entries Table
                    const tbody = document.getElementById('entry-table-body');
                    tbody.innerHTML = '';
                    func.entries.forEach(entry => {{
                        const tr = document.createElement('tr');
                        tr.onclick = () => selectEntryItem(entry, tr);
                        tr.innerHTML = `
                            <td>${{entry.key.substring(0, 20)}}...</td>
                            <td>${{entry.timestamp_human || 'N/A'}}</td>
                            <td>${{formatBytes(entry.size || 0)}}</td>
                        `;
                        tbody.appendChild(tr);
                    }});

                    // Clear details
                    document.getElementById('detail-panel').innerHTML = '<div style="color: #999; text-align: center;">Select an entry to view details</div>';
                    document.getElementById('detail-panel').classList.remove('visible');
                }}

                function selectEntryItem(entry, element) {{
                    selectedEntry = entry;

                    document.querySelectorAll('.entry-table tr').forEach(el => el.classList.remove('active'));
                    element.classList.add('active');

                    const panel = document.getElementById('detail-panel');
                    panel.classList.add('visible');
                    panel.innerHTML = `
                        <h3>Entry Details</h3>
                        <div class="meta-grid">
                            <div class="meta-label">Key</div>
                            <div class="meta-value">${{entry.key}}</div>

                            <div class="meta-label">Timestamp</div>
                            <div class="meta-value">${{entry.timestamp_human}}</div>

                            <div class="meta-label">Size</div>
                            <div class="meta-value">${{formatBytes(entry.size || 0)}}</div>

                            <div class="meta-label">Args Hash</div>
                            <div class="meta-value">${{entry.args_hash || 'N/A'}}</div>
                        </div>
                        <h4>Source Code</h4>
                        <div class="source-code">${{escapeHtml(entry.source_code || 'No source available')}}</div>
                    `;
                }}

                function switchTab(tabName) {{
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => {{
                        c.classList.remove('active');
                        c.style.display = 'none';
                    }});

                    // Find the tab element by text content (simple hack)
                    const tabs = document.querySelectorAll('.tab');
                    for(let t of tabs) {{
                        if(t.textContent.toLowerCase() === tabName) t.classList.add('active');
                    }}

                    const content = document.getElementById(tabName + '-tab');
                    content.classList.add('active');
                    content.style.display = tabName === 'entries' ? 'flex' : 'block';
                }}

                function escapeHtml(text) {{
                    const map = {{
                        '&': '&amp;',
                        '<': '&lt;',
                        '>': '&gt;',
                        '"': '&quot;',
                        "'": '&#039;'
                    }};
                    return text.replace(/[&<>"']/g, function(m) {{ return map[m]; }});
                }}

                renderTree();
            </script>
        </body>
        </html>
        """

        html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{html_base64}"

        return IFrame(src=data_uri, width="100%", height=height)

    def _build_widget_layout(self, widgets: Any, func_options: list) -> tuple:
        """Create and return all ipywidgets and the top-level layout.

        Returns:
            (func_selector, overview_output, clear_func_btn,
             entries_selector, delete_entry_btn, entry_details,
             preview_output, app_layout)
        """
        from IPython.display import clear_output  # noqa: F401

        func_selector = widgets.Select(
            options=sorted(func_options),
            description='Functions:',
            layout=widgets.Layout(width='100%', height='100%')
        )

        # Tab 1: Overview
        overview_output = widgets.Output()
        clear_func_btn = widgets.Button(
            description='Clear Function Cache',
            button_style='danger',
            icon='trash'
        )

        # Tab 2: Entries
        entries_selector = widgets.Select(
            options=[],
            description='Entries:',
            layout=widgets.Layout(width='100%', height='150px')
        )
        delete_entry_btn = widgets.Button(
            description='Delete Entry',
            button_style='warning',
            icon='times'
        )
        entry_details = widgets.HTML()

        # Tab 3: Preview (Placeholder)
        preview_output = widgets.Output()
        with preview_output:
            print("Select an entry to preview data (Coming Soon)")

        tabs = widgets.Tab(children=[
            widgets.VBox([overview_output, clear_func_btn]),
            widgets.VBox([entries_selector, delete_entry_btn, entry_details]),
            preview_output
        ])
        tabs.set_title(0, 'Overview')
        tabs.set_title(1, 'Entries')
        tabs.set_title(2, 'Preview')

        main_view = widgets.VBox([tabs], layout=widgets.Layout(width='70%', padding='10px'))
        sidebar = widgets.VBox([func_selector], layout=widgets.Layout(width='30%'))
        app_layout = widgets.HBox([sidebar, main_view], layout=widgets.Layout(height='500px', border='1px solid #ccc'))

        return (func_selector, overview_output, clear_func_btn,
                entries_selector, delete_entry_btn, entry_details,
                preview_output, app_layout)

    def _make_func_select_handler(self, overview_output: Any, entries_selector: Any, entry_details: Any) -> Any:
        """Return the on_func_select closure for the widget."""
        from IPython.display import clear_output

        def on_func_select(change):
            if not change['new']:
                return
            full_name = change['new']
            if '.' in full_name:
                module, func = full_name.rsplit('.', 1)
            else:
                module, func = "Global", full_name

            if module in self._hierarchy and func in self._hierarchy[module]:
                data = self._hierarchy[module][func]
                with overview_output:
                    clear_output()
                    print(f"Function: {func}")
                    print(f"Module: {module}")
                    print(f"Cached Entries: {len(data['entries'])}")
                    print(f"Total Size: {self._format_bytes(data['total_size'])}")

                entry_options = []
                for e in data['entries']:
                    time_str = e.get('timestamp_human', 'N/A')
                    key_short = e['key'][:8] + '...'
                    entry_options.append((f"{time_str} ({key_short})", e['key']))
                entries_selector.options = entry_options
                entries_selector.value = None
                entry_details.value = ""

        return on_func_select

    def _make_entry_select_handler(self, entry_details: Any, preview_output: Any) -> Any:
        """Return the on_entry_select closure for the widget."""
        from IPython.display import clear_output

        def on_entry_select(change):
            key = change['new']
            if not key:
                return
            entry = next((e for e in self._entries if e['key'] == key), None)
            if entry:
                details_html = f"""
                <b>Key:</b> {entry['key']}<br>
                <b>Timestamp:</b> {entry.get('timestamp_human')}<br>
                <b>Size:</b> {self._format_bytes(entry.get('size', 0))}<br>
                <b>TTL:</b> {entry.get('ttl')}<br>
                <hr>
                <b>Source Code:</b><br>
                <pre style="background-color: #f4f4f4; padding: 5px;">{entry.get('source_code', 'N/A')}</pre>
                """
                entry_details.value = details_html
                with preview_output:
                    clear_output()
                    print(f"Loading preview for {key}...")
                    preview_text = self.get_preview(key)
                    clear_output()
                    print(preview_text)

        return on_entry_select

    def _make_clear_func_handler(self, func_selector: Any, overview_output: Any) -> Any:
        """Return the on_clear_func closure for the widget."""
        from IPython.display import clear_output

        def on_clear_func(b):
            full_name = func_selector.value
            if not full_name:
                return
            # Extract func name (heuristic, might need better mapping)
            # The list_entries logic used func_name directly.
            target_func_name = full_name
            if full_name.startswith("Global."):
                target_func_name = full_name[7:]  # Remove "Global."
            count = self.clear_function(target_func_name)
            self._refresh_data()
            with overview_output:
                clear_output()
                print(f"Cleared {count} entries for {target_func_name}")

        return on_clear_func

    def _make_delete_entry_handler(self, entries_selector: Any) -> Any:
        """Return the on_delete_entry closure for the widget."""
        def on_delete_entry(b):
            key = entries_selector.value
            if not key:
                return
            self.app.backend.delete(key)
            # Refresh
            self._refresh_data()
            # Try to keep func selection
            # on_func_select will trigger if value is set?
            # self._refresh_data re-sets options, which might clear value.

        return on_delete_entry

    def _wire_widget_events(self, func_selector: Any, overview_output: Any, clear_func_btn: Any,
                            entries_selector: Any, delete_entry_btn: Any, entry_details: Any,
                            preview_output: Any) -> None:
        """Wire up event handlers for the explorer widget."""
        on_func_select = self._make_func_select_handler(overview_output, entries_selector, entry_details)
        on_entry_select = self._make_entry_select_handler(entry_details, preview_output)
        on_clear_func = self._make_clear_func_handler(func_selector, overview_output)
        on_delete_entry = self._make_delete_entry_handler(entries_selector)

        func_selector.observe(on_func_select, names='value')
        entries_selector.observe(on_entry_select, names='value')
        clear_func_btn.on_click(on_clear_func)
        delete_entry_btn.on_click(on_delete_entry)

    def widget(self) -> Any:
        """
        Return an interactive widget for Jupyter using ipywidgets.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display  # noqa: F401
        except ImportError:
            return self._widget_html()

        # State
        self._entries = self.list_entries()
        self._hierarchy = self._build_hierarchy(self._entries)

        func_options = []
        for module, funcs in self._hierarchy.items():
            for func in funcs:
                func_options.append(f"{module}.{func}")

        (func_selector, overview_output, clear_func_btn,
         entries_selector, delete_entry_btn, entry_details,
         preview_output, app_layout) = self._build_widget_layout(widgets, func_options)

        self._wire_widget_events(func_selector, overview_output, clear_func_btn,
                                 entries_selector, delete_entry_btn, entry_details,
                                 preview_output)

        return app_layout

    def _refresh_data(self) -> None:
        self._entries = self.list_entries()
        self._hierarchy = self._build_hierarchy(self._entries)
        # Update func selector options
        func_options = []
        for module, funcs in self._hierarchy.items():
            for func in funcs:
                func_options.append(f"{module}.{func}")
        # TODO: Update widget options without losing selection if possible
        # For now, just simplistic refresh

    def _build_hierarchy(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        hierarchy = {}
        for entry in entries:
            func_name = entry.get('func_name', 'Unknown')
            parts = func_name.rsplit('.', 1)
            if len(parts) == 2:
                module, func = parts
            else:
                module = "Global"
                func = func_name

            if module not in hierarchy:
                hierarchy[module] = {}
            if func not in hierarchy[module]:
                hierarchy[module][func] = {'entries': [], 'total_size': 0}

            hierarchy[module][func]['entries'].append(entry)
            hierarchy[module][func]['total_size'] += entry.get('size', 0)
        return hierarchy

    def _format_bytes(self, bytes_val: int, decimals: int = 2) -> str:
        if bytes_val == 0:
            return '0 Bytes'
        k = 1024
        dm = max(decimals, 0)
        sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        import math
        i = math.floor(math.log(bytes_val) / math.log(k))
        return f"{bytes_val / (k**i):.{dm}f} {sizes[i]}"


    def show(self) -> None:
        """
        Display the cache entries in an interactive way (if in notebook) or print them.
        """
        try:
            from IPython.display import display
            # Try to display widget if in notebook
            # How to detect?
            # Usually if IPython is available we can try.
            display(self.widget())
        except ImportError:
            # Fallback for non-notebook environments
            df = self.to_dataframe()
            if df.empty:
                print("Cache is empty.")
            else:
                print(df.to_string())
