"""Interactive matplotlib dashboard for cache performance visualisation."""

from __future__ import annotations

__all__ = ["show_analytics_dashboard"]

import logging

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    import base64
    import io

    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..analytics import AnalyticsManager

logger = logging.getLogger(__name__)

def show_analytics_dashboard(mgr: AnalyticsManager | None = None):
    """
    Display an interactive dashboard for Cash analytics.

    Args:
        mgr: Optional AnalyticsManager instance. If None, a new one is created.
    """
    if not HAS_WIDGETS:
        print("ipywidgets is required for the dashboard.")
        return

    if mgr is None:
        mgr = AnalyticsManager()

    # Style

    # Header
    header = widgets.HTML("<h2>Cash Analytics Dashboard</h2>")

    # Stats Area
    stats_output = widgets.Output()

    # Controls
    refresh_btn = widgets.Button(
        description='Refresh',
        icon='refresh',
        button_style='info',
        tooltip='Refresh stats'
    )

    def render_stats(_=None):
        with stats_output:
            clear_output(wait=True)

            # Fetch data
            session_stats = mgr.get_session_stats()
            global_stats = mgr.get_global_stats()
            daily_savings = mgr.get_daily_savings()

            # Session Stats Widget
            s_hits = session_stats.get('hits', 0)
            session_stats.get('misses', 0)
            s_total = session_stats.get('total_events', 0)
            s_saved = session_stats.get('total_saved_time', 0)
            s_rate = (s_hits / s_total * 100) if s_total > 0 else 0

            session_html = f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-right: 10px; width: 45%; display: inline-block; vertical-align: top;">
                <h3>Current Session</h3>
                <p><b>Total Events:</b> {s_total}</p>
                <p><b>Cache Hits:</b> {s_hits} ({s_rate:.1f}%)</p>
                <p><b>Time Saved:</b> <span style="color: green; font-weight: bold;">{s_saved:.2f}s</span></p>
            </div>
            """

            # Global Stats Widget
            g_hits = global_stats.get('hits', 0)
            g_total = global_stats.get('total_events', 0)
            g_saved = global_stats.get('total_saved_time', 0)
            g_rate = (g_hits / g_total * 100) if g_total > 0 else 0
            g_sessions = global_stats.get('total_sessions', 0)

            global_html = f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; width: 45%; display: inline-block; vertical-align: top;">
                <h3>All Time</h3>
                <p><b>Sessions:</b> {g_sessions}</p>
                <p><b>Total Events:</b> {g_total}</p>
                <p><b>Hit Rate:</b> {g_rate:.1f}%</p>
                <p><b>Total Time Saved:</b> <span style="color: green; font-weight: bold;">{g_saved:.2f}s</span></p>
            </div>
            """

            display(widgets.HTML(session_html + global_html))

            # Plot Daily Savings
            if HAS_MATPLOTLIB and daily_savings:
                dates = [d[0] for d in daily_savings]
                savings = [d[1] for d in daily_savings]

                # Check if we have enough data (at least one point)
                if dates:
                    plt.figure(figsize=(8, 3))
                    plt.bar(dates, savings, color='#006644', alpha=0.7)
                    plt.title('Daily Time Savings (Last 7 Days)')
                    plt.ylabel('Seconds')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.grid(axis='y', linestyle='--', alpha=0.3)

                    # Capture plot to widget
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('ascii')
                    img_html = f'<div style="margin-top: 20px; text-align: center;"><img src="data:image/png;base64,{img_str}" /></div>'
                    display(widgets.HTML(img_html))
                    plt.close()
            elif not HAS_MATPLOTLIB:
                display(widgets.HTML("<p><i>Matplotlib not installed, skipping charts.</i></p>"))
            elif not daily_savings:
                display(widgets.HTML("<p><i>No historical data available for charts yet.</i></p>"))

    refresh_btn.on_click(render_stats)

    # Initial Render
    render_stats()

    # Layout
    display(widgets.VBox([
        widgets.HBox([header, refresh_btn], layout=widgets.Layout(justify_content='space-between', align_items='center')),
        stats_output
    ]))
