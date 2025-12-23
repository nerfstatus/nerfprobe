"""Main TUI Application."""

import json
from datetime import datetime
from typing import List

from crontab import CronTab
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from nerfprobe.storage import ResultStore, BaselineStore


class Dashboard(Container):
    """Dashboard view showing recent runs."""

    def compose(self) -> ComposeResult:
        yield Label("Recent Runs", id="dashboard-title", classes="section-title")
        yield DataTable(id="recent_runs_table")

    def on_mount(self) -> None:
        table = self.query_one("#recent_runs_table", DataTable)
        table.add_columns("Time", "Model", "Probe", "Status", "Score", "Latency")
        self.load_data()

    def load_data(self) -> None:
        store = ResultStore()
        table = self.query_one("#recent_runs_table", DataTable)
        table.clear()
        
        recent = store.get_recent(limit=50)
        for r in recent:
            # Parse time
            ts = r.get("stored_at", "")
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%H:%M:%S")
            except ValueError:
                time_str = ts

            target = r.get("target", {})
            model = target.get("model_name", "unknown")
            probe = r.get("probe_name", "unknown")
            passed = r.get("passed", False)
            score = r.get("score", 0.0)
            latency = r.get("latency_ms", 0.0)

            status = Text("PASS", style="green") if passed else Text("FAIL", style="red")
            
            table.add_row(
                time_str,
                model,
                probe,
                status,
                f"{score:.2f}",
                f"{latency:.0f}ms",
            )


class BaselineView(Container):
    """View current baselines."""

    def compose(self) -> ComposeResult:
        yield Label("Baselines", classes="section-title")
        yield DataTable(id="baselines_table")
        yield Button("Refresh", id="refresh_baselines")

    def on_mount(self) -> None:
        table = self.query_one("#baselines_table", DataTable)
        table.add_columns("Model", "Probe", "Avg Score", "Samples", "Last Updated")
        self.load_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh_baselines":
            self.load_data()

    def load_data(self) -> None:
        store = BaselineStore()
        table = self.query_one("#baselines_table", DataTable)
        table.clear()
        
        # We need a way to look into the file structure roughly since get_all is not standard
        # But we can access the file directly or add functionality.
        # Let's rely on _load private method or similar logic for now
        # Actually storage.py's get_model_baselines returns dict for one model.
        # But we want ALL. Let's just read the file here as we are in the app
        
        path = store.baseline_file
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)
                
            for model, probes in data.items():
                for probe, info in probes.items():
                    table.add_row(
                        model,
                        probe,
                        f"{info.get('score', 0):.2f}",
                        str(info.get('samples', 0)),
                        info.get('last_updated', '')
                    )
        except Exception:
            pass


class ScheduleView(Container):
    """View active schedules."""

    def compose(self) -> ComposeResult:
        yield Label("Automated Schedules (Cron)", classes="section-title")
        yield DataTable(id="schedule_table")
        yield Button("Refresh", id="refresh_schedule")

    def on_mount(self) -> None:
        table = self.query_one("#schedule_table", DataTable)
        table.add_columns("Job/Comment", "Schedule", "Command")
        self.load_data()
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh_schedule":
            self.load_data()

    def load_data(self) -> None:
        try:
            cron = CronTab(user=True)
            table = self.query_one("#schedule_table", DataTable)
            table.clear()
            
            for job in cron:
                if job.comment.startswith("nerfprobe"):
                    table.add_row(
                        job.comment,
                        str(job.slices),
                        job.command
                    )
        except Exception:
            pass


class NerfProbeApp(App):
    """NerfProbe Terminal Dashboard."""

    CSS = """
    Screen {
        layout: vertical;
    }
    .section-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        background: $accent;
        color: $text;
        width: 100%;
    }
    DataTable {
        height: 1fr;
        border: solid $secondary;
    }
    Button {
        margin: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh All"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard", id="tab_dashboard"):
                yield Dashboard()
            with TabPane("Baselines", id="tab_baselines"):
                yield BaselineView()
            with TabPane("Schedule", id="tab_schedule"):
                yield ScheduleView()
        yield Footer()

    def action_refresh(self) -> None:
        self.query_one(Dashboard).load_data()
        self.query_one(BaselineView).load_data()
        self.query_one(ScheduleView).load_data()


if __name__ == "__main__":
    app = NerfProbeApp()
    app.run()
