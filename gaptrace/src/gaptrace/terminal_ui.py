"""Terminal activity from real execution events; never model private reasoning."""
from __future__ import annotations

from contextvars import ContextVar
from time import perf_counter
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

_listener: ContextVar[Callable | None] = ContextVar("terminal_activity", default=None)


def emit(message: str, **details) -> None:
    listener = _listener.get()
    if listener is not None:
        listener(message, **details)


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02}m" if hours else f"{minutes}m {seconds:02}s" if minutes else f"{seconds}s"


def banner(data: str) -> None:
    console = Console(stderr=True, highlight=False)
    logo = Text("  GapTracer", style="bold cyan")
    logo.append("\n  Research from evidence", style="dim")
    console.print(Panel(Group(logo, Text(f"\nCorpus: {data}"), Text(
        'research                  Choose a topic and investigate gaps\n'
        'ask "your question"       Answer using the active corpus\n'
        'status / gaps / help      Explore the corpus and commands\n'
        'use "PATH" / exit         Switch corpus or close the session', style="dim")),
        border_style="cyan", title="Welcome", subtitle="local research workspace"))


class Activity:
    """Live TTY footer, or stable timestamped lines when redirected.

    Estimates apply only to a model request of the same schema and model in
    this operation. Variable tool loops have no known total and no invented ETA.
    """

    def __init__(self, title: str, model: str = "", *, clock=perf_counter, console=None):
        self.title, self.model, self.clock = title, model, clock
        self.console = console or Console(stderr=True, highlight=False)
        self.started = clock()
        self.message = "Starting"
        self.samples: dict[tuple, list[float]] = {}
        self.request_started = None
        self.request_key = None
        self.calls = self.cache_hits = 0
        self.live = None

    def __enter__(self):
        self.started = self.clock()
        self.token = _listener.set(self.event)
        self.console.print(Text(f"{self.title}" + (f" | model: {self.model}" if self.model else ""), style="bold cyan"))
        if self.console.is_terminal:
            self.live = Live(self, console=self.console, refresh_per_second=4, transient=True)
            self.live.start()
        self.event("Starting; total remaining time unknown until work can be measured")
        return self

    def estimate(self) -> str:
        if self.request_started is None:
            return "remaining: unknown (variable work)"
        samples = self.samples.get(self.request_key, [])
        if not samples:
            return "request ETA: learning (no matching completed calls)"
        expected = sum(samples) / len(samples)
        elapsed = self.clock() - self.request_started
        if elapsed >= expected:
            return "request exceeds measured average; remaining unknown"
        return f"request ~{duration(expected - elapsed)} left (mean of {len(samples)} matching calls)"

    def __rich__(self):
        elapsed = self.clock() - self.started
        frame = "|/-\\"[int(elapsed * 4) % 4]
        return Text(f"{frame} {self.message}\n  elapsed {duration(elapsed)} | {self.estimate()}", style="cyan")

    def event(self, message: str, **details):
        kind = details.get("kind")
        if kind == "request_start":
            self.request_started = self.clock()
            self.request_key = (details.get("schema"), details.get("model"))
        elif kind in {"request_done", "request_error"}:
            if kind == "request_done":
                self.samples.setdefault(self.request_key, []).append(details["seconds"])
                self.calls += 1
            self.request_started = None
        elif kind == "cache_hit":
            self.cache_hits += 1
        # Render all source text literally, including paths and bracketed IDs.
        self.message = message
        stamp = duration(self.clock() - self.started)
        self.console.print(Text(f"  [{stamp}] {message}", style="dim"))
        if kind == "request_start" and not self.console.is_terminal:
            self.console.print(Text(f"    {self.estimate()}", style="dim"))

    def __exit__(self, exc_type, exc, tb):
        _listener.reset(self.token)
        if self.live:
            self.live.stop()
        state = "Interrupted" if exc_type and issubclass(exc_type, (KeyboardInterrupt, EOFError)) else "Failed" if exc_type else "Completed"
        self.console.print(Text(f"{state} in {duration(self.clock() - self.started)} | "
                                f"{self.calls} completed model calls | {self.cache_hits} cache hits",
                                style="red" if exc_type else "green"))
        return False
