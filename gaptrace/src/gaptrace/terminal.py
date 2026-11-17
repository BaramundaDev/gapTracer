"""Persistent terminal session over the public CLI commands."""
from __future__ import annotations

import shlex
from pathlib import Path

import click
import typer
from typer.main import get_command


def split_command(line: str) -> list[str]:
    # Retain Windows path separators while supporting quoted arguments.
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""
    return list(lexer)


def run_shell(app: typer.Typer, data: str) -> None:
    command = get_command(app)
    active = str(Path(data).resolve())
    from .terminal_ui import banner
    banner(active)
    if not (Path(active) / "bundle").is_dir():
        typer.echo("No corpus loaded. Run research to choose a topic and collect papers online, or demo for a sample.")
    while True:
        try:
            line = input(f"gaptrace [{Path(active).name}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nSession closed.")
            return
        if not line:
            continue
        try:
            args = split_command(line)
            if not args:
                continue
            name = args[0]
            if name in {"exit", "quit"}:
                return
            if name == "help":
                typer.echo("research     Choose/type a topic, collect papers, and investigate gaps. Switches to the new corpus.")
                typer.echo("use PATH     Switch the active corpus (quote paths with spaces).")
                typer.echo("exit         Close this session. Each ask starts a fresh evidence query.")
                typer.echo(command.get_help(click.Context(command)))
                continue
            if name == "use":
                if len(args) != 2:
                    raise ValueError('Usage: use "path/to/corpus"')
                candidate = Path(args[1]).resolve()
                if not (candidate / "bundle").is_dir():
                    raise ValueError(f"No corpus bundle at {candidate}")
                active = str(candidate)
                typer.echo(f"Corpus: {active}")
                continue
            if name == "shell":
                raise ValueError("Already in a terminal session.")
            if name != "research" and not any(arg == "--data" or arg.startswith("--data=") for arg in args):
                args += ["--data", active]
            result = command.main(args=args, prog_name="gaptrace", standalone_mode=False)
            if name == "research" and result and (Path(result) / "bundle").is_dir():
                active = str(Path(result).resolve())
                typer.echo(f"Corpus: {active}")
        except click.ClickException as exc:
            exc.show()
        except (click.Abort, KeyboardInterrupt):
            typer.echo("Command interrupted; session remains open.")
        except click.exceptions.Exit:
            pass
        except Exception as exc:
            typer.echo(f"Command failed: {exc}", err=True)
