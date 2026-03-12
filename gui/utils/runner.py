"""
gui/utils/runner.py — Subprocess runner that streams output to a Streamlit UI.

Usage
-----
    from gui.utils.runner import run_command

    with st.status("Running...", expanded=True) as status:
        log = st.empty()
        exit_code, output = run_command(["python", "cli.py", "--help"], log)
        if exit_code == 0:
            status.update(label="Done ✅", state="complete")
        else:
            status.update(label="Failed ❌", state="error")
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

import streamlit as st


# Root of the repository — runner.py lives in gui/utils/, so go up two levels.
REPO_ROOT = Path(__file__).parent.parent.parent


def run_command(
    cmd: Sequence[str],
    log_placeholder,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """
    Run *cmd* as a subprocess, streaming stdout+stderr to *log_placeholder*
    (a Streamlit `st.empty()` element) as the process runs.

    Parameters
    ----------
    cmd:
        Command and arguments list, e.g. ``["python", "cli.py", "--help"]``.
    log_placeholder:
        A ``st.empty()`` element.  Updated with accumulated output each line.
    cwd:
        Working directory for the subprocess.  Defaults to the repo root.
    env:
        Extra environment variables merged on top of the current process env.

    Returns
    -------
    (exit_code, full_output)
    """
    import os

    effective_cwd = cwd or REPO_ROOT
    # Always ensure the subprocess uses UTF-8 for stdout/stderr so that
    # Unicode characters (♔, →, —, etc.) in print() calls don't crash
    # with UnicodeEncodeError on Windows when stdout is a pipe.
    effective_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env:
        effective_env.update(env)

    # Use sys.executable so the subprocess uses the same Python / venv
    effective_cmd = [sys.executable if cmd[0] == "python" else cmd[0]] + list(cmd[1:])

    accumulated: list[str] = []

    try:
        proc = subprocess.Popen(
            effective_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(effective_cwd),
            env=effective_env,
        )
    except FileNotFoundError as exc:
        msg = f"[runner] Could not start process: {exc}"
        log_placeholder.code(msg, language="text")
        return 1, msg

    for line in iter(proc.stdout.readline, ""):
        accumulated.append(line.rstrip("\n"))
        # Show the last 200 lines to avoid Streamlit slowdown on very long output
        visible = accumulated[-200:]
        log_placeholder.code("\n".join(visible), language="text")

    proc.wait()
    full_output = "\n".join(accumulated)
    # Final render with full (capped) content
    log_placeholder.code("\n".join(accumulated[-200:]), language="text")
    return proc.returncode, full_output


def build_cli_command(args: list[str]) -> list[str]:
    """
    Helper that prepends 'python cli.py' (resolved to repo root) to *args*.
    """
    cli_path = str(REPO_ROOT / "cli.py")
    return ["python", cli_path] + args


def build_script_command(script_relative: str, args: list[str]) -> list[str]:
    """
    Helper that runs *script_relative* (relative to repo root) with *args*.
    e.g. build_script_command("stylometry/stylometry_analyzer.py", ["CMS-2025-0050"])
    """
    script_path = str(REPO_ROOT / script_relative)
    return ["python", script_path] + args
