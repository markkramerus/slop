"""
gui/utils/state.py — Shared session-state helpers and pipeline status detection.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st


# ── Docket ID ─────────────────────────────────────────────────────────────────

def get_docket_id() -> str:
    return st.session_state.get("docket_id", "")


def set_docket_id(value: str) -> None:
    st.session_state["docket_id"] = value.strip()


def docket_id_widget(label: str = "Docket ID", key: str = "docket_id_input") -> str:
    """
    Render a docket-ID text input that is synced with session state.
    Returns the current (possibly empty) docket ID string.
    """
    current = get_docket_id()
    value = st.text_input(
        label,
        value=current,
        placeholder="e.g. CMS-2025-0050",
        key=key,
    )
    if value != current:
        set_docket_id(value)
    return value.strip()


# ── Pipeline status ────────────────────────────────────────────────────────────

def pipeline_status(docket_id: str) -> dict[str, bool | None]:
    """
    Inspect on-disk outputs to decide which pipeline steps have been completed.

    Returns a dict with keys:
        download    – True if comment_attachments dir exists and is non-empty
        stylometry  – True if {docket_id}/stylometry/index.json exists
        campaign    – True if campaign_plan.json exists (None = not applicable)
        generate    – True if synthetic.txt exists and is non-empty
        shuffle     – True if combined.csv exists
    """
    if not docket_id:
        return {k: None for k in ("download", "stylometry", "campaign", "generate", "shuffle")}

    base = Path(docket_id)

    def _nonempty_dir(p: Path) -> bool:
        return p.is_dir() and any(p.iterdir())

    def _nonempty_file(p: Path) -> bool:
        return p.is_file() and p.stat().st_size > 0

    download = _nonempty_dir(base / "comment_attachments")
    stylometry = _nonempty_file(base / "stylometry" / "index.json")
    campaign_plan = base / "campaign" / "campaign_plan.json"
    campaign = campaign_plan.is_file()
    generate = _nonempty_file(base / "synthetic_comments" / "synthetic.txt")
    shuffle = _nonempty_file(base / "shuffled_comments" / "combined.csv")

    return {
        "download": download,
        "stylometry": stylometry,
        "campaign": campaign,
        "generate": generate,
        "shuffle": shuffle,
    }


def status_badge(ok: bool | None) -> str:
    """Return an emoji badge string for a pipeline status value."""
    if ok is True:
        return "✅"
    if ok is False:
        return "⚠️"
    return "—"


# ── .env file management ───────────────────────────────────────────────────────

ENV_PATH = Path(".env")

_ENV_KEYS = [
    "SLOP_API_KEY",
    "SLOP_EMBED_API_KEY",
    "SLOP_API_BASE_URL",
    "SLOP_EMBED_API_BASE_URL",
    "SLOP_CHAT_MODEL",
    "SLOP_EMBED_MODEL",
]

_SECRET_KEYS = {"SLOP_API_KEY", "SLOP_EMBED_API_KEY"}


def read_env() -> dict[str, str]:
    """Return a dict of the known SLOP env vars from .env (raw values, not masked)."""
    values: dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                if k in _ENV_KEYS:
                    values[k] = v.strip()
    return values


def write_env(values: dict[str, str]) -> None:
    """
    Merge *values* into .env, preserving any existing lines we don't touch.
    Lines for keys in *values* are updated; new keys are appended.
    """
    existing_lines: list[str] = []
    if ENV_PATH.is_file():
        existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in values:
                new_lines.append(f"{k}={values[k]}")
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in values.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def masked(value: str) -> str:
    """Mask all but the last 4 characters of a secret."""
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


# ── Misc helpers ───────────────────────────────────────────────────────────────

def list_voice_skills(docket_id: str) -> list[Path]:
    """Return a sorted list of .md voice skill files in {docket_id}/stylometry/."""
    if not docket_id:
        return []
    return sorted(Path(docket_id, "stylometry").glob("*.md"))


def read_campaign_plan(docket_id: str) -> dict[str, Any] | None:
    """Read and parse campaign_plan.json for the given docket, or return None."""
    path = Path(docket_id, "campaign", "campaign_plan.json")
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def count_synthetic_comments(docket_id: str) -> int:
    """Count ♔-delimited comments in synthetic.txt."""
    path = Path(docket_id, "synthetic_comments", "synthetic.txt")
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    # Each comment record starts with a non-empty line after a double-newline;
    # easier to just count the separator character used in the header row.
    lines = [l for l in text.splitlines() if l.strip()]
    # First line is the header; subsequent groups are records
    return max(0, len(lines) - 1)


def iter_attachment_dirs(docket_id: str):
    """Yield subdirectory Paths inside comment_attachments/."""
    base = Path(docket_id, "comment_attachments")
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir():
                yield child
