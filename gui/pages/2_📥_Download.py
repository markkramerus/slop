"""
gui/pages/2_📥_Download.py — Download attachments from regulations.gov.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget, get_docket_id, iter_attachment_dirs
from gui.utils.runner import run_command, build_script_command

st.set_page_config(page_title="Download — SLOP", page_icon="📥", layout="wide")
st.title("📥 Step 1 — Download Attachments")
st.caption(
    "Downloads attachment files from regulations.gov and organises them under "
    "`{docket_id}/comment_attachments/`."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="download_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite check ─────────────────────────────────────────────────────────
st.subheader("Prerequisites")

comments_csv = Path(docket_id, "comments", f"{docket_id}.csv")
comments_xlsx = Path(docket_id, "comments", f"{docket_id}.xlsx")

csv_found = comments_csv.is_file()
xlsx_found = comments_xlsx.is_file()

if csv_found:
    st.success(f"✅ Comments CSV found: `{comments_csv}`")
elif xlsx_found:
    st.info(
        f"ℹ️ XLSX found (`{comments_xlsx}`), but the downloader expects a CSV.  "
        "Export the XLSX to CSV and place it at the path shown above."
    )
else:
    st.warning(
        f"⚠️ No comments file found at `{comments_csv}`.  \n"
        "Download the docket's CSV export from "
        "[regulations.gov](https://www.regulations.gov) and place it there before running."
    )

st.divider()

# ── Options ────────────────────────────────────────────────────────────────────
st.subheader("Options")

convert_text = st.checkbox(
    "Convert PDF/DOCX attachments to `.txt` for downstream processing",
    value=True,
    help="Adds `--convert-text` flag.  Requires pypdf and python-docx (already in requirements.txt).",
)

with st.expander("Advanced — Explicit CSV path override"):
    explicit_csv = st.text_input(
        "CSV path (leave blank to use convention-based default)",
        value="",
        placeholder=str(comments_csv),
    )

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────────
st.subheader("Run Downloader")

if st.button("📥 Download Attachments", type="primary", disabled=not csv_found and not explicit_csv):
    target = explicit_csv.strip() if explicit_csv.strip() else docket_id
    cmd = build_script_command(
        "downloader/download_attachments.py",
        [target] + (["--convert-text"] if convert_text else []),
    )
    st.caption(f"Command: `{' '.join(cmd)}`")

    with st.status("Running downloader…", expanded=True) as run_status:
        log = st.empty()
        exit_code, output = run_command(cmd, log)
        if exit_code == 0:
            run_status.update(label="Download complete ✅", state="complete")
            st.success("Downloader finished successfully.")
        else:
            run_status.update(label="Downloader failed ❌", state="error")
            st.error(f"Downloader exited with code {exit_code}.")

st.divider()

# ── Output browser ─────────────────────────────────────────────────────────────
st.subheader("Downloaded Files")

att_dirs = list(iter_attachment_dirs(docket_id))
if not att_dirs:
    st.info("No attachment directories found yet.  Run the downloader above.")
else:
    st.write(f"Found **{len(att_dirs)}** comment attachment directories:")
    rows = []
    for d in att_dirs:
        files = list(d.iterdir())
        txt_count = sum(1 for f in files if f.suffix == ".txt")
        other_count = len(files) - txt_count
        rows.append({
            "Directory": d.name,
            "Total Files": len(files),
            "Text Files": txt_count,
            "Other Files": other_count,
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Text converter standalone
    st.divider()
    st.subheader("Convert Already-Downloaded Files")
    st.caption(
        "If you skipped `--convert-text` earlier, run the converter now on "
        "already-downloaded files."
    )
    if st.button("🔄 Run Text Converter Only"):
        cmd = build_script_command("downloader/text_converter.py", [docket_id])
        with st.status("Converting…", expanded=True) as conv_status:
            log2 = st.empty()
            rc, _ = run_command(cmd, log2)
            if rc == 0:
                conv_status.update(label="Conversion complete ✅", state="complete")
            else:
                conv_status.update(label="Conversion failed ❌", state="error")
