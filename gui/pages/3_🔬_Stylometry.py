"""
gui/pages/3_🔬_Stylometry.py — Analyze writing styles in real docket comments.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget, list_voice_skills
from gui.utils.runner import run_command, build_script_command

st.set_page_config(page_title="Stylometry — SLOP", page_icon="🔬", layout="wide")
st.title("🔬 Step 2 — Stylometry Analysis")
st.caption(
    "Analyses real commenter writing styles and generates **voice skill** `.md` files "
    "that guide the generator to mimic authentic writing patterns."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="stylometry_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite checks ────────────────────────────────────────────────────────
st.subheader("Prerequisites")

comments_csv = Path(docket_id, "comments", f"{docket_id}.csv")
csv_found = comments_csv.is_file()

if csv_found:
    st.success(f"✅ Comments CSV found: `{comments_csv}`")
else:
    st.error(
        f"❌ Comments CSV not found at `{comments_csv}`.  \n"
        "The stylometry analyzer reads from this file.  Place the CSV there before running."
    )

st.divider()

# ── Options ────────────────────────────────────────────────────────────────────
st.subheader("Options")

with st.expander("Advanced — Explicit CSV path override"):
    explicit_csv = st.text_input(
        "CSV path (leave blank to use convention-based default)",
        value="",
        placeholder=str(comments_csv),
    )

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────────
st.subheader("Run Stylometry Analyzer")

if st.button("🔬 Analyze Writing Styles", type="primary", disabled=not csv_found and not explicit_csv.strip()):
    target = explicit_csv.strip() if explicit_csv.strip() else docket_id
    cmd = build_script_command("stylometry/stylometry_analyzer.py", [target])
    st.caption(f"Command: `{' '.join(cmd)}`")

    with st.status("Analysing…", expanded=True) as run_status:
        log = st.empty()
        exit_code, _ = run_command(cmd, log)
        if exit_code == 0:
            run_status.update(label="Analysis complete ✅", state="complete")
            st.success("Stylometry analysis finished successfully.")
        else:
            run_status.update(label="Analysis failed ❌", state="error")
            st.error(f"Stylometry analyzer exited with code {exit_code}.")

st.divider()

# ── Output browser ─────────────────────────────────────────────────────────────
st.subheader("Generated Voice Skills")

skills = list_voice_skills(docket_id)

if not skills:
    st.info(
        f"No voice skill files found yet in `{docket_id}/stylometry/`.  "
        "Run the analyzer above."
    )
else:
    st.write(f"Found **{len(skills)}** voice skill files:")

    # index.json summary
    index_path = Path(docket_id, "stylometry", "index.json")
    if index_path.is_file():
        import json
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            with st.expander("📋 index.json — Voice Skill Index", expanded=False):
                st.json(index_data)
        except Exception:
            pass

    # Individual skill files
    skill_names = [s.stem for s in skills]
    selected_skill = st.selectbox(
        "Preview a voice skill file",
        options=["— select —"] + skill_names,
    )

    if selected_skill and selected_skill != "— select —":
        skill_path = Path(docket_id, "stylometry", f"{selected_skill}.md")
        if skill_path.is_file():
            content = skill_path.read_text(encoding="utf-8", errors="replace")
            with st.expander(f"📄 {selected_skill}.md", expanded=True):
                st.markdown(content)

    # Summary table
    st.write("")
    rows = []
    for s in skills:
        size_kb = round(s.stat().st_size / 1024, 1)
        rows.append({"Skill File": s.name, "Size (KB)": size_kb})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
