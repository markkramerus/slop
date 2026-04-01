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
    "that guide the generator to mimic authentic writing patterns.  "
    "If the preprocessed PSV does not yet exist, it will be generated automatically "
    "at the start of the analysis run."
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
comments_psv = Path(docket_id, "comments", f"{docket_id}.psv")
csv_found = comments_csv.is_file()
psv_found = comments_psv.is_file()

pre_cols = st.columns(2)

with pre_cols[0]:
    if csv_found:
        st.success(f"✅ Comments CSV found: `{comments_csv}`")
    else:
        st.error(
            f"❌ Comments CSV not found at `{comments_csv}`.  \n"
            "Place the docket CSV there before continuing."
        )

with pre_cols[1]:
    if psv_found:
        size_kb = round(comments_psv.stat().st_size / 1024, 1)
        st.success(f"✅ Preprocessed PSV found: `{comments_psv}` ({size_kb} KB)")
    else:
        st.info(
            f"ℹ️ Preprocessed PSV not yet found at `{comments_psv}`.  \n"
            "It will be **generated automatically** when you click "
            "**Analyze Writing Styles** below."
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

analyze_disabled = not csv_found and not explicit_csv.strip()

if st.button("🔬 Analyze Writing Styles", type="primary", disabled=analyze_disabled):
    target = explicit_csv.strip() if explicit_csv.strip() else docket_id

    with st.status("Running stylometry analysis…", expanded=True) as run_status:
        log = st.empty()

        # ── Step 1: Preprocess (generate PSV) if it doesn't exist yet ──────────
        if not psv_found:
            log.code("Step 1/2 — Preprocessing: generating PSV from CSV + attachments…", language="text")
            pre_cmd = ["python", "-m", "shuffler", "preprocess", docket_id]
            pre_exit, _ = run_command(pre_cmd, log)
            if pre_exit != 0:
                run_status.update(label="Preprocessing failed ❌", state="error")
                st.error(
                    f"Preprocessor exited with code {pre_exit}.  "
                    "Check the log above for details."
                )
                st.stop()

        # ── Step 2: Run the stylometry analyzer ────────────────────────────────
        step_label = "Step 2/2" if not psv_found else "Step 1/1"
        log.code(f"{step_label} — Analysing writing styles…", language="text")
        ana_cmd = build_script_command("stylometry/stylometry_analyzer.py", [target])
        ana_exit, _ = run_command(ana_cmd, log)

        if ana_exit == 0:
            run_status.update(label="Analysis complete ✅", state="complete")
            st.success("Stylometry analysis finished successfully.")
        else:
            run_status.update(label="Analysis failed ❌", state="error")
            st.error(f"Stylometry analyzer exited with code {ana_exit}.")

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
