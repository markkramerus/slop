"""
gui/app.py — SLOP Web Interface — Pipeline Dashboard

Launch with:
    streamlit run gui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable from any working directory
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import (
    docket_id_widget,
    get_docket_id,
    pipeline_status,
    status_badge,
    count_synthetic_comments,
    list_voice_skills,
    read_campaign_plan,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SLOP — Synthetic Comment Platform",
    page_icon="🤮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤮 SLOP — Synthetic Letter-writing Opposition Platform")
st.caption("Generate realistic synthetic public comments for regulatory docket research.")
st.divider()

# ── Docket ID selector ─────────────────────────────────────────────────────────
col_id, col_refresh = st.columns([4, 1])
with col_id:
    docket_id = docket_id_widget("Active Docket ID", key="dashboard_docket_id")
with col_refresh:
    st.write("")
    st.write("")
    refresh = st.button("🔄 Refresh Status")

# ── Pipeline status overview ───────────────────────────────────────────────────
st.subheader("Pipeline Status")

if not docket_id:
    st.info("Enter a Docket ID above to see pipeline status.")
else:
    status = pipeline_status(docket_id)

    STEPS = [
        ("📥 Download Attachments","download",   "2_📥_Download Attachments", "Download attachments from regulations.gov"),
        ("🔬 Stylometry",       "stylometry", "3_🔬_Stylometry",  "Analyze writing styles in real comments"),
        ("📋 Campaign Planner", "campaign",   "4_📋_Campaign",    "Generate a structured campaign plan (optional)"),
        ("✍️ Generate",          "generate",   "5_✍️_Generate",    "Generate synthetic comments"),
        ("🔀 Shuffle",          "shuffle",    "6_🔀_Shuffle",     "Interleave synthetic comments with real ones"),
    ]

    cols = st.columns(len(STEPS))
    for col, (label, key, page, desc) in zip(cols, STEPS):
        ok = status.get(key)
        badge = status_badge(ok)
        with col:
            st.markdown(
                f"""
<div style="
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px 12px;
    text-align: center;
    background: {'#f0fff4' if ok else '#fff8f0' if ok is False else '#f9f9f9'};
">
<div style="font-size:1.6rem">{badge}</div>
<div style="font-weight:600; margin:6px 0 4px">{label}</div>
<div style="font-size:0.78rem; color:#555">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")

    # ── Summary stats ──────────────────────────────────────────────────────────
    st.subheader("Docket Summary")

    metric_cols = st.columns(5)

    # Attachment dirs
    att_dir = Path(docket_id, "comment_attachments")
    att_count = sum(1 for p in att_dir.iterdir() if p.is_dir()) if att_dir.is_dir() else 0
    metric_cols[0].metric("Attachment Dirs", att_count)

    # Voice skills
    skills = list_voice_skills(docket_id)
    metric_cols[1].metric("Voice Skills", len(skills))

    # Campaign plan
    plan = read_campaign_plan(docket_id)
    plan_label = "Yes" if plan else "No"
    metric_cols[2].metric("Campaign Plan", plan_label)

    # Synthetic comments
    synth_count = count_synthetic_comments(docket_id)
    metric_cols[3].metric("Synthetic Comments", synth_count)

    # Shuffled / combined
    combined_path = Path(docket_id, "shuffled_comments", "combined.csv")
    combined_label = "Ready" if combined_path.is_file() else "Not yet"
    metric_cols[4].metric("Combined CSV", combined_label)

    st.divider()

    # ── Quick-start instructions ───────────────────────────────────────────────
    incomplete = [label for label, key, _, _ in STEPS if not status.get(key)]
    if not incomplete:
        st.success("🎉 All pipeline steps are complete for this docket!")
    else:
        next_step_label = incomplete[0]
        st.info(
            f"**Next step:** {next_step_label}  \n"
            "Use the sidebar to navigate to each step in order."
        )

# ── Sidebar nav hint ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown(
        """
Use the pages listed above to work through the pipeline in order:

1. ⚙️ **Configuration** — Set API keys
2. 📥 **Download Attachments** — Fetch attachments
3. 🔬 **Stylometry** — Analyze writing styles
4. 📋 **Campaign Planner** — Plan comment strategy
5. ✍️ **Generate** — Create synthetic comments
6. 🔀 **Shuffle** — Mix into real comment file
7. 📄 **Results** — Explore outputs
"""
    )
    st.divider()
    st.caption("SLOP — for research purposes only.")
