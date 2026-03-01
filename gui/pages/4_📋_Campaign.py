"""
gui/pages/4_📋_Campaign.py — Campaign Planner: write a scenario brief and generate/edit the campaign plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget, read_campaign_plan, list_voice_skills
from gui.utils.runner import run_command, build_script_command

st.set_page_config(page_title="Campaign Planner — SLOP", page_icon="📋", layout="wide")
st.title("📋 Step 3 — Campaign Planner")
st.caption(
    "*(Optional but recommended)*  "
    "Translates a natural-language scenario brief into a structured JSON campaign plan "
    "that controls argument distribution, stakeholder mix, and attack vectors."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="campaign_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite check ─────────────────────────────────────────────────────────
st.subheader("Prerequisites")

rule_path = Path(docket_id, "rule", "rule.txt")
skills = list_voice_skills(docket_id)

col_pre1, col_pre2 = st.columns(2)
with col_pre1:
    if rule_path.is_file():
        st.success(f"✅ Rule text found: `{rule_path}`")
    else:
        st.warning(
            f"⚠️ Rule text not found at `{rule_path}`.  \n"
            "The planner reads this file for context.  "
            "Place the proposed rule text there (plain `.txt`)."
        )
with col_pre2:
    if skills:
        st.success(f"✅ {len(skills)} voice skills found in `{docket_id}/stylometry/`")
    else:
        st.info("ℹ️ No voice skills yet — run Stylometry (step 2) first for best results.")

st.divider()

# ── Scenario Brief ─────────────────────────────────────────────────────────────
st.subheader("1. Write a Scenario Brief")
st.caption(
    "Describe in plain language what position the campaign should advance or oppose, "
    "who the stakeholders are, and any key arguments.  "
    "The LLM planner will convert this into a structured JSON campaign plan."
)

brief_path = Path(docket_id, "campaign", "scenario_brief.txt")
existing_brief = ""
if brief_path.is_file():
    existing_brief = brief_path.read_text(encoding="utf-8", errors="replace")

brief_text = st.text_area(
    "Scenario Brief",
    value=existing_brief,
    height=200,
    placeholder=(
        "Example: We want to oppose the proposed reduction of Medicare Advantage quality "
        "bonus payments. Stakeholders include small clinic operators, patients with "
        "chronic conditions, and healthcare IT vendors. Key arguments: reduced bonuses "
        "will limit investment in digital health tools, hurt rural access, and undermine "
        "interoperability."
    ),
)

if st.button("💾 Save Brief"):
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief_text, encoding="utf-8")
    st.success(f"Brief saved to `{brief_path}`")

st.divider()

# ── Generate Campaign Plan ─────────────────────────────────────────────────────
st.subheader("2. Generate Campaign Plan")

with st.expander("Advanced options"):
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        rule_override = st.text_input(
            "Rule text path override",
            value="",
            placeholder=str(rule_path),
        )
        scenario_override = st.text_input(
            "Scenario brief path override",
            value="",
            placeholder=str(brief_path),
        )
    with col_a2:
        output_override = st.text_input(
            "Campaign plan output path override",
            value="",
            placeholder=str(Path(docket_id, "campaign", "campaign_plan.json")),
        )

if st.button("📋 Generate Campaign Plan", type="primary"):
    if not brief_path.is_file() and not scenario_override.strip():
        st.error("Save a scenario brief first (or provide an override path).")
    else:
        cmd_args = ["--docket-id", docket_id]
        if rule_override.strip():
            cmd_args += ["--rule-text", rule_override.strip()]
        if scenario_override.strip():
            cmd_args += ["--scenario", scenario_override.strip()]
        if output_override.strip():
            cmd_args += ["--output", output_override.strip()]

        cmd = build_script_command("campaign/planner.py", cmd_args)
        st.caption(f"Command: `{' '.join(cmd)}`")

        with st.status("Generating campaign plan…", expanded=True) as run_status:
            log = st.empty()
            exit_code, _ = run_command(cmd, log)
            if exit_code == 0:
                run_status.update(label="Plan generated ✅", state="complete")
                st.success("Campaign plan created successfully.")
            else:
                run_status.update(label="Plan generation failed ❌", state="error")
                st.error(f"Planner exited with code {exit_code}.")

st.divider()

# ── Edit Campaign Plan ─────────────────────────────────────────────────────────
st.subheader("3. Review & Edit Campaign Plan")

plan_path = Path(docket_id, "campaign", "campaign_plan.json")

if not plan_path.is_file():
    st.info("No campaign plan found.  Generate one above (or skip this step for direct mode generation).")
else:
    plan_raw = plan_path.read_text(encoding="utf-8")

    # Pretty-printed for display and editing
    try:
        plan_obj = json.loads(plan_raw)
        pretty = json.dumps(plan_obj, indent=2)
    except json.JSONDecodeError:
        plan_obj = None
        pretty = plan_raw

    # Structured summary at a glance
    if plan_obj:
        st.markdown("**Plan summary**")
        summary_cols = st.columns(3)
        summary_cols[0].metric(
            "Objective",
            (plan_obj.get("objective", "")[:60] + "…") if len(plan_obj.get("objective", "")) > 60 else plan_obj.get("objective", "—"),
        )
        angles = plan_obj.get("argument_angles", [])
        summary_cols[1].metric("Argument Angles", len(angles))
        vectors = plan_obj.get("vector_weights", {})
        summary_cols[2].metric("Vectors defined", len(vectors))

        if angles:
            with st.expander("📐 Argument Angles", expanded=False):
                for i, angle in enumerate(angles, 1):
                    st.markdown(f"**{i}.** {angle.get('name', angle)}")

        if vectors:
            with st.expander("⚡ Vector Weights", expanded=False):
                st.json(vectors)

    edited_json = st.text_area(
        "Edit campaign_plan.json (raw JSON)",
        value=pretty,
        height=400,
    )

    col_save, col_validate = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save Campaign Plan"):
            try:
                validated = json.loads(edited_json)
                plan_path.write_text(
                    json.dumps(validated, indent=2),
                    encoding="utf-8",
                )
                st.success(f"Campaign plan saved to `{plan_path}`")
            except json.JSONDecodeError as exc:
                st.error(f"❌ Invalid JSON: {exc}")
    with col_validate:
        if st.button("✅ Validate JSON"):
            try:
                json.loads(edited_json)
                st.success("JSON is valid.")
            except json.JSONDecodeError as exc:
                st.error(f"❌ JSON error: {exc}")
