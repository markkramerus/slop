"""
gui/pages/5_✍️_Generate.py — Generate synthetic public comments.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget, read_campaign_plan, list_voice_skills
from gui.utils.runner import run_command, build_cli_command

st.set_page_config(page_title="Generate — SLOP", page_icon="✍️", layout="wide")
st.title("✍️ Step 4 — Generate Synthetic Comments")
st.caption(
    "Produces realistic synthetic public comments using LLM-driven persona mimicry, "
    "guided by the docket's voice skills and your campaign plan (or direct-mode settings)."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="generate_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite checks ────────────────────────────────────────────────────────
st.subheader("Prerequisites")

rule_path = Path(docket_id, "rule", "rule.txt")
skills = list_voice_skills(docket_id)
plan = read_campaign_plan(docket_id)

pre_cols = st.columns(3)
with pre_cols[0]:
    if rule_path.is_file():
        st.success(f"✅ Rule text: `{rule_path}`")
    else:
        st.warning(f"⚠️ Rule text not found at `{rule_path}`")
with pre_cols[1]:
    if skills:
        st.success(f"✅ {len(skills)} voice skills")
    else:
        st.warning("⚠️ No voice skills — run Stylometry first")
with pre_cols[2]:
    if plan:
        st.success("✅ Campaign plan detected")
    else:
        st.info("ℹ️ No campaign plan — using Direct mode")

st.divider()

# ── Mode selector ──────────────────────────────────────────────────────────────
st.subheader("Generation Mode")

default_mode = "Campaign Plan" if plan else "Direct"
mode = st.radio(
    "Mode",
    options=["Direct", "Campaign Plan"],
    index=0 if default_mode == "Direct" else 1,
    horizontal=True,
    help=(
        "**Direct**: you specify the objective and attack vector directly.  "
        "**Campaign Plan**: settings come from `campaign_plan.json`, "
        "which distributes comments across argument angles, archetypes, and vectors."
    ),
)

st.divider()

# ── Mode-specific controls ─────────────────────────────────────────────────────

VECTOR_DESCRIPTIONS = {
    1: "Semantic Variance — Same argument, maximally varied surface forms",
    2: "Persona Mimicry — Diverse stakeholders, same underlying position",
    3: "Citation Flooding — Arguments loaded with plausible-sounding references",
    4: "Dilution / Noise — High-volume, low-substance vague agreement",
}

if mode == "Direct":
    st.subheader("Direct Mode Settings")
    col_obj, col_vec = st.columns([3, 2])

    with col_obj:
        objective = st.text_area(
            "Objective *",
            height=100,
            placeholder='e.g. "oppose the proposed reduction of Medicare Advantage quality bonus payments"',
            help="The position to advance or oppose.  Required in Direct mode.",
        )

    with col_vec:
        vector = st.radio(
            "Attack Vector *",
            options=list(VECTOR_DESCRIPTIONS.keys()),
            format_func=lambda v: f"{v} — {VECTOR_DESCRIPTIONS[v].split(' — ')[0]}",
            help="Controls how the synthetic comments vary from each other.",
        )
        st.caption(VECTOR_DESCRIPTIONS[vector])

else:
    st.subheader("Campaign Plan Mode")
    if not plan:
        st.error(
            "No campaign plan found for this docket.  "
            "Generate one on the **Campaign Planner** page, or switch to Direct mode."
        )
        st.stop()

    objective_from_plan = plan.get("objective", "")
    st.info(f"**Objective from plan:** {objective_from_plan}")

    angles = plan.get("argument_angles", [])
    if angles:
        with st.expander(f"📐 {len(angles)} Argument Angles in Plan", expanded=False):
            for i, angle in enumerate(angles, 1):
                name = angle.get("name", str(angle))
                desc = angle.get("description", "")
                st.markdown(f"**{i}. {name}**" + (f" — {desc}" if desc else ""))

    with st.expander("⚡ Optional Vector Override", expanded=False):
        st.caption("Leave as 'Plan default' to use the plan's vector weight distribution.")
        vector_override = st.radio(
            "Force a specific vector",
            options=["Plan default"] + list(VECTOR_DESCRIPTIONS.keys()),
            format_func=lambda v: "Plan default" if v == "Plan default" else f"{v} — {VECTOR_DESCRIPTIONS[v].split(' — ')[0]}",
            horizontal=True,
        )

st.divider()

# ── Common settings ────────────────────────────────────────────────────────────
st.subheader("Common Settings")

common_cols = st.columns([1, 1, 2])
with common_cols[0]:
    volume = st.number_input(
        "Volume (# comments) *",
        min_value=1,
        max_value=10000,
        value=100,
        step=1,
        help="Number of accepted synthetic comments to produce.",
    )
with common_cols[1]:
    output_override = st.text_input(
        "Output path override",
        value="",
        placeholder=f"{docket_id}/synthetic_comments/synthetic.txt",
        help="Leave blank to use the convention-based default.",
    )

# ── Advanced / QC options ──────────────────────────────────────────────────────
with st.expander("⚙️ Advanced & Quality Control Options"):
    adv_cols = st.columns(3)

    with adv_cols[0]:
        st.markdown("**QC Checks**")
        no_relevance = st.checkbox("Skip relevance check", value=False,
                                   help="--no-relevance-check")
        no_argument = st.checkbox("Skip argument-presence check", value=False,
                                  help="--no-argument-check")
        no_embedding = st.checkbox("Skip embedding dedup check", value=False,
                                   help="--no-embedding-check")
        include_failed = st.checkbox("Include failed-QC rows in output", value=False,
                                     help="--include-failed-qc")

    with adv_cols[1]:
        st.markdown("**Generation**")
        seed = st.number_input("Random seed", value=42, min_value=0, help="--seed")
        comment_period = st.number_input(
            "Comment period (days)", value=60, min_value=1, help="--comment-period-days"
        )
        max_concurrent = st.number_input(
            "Max concurrent requests", value=10, min_value=1, max_value=50,
            help="--max-concurrent  (async mode)"
        )
        no_async = st.checkbox("Disable async (slower)", value=False,
                               help="--no-async")

    with adv_cols[2]:
        st.markdown("**Quality Thresholds**")
        sim_threshold = st.slider(
            "Similarity threshold (dedup)",
            min_value=0.70, max_value=1.00, value=0.92, step=0.01,
            help="--similarity-threshold"
        )
        max_retries = st.number_input(
            "Max retries per slot", value=3, min_value=1, max_value=20,
            help="--max-retries"
        )

        st.markdown("**API Overrides**")
        api_key_override = st.text_input("API key override", value="", type="password")
        chat_model_override = st.text_input("Chat model override", value="")

st.divider()

# ── Build & show command ───────────────────────────────────────────────────────
def build_generate_cmd() -> list[str]:
    args = ["--docket-id", docket_id, "--volume", str(volume)]

    if mode == "Direct":
        args += ["--objective", objective, "--vector", str(vector)]
    else:
        # Campaign plan mode — auto-detected by cli.py from the docket directory
        if vector_override != "Plan default":
            args += ["--vector", str(vector_override)]

    if output_override.strip():
        args += ["--output", output_override.strip()]

    # QC
    if no_relevance:
        args.append("--no-relevance-check")
    if no_argument:
        args.append("--no-argument-check")
    if no_embedding:
        args.append("--no-embedding-check")
    if include_failed:
        args.append("--include-failed-qc")

    # Generation
    args += ["--seed", str(seed)]
    args += ["--comment-period-days", str(comment_period)]
    args += ["--max-concurrent", str(max_concurrent)]
    if no_async:
        args.append("--no-async")

    # QC thresholds
    args += ["--similarity-threshold", str(sim_threshold)]
    args += ["--max-retries", str(max_retries)]

    # API overrides
    if api_key_override.strip():
        args += ["--api-key", api_key_override.strip()]
    if chat_model_override.strip():
        args += ["--chat-model", chat_model_override.strip()]

    return build_cli_command(args)


# Validate before running
can_run = bool(docket_id)
if mode == "Direct":
    can_run = can_run and bool(objective.strip() if "objective" in dir() else False)

# Show the command that will be run
try:
    cmd_preview = build_generate_cmd()
    with st.expander("🖥️ Preview command", expanded=False):
        st.code(" ".join(cmd_preview), language="bash")
except Exception:
    pass

# ── Run ────────────────────────────────────────────────────────────────────────
run_disabled = (mode == "Direct" and not (objective.strip() if "objective" in dir() else False))

if st.button("✍️ Generate Comments", type="primary", disabled=run_disabled):
    cmd = build_generate_cmd()

    with st.status("Generating synthetic comments…", expanded=True) as run_status:
        log = st.empty()
        exit_code, output = run_command(cmd, log)
        if exit_code == 0:
            run_status.update(label="Generation complete ✅", state="complete")
            st.success("Comments generated successfully.")
        else:
            run_status.update(label="Generation failed ❌", state="error")
            st.error(f"Generator exited with code {exit_code}.")

st.divider()

# ── Preview output ─────────────────────────────────────────────────────────────
st.subheader("Output Preview")

output_path = Path(
    output_override.strip() if output_override.strip()
    else Path(docket_id, "synthetic_comments", "synthetic.txt")
)

if output_path.is_file():
    raw = output_path.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l.strip()]
    total = max(0, len(lines) - 1)  # subtract header row
    st.write(f"**{total}** comment records in `{output_path}` ({round(output_path.stat().st_size / 1024, 1)} KB)")

    if total > 0:
        # Parse ♔-delimited records
        header_line = lines[0] if lines else ""
        headers = header_line.split("♔")
        records = []
        for line in lines[1:]:
            parts = line.split("♔")
            records.append(dict(zip(headers, parts + [""] * max(0, len(headers) - len(parts)))))

        # Show first 5 as cards
        preview_count = min(5, len(records))
        st.write(f"Showing first {preview_count} of {len(records)} comments:")
        for rec in records[:preview_count]:
            archetype = rec.get("synth_archetype", "")
            sophis = rec.get("synth_sophistication", "")
            state = rec.get("synth_persona_state", "")
            occ = rec.get("synth_persona_occupation", "")
            comment_text = rec.get("Comment", rec.get("comment", ""))
            qc = rec.get("synth_qc_passed", "")
            qc_badge = "✅" if qc == "TRUE" else "⚠️"

            label = f"{qc_badge} {archetype} / {sophis} — {occ}, {state}"
            with st.expander(label, expanded=False):
                st.write(comment_text[:2000] + ("…" if len(comment_text) > 2000 else ""))
else:
    st.info(f"No output file found at `{output_path}`.  Generate comments above.")
