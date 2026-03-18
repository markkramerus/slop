"""
gui/pages/6_🔀_Shuffle.py — Translate synthetic comments to PSV format and shuffle with real comments.
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget
from gui.utils.runner import run_command, build_cli_command

st.set_page_config(page_title="Shuffle — SLOP", page_icon="🔀", layout="wide")
st.title("🔀 Step 6 — Pre-process, Translate & Shuffle")
st.caption(
    "**Step 1** — Pre-processes the real comments CSV by substituting attachment text "
    "where it is longer than the inline comment body and converting PSV format. "
    "**Step 2** — Convert the ♔-delimited synthetic comments to PSV format.  "
    "**Step 3** — Randomly interleaves synthetic comments with the pre-processed real "
    "comments and produces `combined.psv` (attachment URLs and tracking numbers cleared) "
    "and a ground-truth `combined_key.csv`."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="shuffle_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite checks ────────────────────────────────────────────────────────
st.subheader("Prerequisites")

synth_input_txt   = Path(docket_id, "synthetic_comments", "synthetic.txt")
real_input_csv        = Path(docket_id, "comments", f"{docket_id}.csv")
attachments_dir = Path(docket_id, "comment_attachments")

pre_cols = st.columns(3)
with pre_cols[0]:
    if synth_input_txt.is_file() and synth_input_txt.stat().st_size > 0:
        size_kb = round(synth_input_txt.stat().st_size / 1024, 1)
        st.success(f"✅ Synthetic output found: `{synth_input_txt}` ({size_kb} KB)")
    else:
        st.error(
            f"❌ Synthetic output not found at `{synth_input_txt}`.  "
            "Run the **Generate** step first."
        )
with pre_cols[1]:
    if real_input_csv.is_file():
        st.success(f"✅ Real comments CSV found: `{real_input_csv}`")
    else:
        st.error(
            f"❌ Real comments CSV not found at `{real_input_csv}`.  "
            "Place the docket CSV there before shuffling."
        )
with pre_cols[2]:
    if attachments_dir.is_dir():
        att_count = sum(1 for p in attachments_dir.iterdir() if p.is_dir())
        st.success(f"✅ Attachments directory found ({att_count:,} comment dirs)")
    else:
        st.warning(
            f"⚠️ Attachments directory not found at `{attachments_dir}`.  "
            "Pre-processing will be skipped — run the **Download** step first "
            "to get attachment text."
        )

st.divider()

# ── Options ────────────────────────────────────────────────────────────────────
st.subheader("Shuffle Options")

opt_cols = st.columns(2)
with opt_cols[0]:
    seed = st.number_input(
        "Random seed",
        value=random.randint(0, 2**32 - 1),
        min_value=0,
        help="Controls the shuffling order. Override for reproducible results.",
    )
    skip_preprocess = st.checkbox(
        "Skip pre-processing step (use raw real CSV directly)",
        value=False,
        help=(
            "--skip-preprocess.  Skip attachment-text substitution and feed the "
            "original comments CSV straight into the shuffle."
        ),
    )
    skip_translation = st.checkbox(
        "Skip translation step (already translated)",
        value=False,
        help=(
            "--skip-translation.  Use if you have already run translation and "
            "`synthetic.psv` exists."
        ),
    )

with opt_cols[1]:
    synth_output_psv = Path(docket_id, "shuffled_comments", "synthetic.psv")
    real_output_psv = Path(docket_id, "shuffled_comments", "real.psv")
    combined_output_psv = Path(docket_id, "shuffled_comments", "combined.psv")
    combined_key_csv = Path(docket_id, "shuffled_comments", "combined_key.csv")
    if skip_translation and synth_output_psv.is_file():
        st.success(f"✅ Translated PSV found: `{synth_output_psv}`")
    elif skip_translation:
        st.warning(f"⚠️ Translated PSV not found at `{synth_output_psv}`")
    if skip_preprocess:
        st.info("ℹ️ Pre-processing skipped — raw real CSV will be used for shuffling.")
    elif real_output_psv.is_file():
        size_kb = round(real_output_psv.stat().st_size / 1024, 1)
        st.success(f"✅ Pre-processed PSV already exists: `{real_output_psv}` ({size_kb} KB)")

with st.expander("Advanced — Explicit path overrides"):
    adv_cols = st.columns(2)
    with adv_cols[0]:
        real_input_csv_override = st.text_input(
            "Real comments input path (.csv)",
            value=str(real_input_csv),
        )
        synth_input_txt_override = st.text_input(
            "Synthetic comments input path (.txt)",
            value=str(synth_input_txt),
        )
        attachments_dir_override = st.text_input(
            "Attachments directory path",
            value=str(attachments_dir),
        )
    with adv_cols[1]:
        real_output_psv_override = st.text_input(
            "Real comment output path (.psv)",
            value=str(real_output_psv),
        )
        synth_output_psv_override = st.text_input(
            "Synthetic comment output path (.psv)",
            value=str(synth_output_psv),
        )
        combined_output_psv_override = st.text_input(
            "Combined output path (.psv)",
            value=str(combined_output_psv),
        )
        combined_key_csv_override = st.text_input(
            "Combined key path (.csv)",
            value=str(combined_key_csv),
        )

st.divider()

# ── Build command ──────────────────────────────────────────────────────────────
def build_shuffle_cmd() -> list[str]:
    args = ["shuffle", "--docket-id", docket_id, "--seed", str(seed)]
    if skip_preprocess:
        args.append("--skip-preprocess")
    if skip_translation:
        args.append("--skip-translation")
    if synth_input_txt_override.strip():
        args += ["--synth-input-txt", synth_input_txt_override.strip()]
    if synth_output_psv_override.strip():
        args += ["--synth-output-psv", synth_output_psv_override.strip()]
    if attachments_dir_override.strip():
        args += ["--attachments-dir", attachments_dir_override.strip()]
    if real_output_psv_override.strip():
        args += ["--real-output-psv", real_output_psv_override.strip()]
    if real_input_csv_override.strip():
        args += ["--real-input-csv", real_input_csv_override.strip()]
    if combined_output_psv_override.strip():
        args += ["--combined-output-psv", combined_output_psv_override.strip()]
    if combined_key_csv_override.strip():
        args += ["--combined-key-csv", combined_key_csv_override.strip()]
    return build_cli_command(args)


with st.expander("🖥️ Preview command", expanded=False):
    st.code(" ".join(build_shuffle_cmd()), language="bash")

# ── Run ────────────────────────────────────────────────────────────────────────
run_disabled = not (synth_input_txt.is_file() or synth_input_txt_override.strip()) or \
               not (real_input_csv.is_file() or real_input_csv_override.strip())

if st.button("🔀 Run Shuffler", type="primary", disabled=run_disabled):
    cmd = build_shuffle_cmd()

    with st.status("Shuffling…", expanded=True) as run_status:
        log = st.empty()
        exit_code, _ = run_command(cmd, log)
        if exit_code == 0:
            run_status.update(label="Shuffle complete ✅", state="complete")
            st.success("Shuffler finished successfully.")
            st.rerun()  # Force page refresh so output metrics update
        else:
            run_status.update(label="Shuffle failed ❌", state="error")
            st.error(f"Shuffler exited with code {exit_code}.")

st.divider()

# ── Output summary & downloads ─────────────────────────────────────────────────
st.subheader("Outputs")

shuffled_dir    = Path(docket_id, "shuffled_comments")
combined_output_psv    = Path(combined_output_psv_override.strip()) if combined_output_psv_override.strip() else shuffled_dir / "combined.psv"
combined_key_csv    = Path(combined_key_csv_override.strip()) if combined_key_csv_override.strip() else shuffled_dir / "combined_key.csv"
synth_output_psv   = Path(synth_output_psv_override.strip()) if synth_output_psv_override.strip() else shuffled_dir / "synthetic.psv"
real_output_psv    = Path(real_output_psv_override.strip()) if real_output_psv_override.strip() else shuffled_dir / "real.psv"

output_files = [
    ("Combined Comments, PSV format (no attachment hints)", combined_output_psv),
    ("Combined Key CSV (real vs. synthetic labels)", combined_key_csv),
    ("Real Comments, PSV format", real_output_psv),
    ("Synthetic Comments, PSV format", synth_output_psv),
]

any_output = False
for label, fpath in output_files:
    if fpath.is_file():
        any_output = True
        size_kb = round(fpath.stat().st_size / 1024, 1)
        col_info, col_dl = st.columns([4, 1])
        with col_info:
            st.write(f"📄 **{label}**: `{fpath}` ({size_kb} KB)")
        with col_dl:
            data = fpath.read_bytes()
            mime = "text/csv" if fpath.suffix.lower() == ".csv" else "text/plain"
            st.download_button(
                label="⬇️ Download",
                data=data,
                file_name=fpath.name,
                mime=mime,
                key=f"dl_{fpath.name}",
            )

if not any_output:
    st.info("No shuffled output found yet.  Run the shuffler above.")
else:
    # Quick stats from key file
    if combined_key_csv.is_file():
        try:
            import pandas as pd
            key_df = pd.read_csv(combined_key_csv)
            if "type" in key_df.columns:
                counts = key_df["type"].value_counts()
                real_count  = counts.get("real", 0)
                synth_count = counts.get("synthetic", 0)
                total       = real_count + synth_count
                stat_cols = st.columns(3)
                stat_cols[0].metric("Total Rows", total)
                stat_cols[1].metric("Real Comments", real_count)
                stat_cols[2].metric("Synthetic Comments", synth_count)
        except Exception:
            pass

# ── Translation only ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Translation Only")
st.caption(
    "Run only the ♔-delimited synthetic comments → PSV translation step, without shuffling.  "
    "Useful for inspection before committing to a shuffle."
)

if st.button("🔄 Translate Only (no shuffle)"):
    from gui.utils.runner import build_script_command
    source = synth_input_txt_override.strip() or str(synth_input_txt)
    dest   = synth_output_psv_override.strip() or str(synth_output_psv)
    cmd = build_script_command(
        "shuffler/translate_to_psv_format.py",
        [source, dest],
    )
    with st.status("Translating…", expanded=True) as ts:
        log3 = st.empty()
        rc, _ = run_command(cmd, log3)
        ts.update(
            label="Translation complete ✅" if rc == 0 else "Translation failed ❌",
            state="complete" if rc == 0 else "error",
        )
