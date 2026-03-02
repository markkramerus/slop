"""
gui/pages/6_🔀_Shuffle.py — Translate synthetic comments to CMS format and shuffle with real comments.
"""
from __future__ import annotations

import sys
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
    "**Step 0** — Pre-processes the real comments CSV by substituting attachment text "
    "where it is longer than the inline comment body.  "
    "**Step 1** — Converts the ♔-delimited synthetic output to CMS CSV format.  "
    "**Step 2** — Randomly interleaves synthetic comments with the pre-processed real "
    "comments and produces `combined.csv` (attachment URLs cleared) and a ground-truth "
    "`combined_key.csv`."
)
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="shuffle_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Prerequisite checks ────────────────────────────────────────────────────────
st.subheader("Prerequisites")

synthetic_txt   = Path(docket_id, "synthetic_comments", "synthetic.txt")
real_csv        = Path(docket_id, "comments", f"{docket_id}.csv")
attachments_dir = Path(docket_id, "comment_attachments")

pre_cols = st.columns(3)
with pre_cols[0]:
    if synthetic_txt.is_file() and synthetic_txt.stat().st_size > 0:
        size_kb = round(synthetic_txt.stat().st_size / 1024, 1)
        st.success(f"✅ Synthetic output found: `{synthetic_txt}` ({size_kb} KB)")
    else:
        st.error(
            f"❌ Synthetic output not found at `{synthetic_txt}`.  "
            "Run the **Generate** step first."
        )
with pre_cols[1]:
    if real_csv.is_file():
        st.success(f"✅ Real comments CSV found: `{real_csv}`")
    else:
        st.error(
            f"❌ Real comments CSV not found at `{real_csv}`.  "
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
        value=42,
        min_value=0,
        help="Controls the shuffling order.  Use the same seed to reproduce results.",
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
            "`synthetic_cms.csv` exists."
        ),
    )

with opt_cols[1]:
    translated_path = Path(docket_id, "shuffled_comments", "synthetic_cms.csv")
    preprocessed_path = Path(docket_id, "shuffled_comments", "preprocessed_real.csv")
    if skip_translation and translated_path.is_file():
        st.success(f"✅ Translated CSV found: `{translated_path}`")
    elif skip_translation:
        st.warning(f"⚠️ Translated CSV not found at `{translated_path}`")
    if skip_preprocess:
        st.info("ℹ️ Pre-processing skipped — raw real CSV will be used for shuffling.")
    elif preprocessed_path.is_file():
        size_kb = round(preprocessed_path.stat().st_size / 1024, 1)
        st.success(f"✅ Pre-processed CSV already exists: `{preprocessed_path}` ({size_kb} KB)")

with st.expander("Advanced — Explicit path overrides"):
    adv_cols = st.columns(2)
    with adv_cols[0]:
        syncom_override = st.text_input(
            "Syncom output path",
            value="",
            placeholder=str(synthetic_txt),
        )
        translated_override = st.text_input(
            "Translated CSV output path",
            value="",
            placeholder=str(translated_path),
        )
        attachments_override = st.text_input(
            "Attachments directory path",
            value="",
            placeholder=str(attachments_dir),
        )
    with adv_cols[1]:
        real_override = st.text_input(
            "Real comments CSV path",
            value="",
            placeholder=str(real_csv),
        )
        combined_override = st.text_input(
            "Combined output path",
            value="",
            placeholder=str(Path(docket_id, "shuffled_comments", "combined.csv")),
        )
        preprocessed_override = st.text_input(
            "Pre-processed output path",
            value="",
            placeholder=str(preprocessed_path),
        )

st.divider()

# ── Build command ──────────────────────────────────────────────────────────────
def build_shuffle_cmd() -> list[str]:
    args = ["shuffle", "--docket-id", docket_id, "--seed", str(seed)]
    if skip_preprocess:
        args.append("--skip-preprocess")
    if skip_translation:
        args.append("--skip-translation")
    if syncom_override.strip():
        args += ["--syncom-output", syncom_override.strip()]
    if translated_override.strip():
        args += ["--translated-output", translated_override.strip()]
    if attachments_override.strip():
        args += ["--attachments-dir", attachments_override.strip()]
    if preprocessed_override.strip():
        args += ["--preprocessed-output", preprocessed_override.strip()]
    if real_override.strip():
        args += ["--real-comments", real_override.strip()]
    if combined_override.strip():
        args += ["--combined-output", combined_override.strip()]
    return build_cli_command(args)


with st.expander("🖥️ Preview command", expanded=False):
    st.code(" ".join(build_shuffle_cmd()), language="bash")

# ── Run ────────────────────────────────────────────────────────────────────────
run_disabled = not (synthetic_txt.is_file() or syncom_override.strip()) or \
               not (real_csv.is_file() or real_override.strip())

if st.button("🔀 Run Shuffler", type="primary", disabled=run_disabled):
    cmd = build_shuffle_cmd()

    with st.status("Shuffling…", expanded=True) as run_status:
        log = st.empty()
        exit_code, _ = run_command(cmd, log)
        if exit_code == 0:
            run_status.update(label="Shuffle complete ✅", state="complete")
            st.success("Shuffler finished successfully.")
        else:
            run_status.update(label="Shuffle failed ❌", state="error")
            st.error(f"Shuffler exited with code {exit_code}.")

st.divider()

# ── Output summary & downloads ─────────────────────────────────────────────────
st.subheader("Outputs")

shuffled_dir    = Path(docket_id, "shuffled_comments")
combined_csv    = Path(combined_override.strip()) if combined_override.strip() else shuffled_dir / "combined.csv"
combined_key    = combined_csv.with_name(combined_csv.stem + "_key.csv")
synthetic_cms   = Path(translated_override.strip()) if translated_override.strip() else shuffled_dir / "synthetic_cms.csv"
preprocessed    = Path(preprocessed_override.strip()) if preprocessed_override.strip() else shuffled_dir / "preprocessed_real.csv"

output_files = [
    ("Combined CSV (no attachment hints)", combined_csv),
    ("Key CSV (real vs. synthetic labels)", combined_key),
    ("Pre-processed Real CSV (attachment text merged)", preprocessed),
    ("Translated Synthetic CSV", synthetic_cms),
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
            st.download_button(
                label="⬇️ Download",
                data=data,
                file_name=fpath.name,
                mime="text/csv",
                key=f"dl_{fpath.name}",
            )

if not any_output:
    st.info("No shuffled output found yet.  Run the shuffler above.")
else:
    # Quick stats from key file
    if combined_key.is_file():
        try:
            import pandas as pd
            key_df = pd.read_csv(combined_key)
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
    "Run only the ♔-delimited → CMS CSV translation step, without shuffling.  "
    "Useful for inspection before committing to a shuffle."
)

if st.button("🔄 Translate Only (no shuffle)"):
    from gui.utils.runner import build_script_command
    source = syncom_override.strip() or str(synthetic_txt)
    dest   = translated_override.strip() or str(synthetic_cms)
    cmd = build_script_command(
        "shuffler/translate_to_cms_format.py",
        [source, dest],
    )
    with st.status("Translating…", expanded=True) as ts:
        log3 = st.empty()
        rc, _ = run_command(cmd, log3)
        ts.update(
            label="Translation complete ✅" if rc == 0 else "Translation failed ❌",
            state="complete" if rc == 0 else "error",
        )
