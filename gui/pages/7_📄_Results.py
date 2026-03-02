"""
gui/pages/7_📄_Results.py — Explore and download all outputs for the active docket.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import docket_id_widget, list_voice_skills

st.set_page_config(page_title="Results — SLOP", page_icon="📄", layout="wide")
st.title("📄 Results — Explore Outputs")
st.caption("Browse and download all generated files for the active docket.")
st.divider()

# ── Docket ID ──────────────────────────────────────────────────────────────────
docket_id = docket_id_widget(key="results_docket_id")

if not docket_id:
    st.warning("Enter a Docket ID to continue.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_comments, tab_combined, tab_key, tab_skills = st.tabs(
    ["✍️ Synthetic Comments", "📊 Combined CSV", "🔑 Key File", "🎨 Voice Skills"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Synthetic Comments
# ─────────────────────────────────────────────────────────────────────────────
with tab_comments:
    st.subheader("Synthetic Comments")

    synth_path = Path(docket_id, "synthetic_comments", "synthetic.txt")
    if not synth_path.is_file():
        st.info(f"No synthetic comment file found at `{synth_path}`.  Run the Generate step first.")
    else:
        raw = synth_path.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        total_records = max(0, len(lines) - 1)
        size_kb = round(synth_path.stat().st_size / 1024, 1)

        st.write(f"**{total_records}** records · {size_kb} KB · `{synth_path}`")

        # Download button
        st.download_button(
            "⬇️ Download synthetic.txt",
            data=synth_path.read_bytes(),
            file_name="synthetic.txt",
            mime="text/plain",
        )

        if total_records == 0:
            st.warning("File appears to contain no comment records.")
        else:
            # Parse ♔-delimited
            header_line = lines[0]
            headers = header_line.split("♔")
            records = []
            for line in lines[1:]:
                parts = line.split("♔")
                records.append(
                    dict(zip(headers, parts + [""] * max(0, len(headers) - len(parts))))
                )

            # ── Filter controls ────────────────────────────────────────────
            st.divider()
            filter_cols = st.columns(4)

            archetypes = sorted({r.get("synth_archetype", "") for r in records if r.get("synth_archetype")})
            sophistications = sorted({r.get("synth_sophistication", "") for r in records if r.get("synth_sophistication")})
            qc_statuses = sorted({r.get("synth_qc_passed", "") for r in records if r.get("synth_qc_passed")})

            with filter_cols[0]:
                selected_archetypes = st.multiselect(
                    "Archetype", options=archetypes, default=archetypes
                )
            with filter_cols[1]:
                selected_sophis = st.multiselect(
                    "Sophistication", options=sophistications, default=sophistications
                )
            with filter_cols[2]:
                selected_qc = st.multiselect(
                    "QC Status", options=qc_statuses, default=qc_statuses
                )
            with filter_cols[3]:
                search_text = st.text_input("Search comment text", value="")

            # Apply filters
            filtered = records
            if selected_archetypes:
                filtered = [r for r in filtered if r.get("synth_archetype") in selected_archetypes]
            if selected_sophis:
                filtered = [r for r in filtered if r.get("synth_sophistication") in selected_sophis]
            if selected_qc:
                filtered = [r for r in filtered if r.get("synth_qc_passed") in selected_qc]
            if search_text:
                q = search_text.lower()
                filtered = [
                    r for r in filtered
                    if q in r.get("Comment", r.get("comment", "")).lower()
                ]

            st.write(f"Showing **{len(filtered)}** of {total_records} records")

            # ── Comment cards ──────────────────────────────────────────────
            show_n = st.slider("Max cards to display", min_value=1, max_value=min(100, len(filtered)), value=min(20, len(filtered)))

            for rec in filtered[:show_n]:
                archetype  = rec.get("synth_archetype", "")
                sophis     = rec.get("synth_sophistication", "")
                state_loc  = rec.get("synth_persona_state", "")
                occ        = rec.get("synth_persona_occupation", "")
                comment_text = rec.get("Comment", rec.get("comment", "(no text)"))
                qc         = rec.get("synth_qc_passed", "")
                vector     = rec.get("synth_vector", "")
                register   = rec.get("synth_emotional_register", "")

                qc_badge = "✅" if qc == "TRUE" else "⚠️"
                label = f"{qc_badge} **{archetype}** / {sophis} — {occ}, {state_loc}"

                with st.expander(label, expanded=False):
                    meta_cols = st.columns(4)
                    meta_cols[0].caption(f"**Vector:** {vector}")
                    meta_cols[1].caption(f"**Register:** {register}")
                    meta_cols[2].caption(f"**QC:** {qc}")
                    meta_cols[3].caption(f"**Sophistication:** {sophis}")

                    st.write(comment_text[:3000] + ("…" if len(comment_text) > 3000 else ""))

                    # Arguments if present
                    args_text = rec.get("synth_core_arguments", "")
                    if args_text:
                        with st.expander("📐 Core Arguments"):
                            st.write(args_text)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Combined CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab_combined:
    st.subheader("Combined CSV (Real + Synthetic)")

    combined_path = Path(docket_id, "shuffled_comments", "combined.csv")
    if not combined_path.is_file():
        st.info(f"No combined CSV found at `{combined_path}`.  Run the Shuffle step first.")
    else:
        try:
            import pandas as pd
            df = pd.read_csv(combined_path, on_bad_lines="skip")
            size_kb = round(combined_path.stat().st_size / 1024, 1)
            st.write(f"**{len(df):,}** rows · {len(df.columns)} columns · {size_kb} KB")

            st.download_button(
                "⬇️ Download combined.csv",
                data=combined_path.read_bytes(),
                file_name="combined.csv",
                mime="text/csv",
            )

            st.dataframe(df, use_container_width=True, height=500)
        except Exception as exc:
            st.error(f"Could not read combined CSV: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Key File
# ─────────────────────────────────────────────────────────────────────────────
with tab_key:
    st.subheader("Key File — Ground Truth Labels")
    st.caption(
        "Each row records whether the corresponding row in `combined.csv` "
        "is `real` or `synthetic`."
    )

    key_path = Path(docket_id, "shuffled_comments", "combined_key.csv")
    if not key_path.is_file():
        st.info(f"No key file found at `{key_path}`.  Run the Shuffle step first.")
    else:
        try:
            import pandas as pd
            key_df = pd.read_csv(key_path)
            size_kb = round(key_path.stat().st_size / 1024, 1)

            # Summary stats
            if "type" in key_df.columns:
                counts = key_df["type"].value_counts()
                real_n = counts.get("real", 0)
                synth_n = counts.get("synthetic", 0)
                total_n = real_n + synth_n

                stat_cols = st.columns(3)
                stat_cols[0].metric("Total Rows", f"{total_n:,}")
                stat_cols[1].metric("Real", f"{real_n:,}")
                stat_cols[2].metric("Synthetic", f"{synth_n:,}")

                # Pie chart
                try:
                    import plotly.express as px
                    fig = px.pie(
                        values=[real_n, synth_n],
                        names=["Real", "Synthetic"],
                        color=["Real", "Synthetic"],
                        color_discrete_map={"Real": "#4c9be8", "Synthetic": "#f0a500"},
                        title="Real vs. Synthetic Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    # Plotly not installed — show a simple bar instead
                    chart_data = pd.DataFrame({"Count": [real_n, synth_n]}, index=["Real", "Synthetic"])
                    st.bar_chart(chart_data)

            st.download_button(
                "⬇️ Download combined_key.csv",
                data=key_path.read_bytes(),
                file_name="combined_key.csv",
                mime="text/csv",
            )

            # Color-coded table
            def _color_type(val: str) -> str:
                if val == "real":
                    return "background-color: #dbeafe; color: #1e3a8a"
                if val == "synthetic":
                    return "background-color: #fef3c7; color: #92400e"
                return ""

            if "type" in key_df.columns:
                styled = key_df.style.applymap(_color_type, subset=["type"])
                st.dataframe(styled, use_container_width=True, height=500)
            else:
                st.dataframe(key_df, use_container_width=True, height=500)

        except Exception as exc:
            st.error(f"Could not read key file: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Voice Skills
# ─────────────────────────────────────────────────────────────────────────────
with tab_skills:
    st.subheader("Voice Skill Files")
    st.caption(
        "Empirical writing-style profiles generated by the Stylometry Analyzer.  "
        "These guide the generator to mimic real commenter patterns."
    )

    skills = list_voice_skills(docket_id)
    if not skills:
        st.info(f"No voice skills found in `{docket_id}/stylometry/`.  Run the Stylometry step.")
    else:
        # Skill selector
        skill_options = {s.stem: s for s in skills}
        selected = st.selectbox(
            "Select a voice skill to preview",
            options=list(skill_options.keys()),
        )

        if selected:
            skill_path = skill_options[selected]
            content = skill_path.read_text(encoding="utf-8", errors="replace")
            size_kb = round(skill_path.stat().st_size / 1024, 1)

            st.caption(f"`{skill_path}` · {size_kb} KB")

            col_render, col_raw = st.columns([1, 1])
            with col_render:
                st.markdown("**Rendered**")
                st.markdown(content)
            with col_raw:
                st.markdown("**Raw Markdown**")
                st.code(content, language="markdown")

        # Full list
        st.divider()
        st.write(f"All **{len(skills)}** voice skill files:")
        rows = [{"File": s.name, "Size (KB)": round(s.stat().st_size / 1024, 1)} for s in skills]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
