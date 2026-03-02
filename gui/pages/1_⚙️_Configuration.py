"""
gui/pages/1_⚙️_Configuration.py — API key and model configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from gui.utils.state import read_env, write_env, masked, _SECRET_KEYS

st.set_page_config(page_title="Configuration — SLOP", page_icon="⚙️", layout="wide")
st.title("⚙️ Configuration")
st.caption("Manage API keys and model settings.  Values are saved to the `.env` file in the repo root.")

st.divider()

# ── Load current values ────────────────────────────────────────────────────────
current = read_env()

DEFAULTS = {
    "SLOP_API_BASE_URL":       "https://api.openai.com/v1",
    "SLOP_EMBED_API_BASE_URL": "https://api.openai.com/v1",
    "SLOP_CHAT_MODEL":         "gpt-4o",
    "SLOP_EMBED_MODEL":        "text-embedding-3-small",
}

def _current(key: str) -> str:
    return current.get(key, DEFAULTS.get(key, ""))


# ── Form ───────────────────────────────────────────────────────────────────────
with st.form("config_form"):
    st.subheader("API Keys")
    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input(
            "Chat API Key (`SLOP_API_KEY`)",
            value=_current("SLOP_API_KEY"),
            type="password",
            help="Required. Your OpenAI (or compatible) API key for comment generation.",
        )
    with col2:
        embed_key = st.text_input(
            "Embedding API Key (`SLOP_EMBED_API_KEY`)",
            value=_current("SLOP_EMBED_API_KEY"),
            type="password",
            help="Required. Used for near-duplicate QC. Can be the same key as above.",
        )

    st.subheader("API Endpoints & Models")
    col3, col4 = st.columns(2)

    with col3:
        api_base = st.text_input(
            "Chat API Base URL (`SLOP_API_BASE_URL`)",
            value=_current("SLOP_API_BASE_URL"),
            help="Base URL for the chat/completions endpoint.",
        )
        chat_model = st.text_input(
            "Chat Model (`SLOP_CHAT_MODEL`)",
            value=_current("SLOP_CHAT_MODEL"),
            help="Model name for comment generation (e.g. gpt-4o, gpt-4o-mini).",
        )

    with col4:
        embed_base = st.text_input(
            "Embedding API Base URL (`SLOP_EMBED_API_BASE_URL`)",
            value=_current("SLOP_EMBED_API_BASE_URL"),
            help="Base URL for the embeddings endpoint.",
        )
        embed_model = st.text_input(
            "Embedding Model (`SLOP_EMBED_MODEL`)",
            value=_current("SLOP_EMBED_MODEL"),
            help="Model name for embeddings (e.g. text-embedding-3-small).",
        )

    submitted = st.form_submit_button("💾 Save to .env", type="primary")

if submitted:
    new_values = {
        "SLOP_API_KEY":            api_key,
        "SLOP_EMBED_API_KEY":      embed_key,
        "SLOP_API_BASE_URL":       api_base,
        "SLOP_EMBED_API_BASE_URL": embed_base,
        "SLOP_CHAT_MODEL":         chat_model,
        "SLOP_EMBED_MODEL":        embed_model,
    }
    # Strip empty strings — don't write blanks for keys not being set
    new_values = {k: v for k, v in new_values.items() if v.strip()}
    write_env(new_values)
    st.success("✅ Settings saved to `.env`")

st.divider()

# ── Current values display ─────────────────────────────────────────────────────
st.subheader("Current .env Values")

refreshed = read_env()
if not refreshed:
    st.warning("No `.env` file found yet.  Fill in the form above and click **Save**.")
else:
    display_rows = []
    all_keys = [
        "SLOP_API_KEY", "SLOP_EMBED_API_KEY",
        "SLOP_API_BASE_URL", "SLOP_EMBED_API_BASE_URL",
        "SLOP_CHAT_MODEL", "SLOP_EMBED_MODEL",
    ]
    for k in all_keys:
        v = refreshed.get(k, "")
        display_rows.append({
            "Variable": k,
            "Value": masked(v) if k in _SECRET_KEYS else (v or "(not set)"),
            "Status": "✅" if v else "⚠️ Not set",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Test connection ────────────────────────────────────────────────────────────
st.subheader("Test API Connection")
st.caption("Sends a minimal request to verify your keys are accepted.")

test_col1, test_col2 = st.columns(2)

with test_col1:
    if st.button("🔌 Test Chat API"):
        env = read_env()
        key = env.get("SLOP_API_KEY", "")
        base = env.get("SLOP_API_BASE_URL", "https://api.openai.com/v1")
        model = env.get("SLOP_CHAT_MODEL", "gpt-4o")
        if not key:
            st.error("SLOP_API_KEY is not set.")
        else:
            with st.spinner("Testing..."):
                try:
                    import openai
                    client = openai.OpenAI(api_key=key, base_url=base)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_tokens=5,
                    )
                    st.success(f"✅ Chat API OK — model `{model}` responded: *{resp.choices[0].message.content.strip()}*")
                except Exception as exc:
                    st.error(f"❌ Chat API error: {exc}")

with test_col2:
    if st.button("🔌 Test Embedding API"):
        env = read_env()
        key = env.get("SLOP_EMBED_API_KEY", "")
        base = env.get("SLOP_EMBED_API_BASE_URL", "https://api.openai.com/v1")
        model = env.get("SLOP_EMBED_MODEL", "text-embedding-3-small")
        if not key:
            st.error("SLOP_EMBED_API_KEY is not set.")
        else:
            with st.spinner("Testing..."):
                try:
                    import openai
                    client = openai.OpenAI(api_key=key, base_url=base)
                    resp = client.embeddings.create(model=model, input=["test"])
                    dim = len(resp.data[0].embedding)
                    st.success(f"✅ Embedding API OK — model `{model}`, vector dim={dim}")
                except Exception as exc:
                    st.error(f"❌ Embedding API error: {exc}")
