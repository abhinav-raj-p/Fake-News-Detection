import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

# --- 2. TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. STABLE MODEL (ZERO-SHOT) ---
MODEL_ID = "facebook/bart-large-mnli"

client = InferenceClient(
    model=MODEL_ID,
    token=hf_token,
    timeout=30
)

def analyze_news(text):
    return client.zero_shot_classification(
        text,
        candidate_labels=["REAL NEWS", "FAKE NEWS"]
    )

# --- 4. UI ---
st.title("🛡️ Fake News Detector")
st.markdown(f"Model: `{MODEL_ID}` (zero-shot)")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Analyze", use_container_width=True):
    if title and body:
        with st.spinner("Analyzing content credibility..."):
            text = f"{title}. {body}"[:1000]
            result = analyze_news(text)

            labels = result["labels"]
            scores = result["scores"]

            top_label = labels[0]
            confidence = int(scores[0] * 100)

            if top_label == "FAKE NEWS":
                st.error(f"### 🚨 FAKE NEWS ({confidence}%)")
            else:
                st.success(f"### ✅ REAL NEWS ({confidence}%)")
    else:
        st.warning("Please provide both headline and content.")

st.divider()
st.caption("⚠️ Linguistic pattern detection — not factual verification.")
