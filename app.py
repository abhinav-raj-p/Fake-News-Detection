import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. LIGHTWEIGHT MODEL (FAST & FREE) ---
MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news"

client = InferenceClient(
    model=MODEL_ID,
    token=hf_token,
    timeout=30
)

def analyze_news(text):
    return client.text_classification(text)

# --- 4. USER INTERFACE ---
st.title("🛡️ Fake News Detector")
st.markdown(f"Model: `{MODEL_ID}`")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Analyze", use_container_width=True):
    if title and body:
        with st.spinner("Analyzing linguistic patterns..."):
            full_text = f"{title}. {body}"[:1000]
            result = analyze_news(full_text)

            if isinstance(result, list) and len(result) > 0:
                pred = max(result, key=lambda x: x["score"])
                label = pred["label"].upper()
                confidence = int(pred["score"] * 100)

                if "FAKE" in label or label == "LABEL_1":
                    st.error(f"### 🚨 FAKE NEWS ({confidence}%)")
                else:
                    st.success(f"### ✅ REAL NEWS ({confidence}%)")
            else:
                st.error("Model returned an unexpected response.")
    else:
        st.warning("Please provide both headline and content.")

st.divider()
st.caption("⚠️ This tool detects linguistic patterns, not factual truth.")
