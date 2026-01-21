import streamlit as st
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

# --- 2. SECURE API ACCESS ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: Please add 'HF_TOKEN' to your Streamlit Cloud Secrets.")
    st.stop()

# --- 3. STABLE MODEL SETUP ---
# Using a "Tiny" model ensures it is almost always active and fast
MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news-detection"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(text):
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }
    # Simple, fast request
    response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
    return response.json()

# --- 4. USER INTERFACE ---
st.title("🛡️ AI Fake News Detector")
st.write("Verifying news authenticity using high-speed Transformer models.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 News Content")
    news_title = st.text_input("Headline")
    news_content = st.text_area("Article Body", height=250)

with col2:
    st.subheader("🔍 Verdict")
    if st.button("Run Analysis", use_container_width=True):
        if news_title and news_content:
            with st.spinner("AI is analyzing..."):
                full_text = f"{news_title} {news_content}"[:1000]
                output = query_model(full_text)
                
                try:
                    # Tiny-BERT model labels: 'fake' and 'real'
                    if isinstance(output, list) and len(output) > 0:
                        res = output[0][0]
                        label = res['label'].lower()
                        score = res['score']

                        if "fake" in label:
                            st.error(f"### Result: 🚨 FAKE NEWS")
                            st.progress(score, text=f"Suspicion Level: {int(score*100)}%")
                        else:
                            st.success(f"### Result: ✅ REAL NEWS")
                            st.progress(score, text=f"AI Confidence: {int(score*100)}%")
                    else:
                        st.error("The API service is currently busy. Please try again in 10s.")
                except Exception:
                    st.error("Analysis Failed. Please check your internet or API token.")
        else:
            st.warning("Please provide both headline and body.")

st.divider()
st.caption("Securely ")