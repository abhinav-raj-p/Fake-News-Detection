import streamlit as st
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

# --- SECURE API SETUP ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Add 'HF_TOKEN' to Streamlit Secrets first.")
    st.stop()

# Stable URL for dhruvpal model
API_URL = "https://api-inference.huggingface.co/models/dhruvpal/fake-news-bert"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(text):
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    return response.json()

# --- UI ---
st.title("🛡️ AI Fake News Detector")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Headline")
    content = st.text_area("Article Content", height=200)

with col2:
    if st.button("Analyze News", use_container_width=True):
        if title and content:
            with st.spinner("AI analyzing..."):
                output = query_model(f"{title} {content}"[:1200])
                try:
                    # Parse standard Transformers response
                    res = output[0][0]
                    # dhruvpal labels: LABEL_1 = Fake, LABEL_0 = Real
                    if res['label'] == "LABEL_1":
                        st.error(f"🚨 FAKE (Confidence: {int(res['score']*100)}%)")
                    else:
                        st.success(f"✅ REAL (Confidence: {int(res['score']*100)}%)")
                except:
                    st.error("API error. Model might still be waking up. Try again in 10s.")
        else:
            st.warning("Please fill both fields.")