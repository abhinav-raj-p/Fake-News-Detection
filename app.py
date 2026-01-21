import streamlit as st
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Fake News Detector", page_icon="🕵️", layout="wide")

# --- 2. SECURE API ACCESS ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: Please add 'HF_TOKEN' to your Streamlit Cloud Secrets.")
    st.stop()

# --- 3. ROBUST API LOGIC ---
# Using the standard Inference endpoint which is most stable for this model
MODEL_ID = "dhruvpal/fake-news-bert"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(text):
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True} # Critical: Tells HF to wait until model is loaded
    }
    
    # Internal Retry Loop to handle "Waking Up" states automatically
    max_retries = 3
    for i in range(max_retries):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        output = response.json()
        
        # If model is loading, it returns an 'estimated_time'
        if isinstance(output, dict) and "estimated_time" in output:
            wait_time = output.get("estimated_time", 10)
            time.sleep(min(wait_time, 15)) # Wait and retry internally
            continue
        return output
    return output

# --- 4. MAIN INTERFACE ---
st.title("🛡️ Student AI Fake News Detector")
st.write("Verifying news authenticity using deep learning Transformer models.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 News Content")
    title = st.text_input("Headline")
    content = st.text_area("Article Body", height=250)

with col2:
    st.subheader("🔍 Verdict")
    if st.button("Analyze News", use_container_width=True):
        if title and content:
            with st.spinner("AI is waking up and analyzing... this may take a minute."):
                output = query_model(f"{title} {content}"[:1200])
                
                try:
                    # Parse output format: [[{'label': 'LABEL_1', 'score': 0.99}]]
                    if isinstance(output, list) and len(output) > 0:
                        res = output[0][0]
                        # dhruvpal labels: LABEL_1 = Fake, LABEL_0 = Real
                        if res['label'] == "LABEL_1":
                            st.error(f"🚨 FAKE NEWS (Confidence: {int(res['score']*100)}%)")
                        else:
                            st.success(f"✅ REAL NEWS (Confidence: {int(res['score']*100)}%)")
                    else:
                        st.error("Server Busy: Hugging Face is overloaded. Please try again in 30 seconds.")
                except:
                    st.error("Unexpected response. Please ensure your API token is valid.")
        else:
            st.warning("Please enter both headline and body.")

st.divider()
st.caption("Developed for Silver Oak University | Powered by BERT")