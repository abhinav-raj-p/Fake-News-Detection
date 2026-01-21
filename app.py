import streamlit as st
import requests
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI News Verifier", page_icon="🛡️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. THE NEW 2026 ROUTER SETUP ---
# THE FIX: Updated the URL to use the new router.huggingface.co endpoint
MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news-detection"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def predict_news(text):
    # 'wait_for_model' handles the "waking up" phase automatically
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    # Internal loop to handle server connection
    for i in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503: # Model still loading
                time.sleep(10)
                continue
            else:
                return {"error": response.status_code, "msg": response.text}
        except Exception as e:
            return {"error": "ConnectionError", "msg": str(e)}
            
    return {"error": "Timeout", "msg": "Server took too long to respond."}

# --- 4. USER INTERFACE ---
st.title("🛡️ AI News Integrity Check")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Analyze Now", use_container_width=True):
    if title and body:
        with st.spinner("AI is analyzing via Hugging Face Router..."):
            # Combining title and body is the standard input format for BERT
            full_text = f"{title} {body}"[:1000] 
            
            data = predict_news(full_text)
            
            if isinstance(data, list):
                # Process standard classification results
                res = data[0][0]
                label = res['label'].upper()
                score = res['score']
                
                if "FAKE" in label or "LABEL_0" in label:
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
            else:
                st.error(f"Technical Issue: {data.get('error')} - {data.get('msg')}")
    else:
        st.warning("Please fill in both fields.")

st.divider()
st.caption("2026 Infrastructure | Powered by Hugging Face Router")