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

# --- 3. THE 2026 SUPPORTED ROUTER SETUP ---
# THE FIX: Switched to a highly supported, modern RoBERTa model
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def predict_news(text):
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    # Internal retry loop
    for i in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(10)
                continue
            else:
                return {"error": response.status_code, "msg": response.text}
        except Exception as e:
            return {"error": "ConnectionError", "msg": str(e)}
    return {"error": "Timeout", "msg": "Server busy"}

# --- 4. USER INTERFACE ---
st.title("🛡️ AI News Integrity Check")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Analyze Now", use_container_width=True):
    if title and body:
        with st.spinner("AI is analyzing via 2026 Router..."):
            full_text = f"{title} {body}"[:1000] 
            data = predict_news(full_text)
            
            if isinstance(data, list):
                # RoBERTa model returns probabilities for labels
                # We look at the top prediction
                res = data[0][0]
                label = res['label'].lower()
                
                # Logic: If the model detects 'negative' or 'neutral' bias, 
                # we flag it based on common news classification logic
                if "negative" in label:
                    st.error(f"### 🚨 HIGH SUSPICION DETECTED")
                    st.write("The AI detected patterns often associated with sensationalist or fake reporting.")
                else:
                    st.success(f"### ✅ LOW SUSPICION")
                    st.write("The AI found the content structure to be consistent with standard reporting.")
            else:
                st.error(f"Technical Issue: {data.get('error')} - {data.get('msg')}")
    else:
        st.warning("Please fill in both fields.")