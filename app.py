import streamlit as st
import requests
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Fake News Detector", page_icon="🕵️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. THE 2026 ROUTER SETUP ---
# THE FIX: Using a DistilBERT model trained specifically on fake news data
MODEL_ID = "vignesh-m/distilbert-base-uncased-finetuned-fake-news"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def predict_news(text):
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
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
    return {"error": "Timeout", "msg": "Server Busy"}

# --- 4. USER INTERFACE ---
st.title("🛡️ AI Fake News Detector")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Run Deep Analysis", use_container_width=True):
    if title and body:
        with st.spinner("AI is analyzing text context..."):
            full_text = f"{title} {body}"[:1000] 
            data = predict_news(full_text)
            
            if isinstance(data, list):
                # DistilBERT Output: Usually [[{'label': 'FAKE', 'score': ...}]] 
                # OR [[{'label': 'LABEL_0', 'score': ...}]]
                res = data[0][0]
                label = res['label'].upper()
                score = res['score']
                
                # Check for FAKE labels (Model specific)
                if "FAKE" in label or label == "LABEL_0":
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("This article shows linguistic patterns typical of misinformation.")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("This content is consistent with standard reporting styles.")
            else:
                st.error(f"Technical Issue: {data.get('error')} - {data.get('msg')}")
    else:
        st.warning("Please fill in both fields.")