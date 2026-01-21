import streamlit as st
import requests
import time

# --- CONFIG ---
st.set_page_config(page_title="News Verifier", page_icon="🕵️")

# --- SECURE TOKEN ---
# 1. DELETE your old token on Hugging Face.
# 2. CREATE a new one. Select ONLY "Inference" -> "Make calls to Inference Providers".
# 3. PASTE it into Streamlit Secrets as HF_TOKEN.
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# Using a high-stability model
API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"
headers = {"Authorization": f"Bearer {hf_token}"}

def predict_news(text):
    # 'wait_for_model' forces the server to stay connected while the model loads
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    # We try 3 times internally so the user doesn't see errors
    for i in range(3):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            time.sleep(10) # Wait 10s if the model is waking up
            continue
        else:
            return {"error": response.status_code, "msg": response.text}
    return {"error": "Timeout", "msg": "Model took too long to wake up."}

# --- UI ---
st.title("🛡️ AI News Integrity Check")

title = st.text_input("News Headline")
body = st.text_area("Article Content")

if st.button("Analyze Now"):
    if title and body:
        with st.spinner("AI is analyzing (this can take 30s for the first run)..."):
            # Combining title and body is CORRECT for BERT models
            full_text = f"{title} {body}"[:1000] 
            
            data = predict_news(full_text)
            
            if isinstance(data, list):
                res = data[0][0]
                label = res['label'].upper()
                score = res['score']
                
                if "FAKE" in label or "LABEL_0" in label:
                    st.error(f"🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                else:
                    st.success(f"✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
            else:
                st.error(f"Technical Issue: {data.get('error')} - {data.get('msg')}")
    else:
        st.warning("Please fill in both fields.")