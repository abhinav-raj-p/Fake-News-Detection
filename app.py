import streamlit as st
import requests
import time

# 1. Using a more lightweight and active model
MODEL_ID = "dhruvpal/fake-news-bert" 
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# 2. Enhanced Query Function with 'wait_for_model'
def query_model(payload):
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    # We add "wait_for_model": True to tell the server NOT to give an error
    # but to keep the connection alive while it loads.
    options = {"wait_for_model": True}
    
    response = requests.post(
        API_URL, 
        headers=headers, 
        json={"inputs": payload["inputs"], "options": options}, 
        timeout=30
    )
    return response.json()

# --- Inside your Button Logic ---
if st.button("Analyze News"):
    with st.spinner("AI is analyzing (this may take 20-30 seconds if the model is cold)..."):
        full_text = f"{news_title} {news_content}"[:1000]
        output = query_model({"inputs": full_text})
        
        # Checking for the common format of Transformers output
        # Usually [[{'label': 'FAKE', 'score': 0.99}]]
        try:
            if isinstance(output, list):
                result = output[0][0]
                label = result['label']
                score = result['score']
                
                # Note: Labels vary by model (e.g., 'LABEL_0', 'fake', 'FAKE')
                # Check your specific model's output and adjust this if-else
                if "fake" in label.lower() or label == "LABEL_0":
                    st.error(f"🚨 FAKE NEWS DETECTED (Confidence: {int(score*100)}%)")
                else:
                    st.success(f"✅ REAL NEWS DETECTED (Confidence: {int(score*100)}%)")
            else:
                st.warning("The AI is still initializing. Please wait 10 seconds and try one last time.")
        except:
            st.error("Model Error: Could not parse response. Please try a shorter text.")