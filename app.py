import streamlit as st
import requests
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Fake News Detector", page_icon="🛡️", layout="wide")

# --- 2. SECURE API SETUP ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: Please add 'HF_TOKEN' to your Streamlit Cloud Secrets.")
    st.stop()

# Model details from your search
MODEL_ID = "dhruvpal/fake-news-bert"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

# --- 3. PERSISTENT QUERY FUNCTION ---
def query_model(text):
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True} # Critical for free-tier reliability
    }
    
    # Internal loop to handle "Model Waking Up" without crashing
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
            output = response.json()
            
            # If model is loading, it provides an estimated wait time
            if isinstance(output, dict) and "estimated_time" in output:
                time.sleep(5)
                continue
            return output
        except Exception as e:
            if attempt == 2: return {"error": str(e)}
            time.sleep(2)
    return output

# --- 4. USER INTERFACE ---
st.title("🛡️ Student AI Fake News Detector")
st.markdown("Verifying news authenticity using **DistilBERT Transformers**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input News")
    news_title = st.text_input("Headline", placeholder="Enter news headline...")
    news_content = st.text_area("Article Body", height=250, placeholder="Paste article content...")

with col2:
    st.subheader("🔍 Analysis Verdict")
    if st.button("Run Deep Analysis", use_container_width=True):
        if news_title and news_content:
            with st.spinner("AI is analyzing (this handles 'cold starts' automatically)..."):
                # Combine title and content
                full_text = f"{news_title} {news_content}"[:1200]
                output = query_model(full_text)

                try:
                    # Logic for dhruvpal/fake-news-bert: LABEL_1 = Fake, LABEL_0 = Real
                    if isinstance(output, list) and len(output) > 0:
                        prediction = output[0][0]
                        label = prediction['label']
                        score = prediction['score']

                        if label == "LABEL_1":
                            st.error(f"### Result: 🚨 PROBABLY FAKE")
                            st.progress(score, text=f"Suspicion Level: {int(score*100)}%")
                        else:
                            st.success(f"### Result: ✅ PROBABLY REAL")
                            st.progress(score, text=f"AI Confidence: {int(score*100)}%")
                    else:
                        st.error("The AI service is currently overloaded. Please try again in a moment.")
                except Exception:
                    st.error("Could not parse AI response. Check your API Token.")
        else:
            st.warning("Please provide both headline and content.")

st.divider()
st.caption("Securely Deployed for Silver Oak University | Powered by Hugging Face Inference API")