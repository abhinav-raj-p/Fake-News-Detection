import streamlit as st
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- 2. SECURE API TOKEN ACCESS ---
# This pulls the token you pasted in the Streamlit Cloud Secrets tab
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: 'HF_TOKEN' not found in Streamlit Secrets. Please check your dashboard.")
    st.stop()

# --- 3. API SETUP ---
# Using the 2026 Router URL for better reliability
MODEL_ID = "dhruvpal/fake-news-bert"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
    return response.json()

# --- 4. SIDEBAR (Matching the professional style of your example PPT) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208035.png", width=80)
    st.title("Project Info")
    st.info("This system uses a BERT (Bidirectional Encoder Representations from Transformers) model to detect misinformation.")
    st.markdown("---")
    st.subheader("How to use:")
    st.write("1. Enter the headline.\n2. Paste the news body.\n3. Click Analyze.")

# --- 5. MAIN USER INTERFACE ---
st.title("🛡️ Student AI Fake News Detector")
st.write("An intelligent system to verify news authenticity using Deep Learning.")

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 News Input")
    news_title = st.text_input("News Headline", placeholder="e.g., Scientist discover life on Mars...")
    news_content = st.text_area("News Content", height=250, placeholder="Paste the full article text here...")

with col2:
    st.subheader("🔍 Result & Analysis")
    if st.button("Run Analysis", use_container_width=True):
        if news_title and news_content:
            with st.spinner("AI is analyzing linguistic context..."):
                # Combine Title and Content for BERT (limit to 1200 characters for API speed)
                combined_text = f"{news_title} {news_content}"[:1200]
                
                output = query_model({"inputs": combined_text})

                # Handle "Model Loading" status
                if isinstance(output, dict) and "error" in output:
                    st.info("The AI model is waking up. Please wait 10 seconds and try again.")
                else:
                    try:
                        # Extract prediction data
                        prediction = output[0][0]
                        label = prediction['label']
                        score = prediction['score']

                        # Display results based on model labels
                        # Pulk17 model: LABEL_1 = Real, LABEL_0 = Fake
                        if label == "LABEL_1":
                            st.success(f"### Result: ✅ PROBABLY REAL")
                            st.progress(score, text=f"Model Confidence: {int(score*100)}%")
                            st.balloons()
                        else:
                            st.error(f"### Result: 🚨 PROBABLY FAKE")
                            st.progress(score, text=f"Suspicion Level: {int(score*100)}%")
                            st.snow()
                            
                        st.markdown("---")
                        st.write("**AI Observation:**")
                        st.write(f"The model detected patterns consistent with **{ 'genuine' if label == 'LABEL_1' else 'misleading' }** reporting.")
                    
                    except (KeyError, IndexError, TypeError):
                        st.error("The API is currently busy. Please try again in a moment.")
        else:
            st.warning("Please provide both a Headline and the Article Content.")

# --- 6. FOOTER ---
st.divider()
st.caption("Developed for Silver Oak University | Securely Deployed via Streamlit Community Cloud")