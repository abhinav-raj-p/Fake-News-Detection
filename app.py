import streamlit as st
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- 2. SECURE API ACCESS ---
# Ensure "HF_TOKEN" is added in your Streamlit Cloud Secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: 'HF_TOKEN' not found in Streamlit Secrets.")
    st.stop()

# --- 3. API SETUP (Using dhruvpal/fake-news-bert) ---
MODEL_ID = "dhruvpal/fake-news-bert"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(payload):
    # 'wait_for_model' helps avoid the "model is waking up" error by waiting for it to load
    response = requests.post(
        API_URL, 
        headers=headers, 
        json={"inputs": payload, "options": {"wait_for_model": True}}, 
        timeout=30
    )
    return response.json()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208035.png", width=80)
    st.title("Project Details")
    st.info("This system leverages the BERT architecture to identify linguistic patterns common in misinformation.")
    st.markdown("---")
    st.subheader("How to Analyze")
    st.write("1. Provide the headline.\n2. Paste the full content.\n3. Click Run Analysis.")

# --- 5. MAIN INTERFACE ---
st.title("🛡️ Student AI Fake News Detector")
st.write("A deep-learning approach to verifying news authenticity.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Section")
    news_title = st.text_input("News Headline", placeholder="e.g., Breaking news headline...")
    news_content = st.text_area("News Content", height=250, placeholder="Paste the full article content here...")

with col2:
    st.subheader("🔍 Analysis Verdict")
    if st.button("Run Analysis", use_container_width=True):
        if news_title and news_content:
            with st.spinner("AI is evaluating context (this may take a moment)..."):
                # Combine Title + Content for full context
                combined_text = f"{news_title} {news_content}"[:1200]
                
                output = query_model(combined_text)

                try:
                    # Parse the standard Transformers output: [[{'label': '...', 'score': ...}]]
                    if isinstance(output, list) and len(output) > 0:
                        prediction = output[0][0]
                        label = prediction['label']
                        score = prediction['score']

                        # Logic to handle dhruvpal's specific labels
                        # Usually 'LABEL_1' is Real and 'LABEL_0' is Fake
                        if label == "LABEL_1":
                            st.success(f"### Result: ✅ PROBABLY REAL")
                            st.progress(score, text=f"AI Confidence: {int(score*100)}%")
                            st.balloons()
                        else:
                            st.error(f"### Result: 🚨 PROBABLY FAKE")
                            st.progress(score, text=f"Suspicion Level: {int(score*100)}%")
                            st.snow()
                            
                        st.markdown("---")
                        st.write("**Technical Insight:**")
                        st.write(f"The Transformer model identified features highly consistent with **{ 'accurate' if label == 'LABEL_1' else 'misleading' }** reporting.")
                    else:
                        st.warning("The model is still initializing on the server. Please try again in 10 seconds.")
                
                except (KeyError, IndexError, TypeError):
                    st.error("API error: Could not process the response. Please try a different news sample.")
        else:
            st.warning("Both the Headline and Content are required for a valid analysis.")

# --- 6. FOOTER ---
st.divider()
st.caption("Securely Deployed for Silver Oak University | Built with Hugging Face & Streamlit")