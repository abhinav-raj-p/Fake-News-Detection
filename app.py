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
# Fetches the token from the Streamlit Cloud "Secrets" tab
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Credential Error: 'HF_TOKEN' not found in Streamlit Secrets.")
    st.stop()

# --- 3. API SETUP (Stable 2026 Router URL) ---
MODEL_ID = "dhruvpal/fake-news-bert"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(payload):
    # 'wait_for_model' tells the server to wait for loading instead of returning a 503 error
    data = {
        "inputs": payload,
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"loading": True, "message": "Model is still loading."}
        else:
            return {"error": f"API Error {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        return {"error": "Connection Error", "details": str(e)}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208035.png", width=80)
    st.title("Project Information")
    st.info("Utilizing BERT (Transformers) to detect linguistic patterns in digital misinformation.")
    st.markdown("---")
    st.subheader("Instructions")
    st.write("1. Enter the news headline.\n2. Paste the full article text.\n3. Click Analyze News.")

# --- 5. MAIN INTERFACE ---
st.title("🛡️ Student AI Fake News Detector")
st.write("An intelligent application for verifying news authenticity via deep learning.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 News Input")
    news_title = st.text_input("News Headline", placeholder="e.g., New discovery on Mars...")
    news_content = st.text_area("News Content", height=250, placeholder="Paste the full article content here...")

with col2:
    st.subheader("🔍 Analysis Verdict")
    if st.button("Analyze News Integrity", use_container_width=True):
        if news_title and news_content:
            with st.spinner("AI is analyzing context (may take up to 30s for cold start)..."):
                # Combine Title + Content for full context
                combined_text = f"{news_title} {news_content}"[:1200]
                
                output = query_model(combined_text)

                # Check for the different types of responses
                if isinstance(output, dict) and "loading" in output:
                    st.info("The AI model is still waking up. Please wait 10 seconds and try again.")
                
                elif isinstance(output, dict) and "error" in output:
                    st.error(f"Analysis Failed: {output['error']}")
                    with st.expander("See technical details"):
                        st.write(output.get("details", "No details available."))
                
                elif isinstance(output, list) and len(output) > 0:
                    try:
                        # Standard format: [[{'label': 'LABEL_1', 'score': 0.99}]]
                        prediction = output[0][0]
                        label = prediction['label']
                        score = prediction['score']

                        # Logic for dhruvpal/fake-news-bert: LABEL_1 = Real, LABEL_0 = Fake
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
                        st.write(f"The Transformer model detected patterns consistent with **{'accurate' if label == 'LABEL_1' else 'misleading'}** reporting style.")
                    
                    except (KeyError, IndexError):
                        st.error("Error: Could not parse the AI's response. Please try again.")
                else:
                    st.error("The API returned an unexpected response format.")
        else:
            st.warning("Please provide both a Headline and Content.")

# --- 6. FOOTER ---
st.divider()
st.caption("Securely Deployed for Silver Oak University | Powered by Hugging Face & Streamlit")