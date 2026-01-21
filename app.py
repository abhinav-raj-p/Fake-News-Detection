import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="High-Speed News AI", page_icon="⚡")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Credential Error: Please add 'HF_TOKEN' to your Streamlit Cloud Secrets.")
    st.stop()

# --- 3. THE 2026 STABLE MODEL ---
# Using a high-uptime DistilBERT model specifically for fake news
MODEL_ID = "therealcyberlord/fake-news-classification-distilbert" 

# Setting timeout to 120s to specifically prevent 504 Gateway Timeouts
client = InferenceClient(model=MODEL_ID, token=hf_token, timeout=120)

def analyze_news(text):
    try:
        # text_classification is faster than zero-shot
        return client.text_classification(text)
    except Exception as e:
        return {"error": str(e)}

# --- 4. USER INTERFACE ---
st.title("⚡ High-Speed News Verifier")
st.write("Optimized for low-latency inference on the 2026 Router.")

title = st.text_input("Headline")
body = st.text_area("Content", height=200)

if st.button("Run Instant Analysis", use_container_width=True):
    if title and body:
        with st.spinner("AI is analyzing (using 120s connection persistence)..."):
            # Limit to 1000 chars for maximum speed
            full_text = f"{title} {body}"[:1000] 
            data = analyze_news(full_text)
            
            if isinstance(data, list) and len(data) > 0:
                # Label logic: 0 = Fake, 1 = Real
                pred = sorted(data, key=lambda x: x['score'], reverse=True)[0]
                label = pred['label']
                score = pred['score']
                
                if label in ["0", "LABEL_0"]:
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}%)")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}%)")
            else:
                st.error(f"Technical Error: {data.get('error', 'The server timed out. Please try again in 10 seconds.')}")
    else:
        st.warning("Please fill in both fields.")