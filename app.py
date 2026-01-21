import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Speed-Optimized News AI", page_icon="⚡")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. HIGH-AVAILABILITY MODEL ---
# Switch to DistilBERT - it is 60% faster and rarely triggers 504 timeouts
MODEL_ID = "therealcyberlord/fake-news-classification-distilbert" 

# Set the timeout to 120 seconds to prevent the 504 Gateway Timeout
client = InferenceClient(model=MODEL_ID, token=hf_token, timeout=120)

def analyze_news(text):
    try:
        # text_classification is significantly faster than zero-shot
        return client.text_classification(text)
    except Exception as e:
        return {"error": str(e)}

# --- 4. USER INTERFACE ---
st.title("⚡ High-Speed News Verifier")

title = st.text_input("Headline")
body = st.text_area("Content", height=200)

if st.button("Instant Analysis", use_container_width=True):
    if title and body:
        with st.spinner("AI is processing (using optimized 120s timeout)..."):
            # Limit text length to prevent processing delays
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
                st.error(f"Technical Error: {data.get('error', 'Server Timeout')}")
    else:
        st.warning("Please fill in both fields.")