import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Qwen AI News Verifier", page_icon="🕵️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. QWEN 2.5 MODEL SETUP ---
MODEL_ID = "Feargal/qwen2.5-fake-news-v1" 
# Increase timeout to 120s as generative models are larger than BERT
client = InferenceClient(model=MODEL_ID, token=hf_token, timeout=120)

def analyze_news(text):
    try:
        # For Qwen 2.5, we use text_classification if it's fine-tuned for it
        # Otherwise, the client automatically handles the request routing
        return client.text_classification(text)
    except Exception as e:
        return {"error": str(e)}

# --- 4. USER INTERFACE ---
st.title("🛡️ Qwen 2.5 Fake News Detector")
st.markdown(f"Running on Advanced 2026 Engine: `{MODEL_ID}`")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Run Advanced Analysis", use_container_width=True):
    if title and body:
        with st.spinner("Qwen 2.5 is evaluating linguistic patterns..."):
            # Combine and truncate for efficiency
            full_text = f"Headline: {title}\nContent: {body}"[:1200]
            data = analyze_news(full_text)
            
            if isinstance(data, list) and len(data) > 0:
                # Sort to get the highest confidence label
                prediction = sorted(data, key=lambda x: x['score'], reverse=True)[0]
                label = prediction['label'].upper()
                score = prediction['score']
                
                # Dynamic Label Mapping for Qwen
                # Typically uses FAKE/REAL or LABEL_0/LABEL_1
                if "FAKE" in label or label == "LABEL_1":
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("The Qwen engine identified stylistic markers typical of misinformation.")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("The content structure matches standard journalistic reporting.")
            else:
                st.error(f"Technical Issue: {data.get('error', 'Model is currently loading or offline.')}")
    else:
        st.warning("Please provide headline and body content.")

st.divider()
st.caption("Powered by Qwen 2.5 Architecture | 2026 Infrastructure")