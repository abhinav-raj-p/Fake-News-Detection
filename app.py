import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="BERT News Verifier", page_icon="🛡️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. DHRUVPAL MODEL SETUP ---
# Switching back to your preferred model
MODEL_ID = "dhruvpal/fake-news-bert" 
client = InferenceClient(model=MODEL_ID, token=hf_token)

def analyze_news(text):
    try:
        # Standard text classification call
        # DistilBERT models usually handle up to 512 tokens
        result = client.text_classification(text)
        return result
    except Exception as e:
        return {"error": str(e)}

# --- 4. USER INTERFACE ---
st.title("🛡️ AI News Integrity Check")
st.markdown(f"Running on Model: `{MODEL_ID}`")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Run Analysis", use_container_width=True):
    if title and body:
        with st.spinner("Analyzing via BERT Transformer..."):
            full_text = f"{title} {body}"[:1200] 
            data = analyze_news(full_text)
            
            if isinstance(data, list) and len(data) > 0:
                # dhruvpal labels: LABEL_1 = Fake, LABEL_0 = Real
                # We sort to get the highest score first
                prediction = sorted(data, key=lambda x: x['score'], reverse=True)[0]
                label = prediction['label']
                score = prediction['score']
                
                if label == "LABEL_1":
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("The model identified linguistic markers associated with misinformation.")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
                    st.write("The content structure is consistent with legitimate reporting.")
            else:
                st.error(f"Technical Issue: {data.get('error', 'Unknown Error')}")
    else:
        st.warning("Please fill in both fields.")