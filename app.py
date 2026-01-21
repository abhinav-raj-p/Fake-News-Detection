import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI News Verifier", page_icon="🛡️")

# --- 2. SECURE TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. STABLE 2026 MODEL ---
# Using this model because it is a 'Recommended' model for text-classification in 2026
MODEL_ID = "facebook/bart-large-mnli" 
client = InferenceClient(model=MODEL_ID, token=hf_token)

def analyze_text(text):
    try:
        # Zero-shot classification: We tell it what labels to look for
        # This is more accurate than pre-set labels that might change
        labels = ["real news", "fake news"]
        result = client.zero_shot_classification(text, candidate_labels=labels)
        return result
    except Exception as e:
        return {"error": str(e)}

# --- 4. USER INTERFACE ---
st.title("🛡️ AI News Integrity Check")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Run Analysis", use_container_width=True):
    if title and body:
        with st.spinner("AI is analyzing via Hugging Face Hub..."):
            full_text = f"{title} {body}"[:1000] 
            data = analyze_text(full_text)
            
            if "error" not in data:
                # The result comes back as a list of dicts with labels and scores
                # e.g., [{'label': 'real news', 'score': 0.95}, {'label': 'fake news', 'score': 0.05}]
                best_match = data[0]
                label = best_match['label']
                score = best_match['score']
                
                if label == "fake news":
                    st.error(f"### 🚨 FAKE NEWS DETECTED ({int(score*100)}% confidence)")
                else:
                    st.success(f"### ✅ REAL NEWS DETECTED ({int(score*100)}% confidence)")
            else:
                st.error(f"Technical Issue: {data['error']}")
    else:
        st.warning("Please fill in both fields.")