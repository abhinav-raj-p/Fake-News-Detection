import streamlit as st
import pickle
import re
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource # Caches the model so it doesn't reload on every interaction
def load_assets():
    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure .pkl files are in the directory.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208035.png", width=100) # Educational icon
    st.title("About Project")
    st.info("This AI tool uses a Passive Aggressive Classifier to detect misinformation patterns in news headlines and bodies.")
    st.markdown("---")
    st.subheader("Instructions")
    st.write("1. Enter the news headline.\n2. Paste the article body.\n3. Click Analyze.")

# --- MAIN UI ---
st.title("🛡️ AI Student Fake News Detector")
st.markdown("Developed to help students verify digital information effectively.")

# Two columns for input
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input News Details")
    news_title = st.text_input("News Title / Headline", placeholder="e.g., Breaking: New discovery found on Mars...")
    news_content = st.text_area("News Article Content", height=250, placeholder="Paste the full article text here...")

# Analysis Logic
with col2:
    st.subheader("🔍 Analysis Result")
    if st.button("Analyze News Integrity", use_container_width=True):
        if not news_title or not news_content:
            st.warning("Please provide both a Title and Content for accurate detection.")
        else:
            with st.spinner("AI is analyzing linguistic patterns..."):
                # Combine Title + Content for the model
                full_input = news_title + " " + news_content
                cleaned = clean_text(full_input)
                vectorized = vectorizer.transform([cleaned])
                
                # Prediction
                prediction = model.predict(vectorized)[0]
                
                # Confidence Score (using decision_function as PAC doesn't have predict_proba)
                # We normalize the distance from the hyperplane to show a pseudo-confidence
                score = model.decision_function(vectorized)[0]
                confidence = float(np.clip((abs(score) / 2), 0.5, 0.99)) # Estimated scale

                if prediction == 'REAL':
                    st.success(f"### Result: ✅ REAL NEWS")
                    st.progress(confidence, text=f"Model Confidence: {int(confidence*100)}%")
                    st.balloons()
                else:
                    st.error(f"### Result: 🚨 FAKE NEWS")
                    st.progress(confidence, text=f"Suspicion Level: {int(confidence*100)}%")
                    st.snow()
                
                st.markdown("---")
                st.write("**Analysis Summary:**")
                st.write(f"The model detected patterns consistent with **{prediction.lower()}** reporting.")

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes. Always cross-verify news with reputable fact-checking sources.")