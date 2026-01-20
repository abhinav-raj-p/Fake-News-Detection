import streamlit as st
import pickle
import re

# 1. Load the saved model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# 2. Function to clean user input (must match training logic)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 3. Streamlit UI
st.title("🛡️ Student Fake News Detector")
st.subheader("Enter a news article below to check its authenticity.")

user_input = st.text_area("Paste News Content Here:", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess the input
        cleaned_input = clean_text(user_input)
        
        # Vectorize the input
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(vectorized_input)
        
        # Display Result
        if prediction[0] == 'REAL':
            st.success("✅ This news appears to be REAL.")
        else:
            st.error("🚨 This news appears to be FAKE.")