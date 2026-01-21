import streamlit as st
from huggingface_hub import InferenceClient
import httpx

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

# --- 2. TOKEN ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Please add HF_TOKEN to your Streamlit Secrets.")
    st.stop()

# --- 3. FAST + STABLE ZERO-SHOT MODEL ---
MODEL_ID = "valhalla/distilbart-mnli-12-1"

client = InferenceClient(
    model=MODEL_ID,
    token=hf_token,
    timeout=20
)

def analyze_news(text):
    try:
        return client.zero_shot_classification(
            text,
            candidate_labels=["REAL NEWS", "FAKE NEWS"]
        )
    except httpx.ReadTimeout:
        return {"error": "Model timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}

# --- 4. UI ---
st.title("🛡️ Fake News Detector")
st.markdown(f"Model: `{MODEL_ID}` (fast zero-shot)")

title = st.text_input("News Headline")
body = st.text_area("Article Content", height=200)

if st.button("Analyze", use_container_width=True):
    if title and body:
        with st.spinner("Analyzing linguistic patterns..."):
            # HARD truncate for speed
            text = f"{title}. {body}"[:600]

            result = analyze_news(text)

            if "error" in result:
                st.warning(result["error"])
            else:
                label = result["labels"][0]
                confidence = int(result["scores"][0] * 100)

                if label == "FAKE NEWS":
                    st.error(f"### 🚨 FAKE NEWS ({confidence}%)")
                else:
                    st.success(f"### ✅ REAL NEWS ({confidence}%)")
    else:
        st.warning("Please provide both headline and content.")

st.divider()
st.caption("⚠️ Detects linguistic patterns, not factual truth.")
