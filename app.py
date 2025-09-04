import os
import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ✅ Google Drive Model File ID (from your link)
FILE_ID = "1rCujfPMTMPgpSJdrAdVG4UGCQaCsvhZD"
MODEL_PATH = "deepfake_model.h5"

def ensure_model():
    """Download model from Google Drive if not already present"""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_deepfake_model():
    ensure_model()
    return load_model(MODEL_PATH)

def preprocess_img(file):
    """Preprocess uploaded image for EfficientNetB0"""
    img = load_img(file, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# ✅ Streamlit UI
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="centered")
st.title("🕵️ Deepfake Detection using EfficientNetB0")

try:
    model = load_deepfake_model()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("📂 Upload an Image", type=["jpg","jpeg","png"])
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    arr = preprocess_img(uploaded)
    score = float(model.predict(arr)[0][0])

    if score >= threshold:
        label = "🔴 Fake"
        confidence = score
    else:
        label = "🟢 Real"
        confidence = 1 - score

    st.markdown(f"### Result: **{label}**")
    st.write(f"Raw score: `{score:.4f}`")
    st.write(f"Confidence: `{confidence:.4f}`")


