#to run the code
#  https://share.streamlit.io/?

# app.py
import os
import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Google Drive model details
# -------------------------
FILE_ID = "1StQVRwoRKxHWR82uwWNw6Msj88zTvoPq"  # Your .h5 file ID
MODEL_PATH = "deepfake_model.h5"

# Download model from Google Drive if not exists
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# Load the trained model
# -------------------------
@st.cache_resource
def load_deepfake_model():
    return load_model(MODEL_PATH)  # Directly load .h5 trained on grayscale

# -------------------------
# Preprocess uploaded images
# -------------------------
def preprocess_img(file):
    # Load grayscale image (1 channel)
    img = image.load_img(file, target_size=(224, 224), color_mode="grayscale")
    arr = image.img_to_array(img)  # Shape: (224,224,1)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # Keep as-is
    return arr

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="centered")
st.title("🕵️ Deepfake Detection using EfficientNetB0 (Grayscale)")

# Load model
try:
    model = load_deepfake_model()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# File uploader
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

