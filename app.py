#to run the code
#  https://share.streamlit.io/?

# app.py
import os
import streamlit as st
import numpy as np
import gdown
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Google Drive weights file
# -------------------------
FILE_ID = "1OFeTCG9E0zoczIuefTSbJJgr6kxd5Odo"  # 🔹 Replace with your new weights file ID
WEIGHTS_PATH = "deepfake_weights.weights.h5"  # 🔹 Keras 3+ requirement

# Download weights if not exists
if not os.path.exists(WEIGHTS_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, WEIGHTS_PATH, quiet=False)

# -------------------------
# Build model architecture & load weights
# -------------------------
def build_model_for_inference():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(WEIGHTS_PATH)
    return model

@st.cache_resource
def load_deepfake_model():
    return build_model_for_inference()

# -------------------------
# Preprocess uploaded images
# -------------------------
def preprocess_img(file):
    img = image.load_img(file, target_size=(224, 224), color_mode="rgb")
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # EfficientNet preprocessing
    return arr

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="centered")
st.title("🕵️ Deepfake Detection (EfficientNetB0 - RGB)")

# Load model
try:
    model = load_deepfake_model()
    st.success("✅ Model loaded successfully (RGB).")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "📂 Upload Image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True
)
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_files:
    results = []
    for uploaded in uploaded_files:
        st.image(uploaded, caption=f"Uploaded: {uploaded.name}", use_column_width=True)
        arr = preprocess_img(uploaded)
        score = float(model.predict(arr)[0][0])

        if score >= threshold:
            label = "🔴 Fake"
            confidence = score
        else:
            label = "🟢 Real"
            confidence = 1 - score

        results.append({
            "Filename": uploaded.name,
            "Result": label,
            "Raw Score": round(score, 4),
            "Confidence": round(confidence, 4)
        })

    df = pd.DataFrame(results)
    st.markdown("### 📊 Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results", data=csv, file_name="deepfake_results.csv", mime="text/csv"
    )
