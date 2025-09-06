# app.py
# -------------------------------
# Streamlit Deepfake Detection App
# -------------------------------
import streamlit as st
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# 1️⃣ Build Model (same as training)
# -------------------------------
def build_model():
    # ⚠️ Use weights=None so it matches your fine-tuned model weights
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Path to weights
WEIGHTS_PATH = "deepfake_img_weights.weights.h5"

# Load model + weights
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# -------------------------------
# 2️⃣ Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Deepfake Detection App")
st.write("Upload an image and the model will predict whether it is **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    # Show result
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Class:** {label}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    # Progress bar
    st.progress(float(confidence))

    # Debug raw score
    st.caption(f"Raw model output: {pred:.4f}")

else:
    st.info("👆 Please upload an image to start prediction.")
