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
# 1️⃣ Load Model
# -------------------------------
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

WEIGHTS_PATH = "deepfake_img_weights.weights.h5"
model = build_model()
model.load_weights(WEIGHTS_PATH)

# -------------------------------
# 2️⃣ Streamlit Layout
# -------------------------------
st.title("Deepfake Detection App 🤖")
st.write("Upload an image and the model will predict whether it is Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence*100:.2f}%**")
