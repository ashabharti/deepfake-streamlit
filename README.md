# 🕵️ Deepfake Detection App (Streamlit + EfficientNetB0)

This project is a **Deepfake Detection Web App** built with **Streamlit** and **TensorFlow/Keras (EfficientNetB0)**.  
The app allows you to upload images and predicts whether they are **Real** or **Fake** using a pre-trained deep learning model.  

---

## 🚀 Features
- Upload one or multiple images (`.jpg`, `.jpeg`, `.png`)
- Predict whether the image is **Real** or **Fake**
- Adjustable **decision threshold**
- Display **confidence score**
- Export results as `.csv`
- Automatic download of model weights from Google Drive

---

## 📂 Project Structure
deepfake-streamlit/
│── app.py                        # Main Streamlit application
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
│── deepfake_img_weights.weights.h5   # (Downloaded automatically from Google Drive)

---

## 🏋️ Model Weights

The trained model weights are stored on **Google Drive** and are automatically downloaded at runtime.  

- **Google Drive File ID**:  1ksJDgcgHN4T12rsGKwoW3FrmqffDyAJ8 
- At runtime, the app will use `gdown` to download the weights and save them locally as:  deepfake_img_weights.weights.h5

  

