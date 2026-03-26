import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Cattle Classifier", layout="centered")

st.title("🐄 Cattle Breed Classifier")
st.write("Upload an image to detect cattle breed")

# ======================
# DEBUG START
# ======================
st.write("🚀 App started...")

# Check model file
if not os.path.exists("model.h5"):
    st.error("❌ model.h5 NOT FOUND")
    st.stop()
else:
    st.success("✅ model.h5 found")

# ======================
# LOAD MODEL (SAFE)
# ======================
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

try:
    model = load_my_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# ======================
# CLASS NAMES
# ======================
class_names = sorted([
    'amritmahal','ayrshire','bargur','dangi','deoni','gir',
    'hallikar','kangayam','kankrej','malvi','nagori',
    'ongole','rathi','red_sindhi','sahiwal'
])

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("📤 Upload cattle image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)[0]

        # Top 3
        top3_idx = preds.argsort()[-3:][::-1]

        st.subheader("🔍 Prediction Results")

        for i in top3_idx:
            st.write(f"👉 {class_names[i]} — {preds[i]*100:.2f}%")

        best = top3_idx[0]
        st.success(f"🏆 Final Prediction: {class_names[best]}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")