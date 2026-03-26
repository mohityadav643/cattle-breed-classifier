import streamlit as st
from PIL import Image
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄")

# -------------------------------
# Title
# -------------------------------
st.title("🐄 Cattle Breed Classifier")
st.write("Upload an image of a cattle to classify its breed")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# If image uploaded
# -------------------------------
if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Dummy preprocessing (for future model)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0

        # -------------------------------
        # DEMO Prediction (TEMP)
        # -------------------------------
        st.subheader("🔍 Prediction Result")

        st.success("✅ Model is working (Demo Mode)")

        st.write("👉 Example Prediction: **Gir**")
        st.write("👉 Confidence: **92.3%**")

        st.info("⚠️ Note: This is demo mode. Model will be added next.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("Made by Mohit 🚀")