import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Satellite Image Classification",
    page_icon="🛰️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_mobilenetv2.keras")

@st.cache_data
def load_classes():
    with open("data/processed/classes.json") as file:
        return json.load(file)

model = load_model()
classes = load_classes()

st.title("🛰️ Satellite Imagery Classification")
st.write("Deep Learning-Based Terrain and Land Cover Analysis")

st.info("Upload a satellite image to identify its land-cover category.")

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    image_resized = image.resize((64, 64))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    with st.spinner("Classifying satellite image..."):
        predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    with col2:
        st.subheader("Classification Result")
        st.success(f"Predicted Category: {classes[predicted_index]}")
        st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("Class-wise Prediction Confidence")

    prediction_data = {
        classes[i]: float(predictions[i]) * 100
        for i in range(len(classes))
    }

    st.bar_chart(prediction_data)

    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(predictions)[-3:][::-1]

    for index in top_indices:
        st.write(
            f"**{classes[index]}:** {predictions[index] * 100:.2f}%"
        )

st.divider()

st.caption(
    "Dataset: EuroSAT RGB | Model: MobileNetV2 | "
    "Purpose: Environmental and land-cover classification"
)