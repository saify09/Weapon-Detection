import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO


# --- Load YOLOv11 model ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model


model = load_model()

# --- Page config ---
st.set_page_config(page_title="Pistol And Knife Detection", layout="wide")

# --- Sidebar ---
st.sidebar.header("🧠 About")
st.sidebar.write(
    """
This application performs **AI-powered Weapon analysis**  
for weapon detection using **YOLO11n**.
"""
)

st.sidebar.markdown("### Weapon Types")
st.sidebar.markdown("- Pistol\n- Knife")

st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ **Model Info**")
st.sidebar.metric("Model Type", "YOLO11n")
st.sidebar.metric("Device", "GPU / CPU")
st.sidebar.markdown("---")

# --- Main UI ---
st.title("🧠 Weapon Detection")
st.write("Upload an image to analyze using the trained YOLO11n model.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with col1:
        st.image(image_np, caption="Original Image", use_container_width=True)

    # --- Run detection ---
    start = time.time()
    results = model.predict(image_np, conf=0.25)
    end = time.time()

    # --- Annotate result ---
    annotated_frame = results[0].plot()  # YOLO auto draws boxes & labels

    with col2:
        st.image(annotated_frame, caption="Detection Results", use_container_width=True)

    # --- Display Metrics ---
    avg_conf = float(np.mean([box.conf.cpu() for box in results[0].boxes]))
    st.sidebar.metric("Model Confidence", f"Avg: {avg_conf*100:.1f}%")
    st.sidebar.metric("Inference Time", f"~ {(end-start)*1000:.1f} ms")

    st.success("✅ Analysis complete.")
else:
    st.info("Please upload an image to start analysis.")
