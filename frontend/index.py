import streamlit as st
import requests
import os

st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 30px;
        }
        .stSlider > div {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
        }
        .predict-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .output-box {
            background-color: #e6f7ff;
            border-left: 6px solid #2196F3;
            padding: 10px 20px;
            margin-top: 20px;
            border-radius: 5px;
            text-transform: capitalize;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧬 Breast Cancer Prediction App</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    clump_thickness = st.slider("Clump Thickness", 1, 10, 5)
    cell_size_uniformity = st.slider("Cell Size Uniformity", 1, 10, 5)
    cell_shape_uniformity = st.slider("Cell Shape Uniformity", 1, 10, 5)
    marginal_adhesion = st.slider("Marginal Adhesion", 1, 10, 5)
    single_epi_cell_size = st.slider("Single Epi Cell Size", 1, 10, 5)

with col2:
    bare_nuclei = st.slider("Bare Nuclei", 1, 10, 5)
    bland_chromatin = st.slider("Bland Chromatin", 1, 10, 5)
    normal_nucleoli = st.slider("Normal Nucleoli", 1, 10, 5)
    mitoses = st.slider("Mitoses", 1, 10, 5)

input_data = {
    "features": [
        clump_thickness,
        cell_size_uniformity,
        cell_shape_uniformity,
        marginal_adhesion,
        single_epi_cell_size,
        bare_nuclei,
        bland_chromatin,
        normal_nucleoli,
        mitoses
    ]
}

st.markdown("### 🔍 Model Input")
st.json(input_data)

DEFAULT_BACKEND_URL="http://0.0.0.0:8080"
BACKEND_URL=os.environ.get("BACKEND_URL",DEFAULT_BACKEND_URL)

if st.button("🚀 Predict"):
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=input_data)
        prediction = response.json()
        predicted_class = prediction.get("class", "Unknown")
        st.markdown(f'<div class="output-box">🔮 <strong>Prediction Outcome:</strong> {predicted_class}</div>', unsafe_allow_html=True)

    except:
        st.error("❌ Could not connect to the prediction server. Check your FastAPI backend.")
