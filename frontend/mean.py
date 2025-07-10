# ML engineer
import streamlit as st
import requests

st.title("Breast Cancer Prediction")

clump_thickness = st.slider("Clump Thickness", 1 , 10)
cell_size_uniformity = st.slider("Cell Size Uniformity", 1 , 10)
cell_shape_uniformity = st.slider("Cell Shape Uniformity", 1 , 10)
marginal_adhesion = st.slider("Marginal Adhesion", 1 , 10)
single_epi_cell_size = st.slider("Single Epi Cell Size", 1 , 10)
bare_nuclei = st.slider("Bare Nuclei", 1 , 10)
bland_chromatin = st.slider("Bland Chromatin", 1 , 10)
normal_nucleoli = st.slider("Normal Nucleoli", 1 , 10)
mitoses = st.slider("Mitoses", 1 , 10)

# string from frontend to int
clump_thickness = int(clump_thickness)
cell_size_uniformity = int(cell_size_uniformity)
cell_shape_uniformity = int(cell_shape_uniformity)
marginal_adhesion = int(marginal_adhesion)
single_epi_cell_size = int(single_epi_cell_size)
bare_nuclei = int(bare_nuclei)
bland_chromatin = int(bland_chromatin)
normal_nucleoli = int(normal_nucleoli)
mitoses = int(mitoses)

input_data = {
  "features": [clump_thickness, 
              cell_size_uniformity,
              cell_shape_uniformity,
              marginal_adhesion, 
              single_epi_cell_size, 
              bare_nuclei, 
              bland_chromatin, 
              normal_nucleoli, 
              mitoses]
}

st.write(input_data)
response = requests.post("http://192.168.0.103:8000/predict", 
                        json=input_data)

# prediction from backend
prediction = response.json()

st.write("Prediction Outcome: ", prediction)




# streamlit run index.py