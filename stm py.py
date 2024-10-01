import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image", type="tif")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)
    
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)
    
    pred_mask = np.array(response.json()['segmentation'])
    st.image(pred_mask, caption="Predicted Segmentation Mask", use_column_width=True)
