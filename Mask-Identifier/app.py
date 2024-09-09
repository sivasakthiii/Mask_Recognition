import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Unmasker")

st.write("Predict if a face is masked or not.")

# Load the model with error handling
try:
    model = load_model("Mask-Identifier/model.h5")
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

labels = ['WithMask', 'WithoutMask']

# Upload the image
uploaded_file = st.file_uploader("Upload an image of a person:", type="jpg")
label = None

if uploaded_file is not None:
    # Open and preprocess the image
    image1 = Image.open(uploaded_file)
    image1 = image1.resize((100, 100))  # Resize the image to 100x100
    image1 = np.array(image1) / 255.0  # Normalize the image
    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    
    # Predict
    result = model.predict(image1)
    label = labels[np.argmax(result)]
    
    # Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"<h2 style='text-align: center;'>Image of {label}</h2>", unsafe_allow_html=True)
else:
    st.write("Please upload a file.")
