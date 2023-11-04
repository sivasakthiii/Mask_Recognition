import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Unmasker")

st.write("Predict face is masked or not.")

model = load_model("model.h5")
labels = ['WithMask','WithoutMask']
uploaded_file = st.file_uploader(
    "Upload an image of a person:", type="jpg"
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(100,100))
    image1=classi=np.array(image1)/255.
    result=model.predict(image1[np.newaxis,...])
    label=labels[np.argmax(result)]
    print(result)

st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file.")








