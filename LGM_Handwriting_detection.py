import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

model = keras.models.load_model(r'C:\Users\ASUS\LGM_character\Handwriting_model (2).h5')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 35, 32)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=300,
    drawing_mode=drawing_mode,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    image = canvas_result.image_data
    image1 = image.copy()
    image1 = image1.astype('uint8')
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1,(28,28))
    st.image(image1)

    image1.resize(1,28,28,1)
    st.title(np.argmax(model.predict(image1)))
if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))