import streamlit as st
import numpy as np
import cv2
from PIL import Image
def image_to_sketch(image):
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    inverted_image = 255-gray_img
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray_img, inverted_blurred, scale=256.0)
    return pencil_sketch
def main():
    content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    if content_file is not None:
        content = Image.open(content_file)
        st.sidebar.image(content)
        content = np.array(content) #pil to cv
        content = image_to_sketch(content)
        st.image(content)
    else:
        st.warning("Please upload an Image")
        st.stop()
if __name__ == "__main__":
    main()
