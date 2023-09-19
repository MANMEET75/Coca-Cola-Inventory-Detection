import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to check if an image is blurred
def is_image_blurred(uploaded_image, threshold=500):
    # Convert BytesIO object to an image
    pil_image = Image.open(uploaded_image)
    
    # Convert to a NumPy array
    image = np.array(pil_image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Determine if the image is blurred
    is_blurred = variance < threshold

    return is_blurred, variance

# Streamlit code to upload an image and check for blurriness
st.title("Image Blurriness Detector")

# Upload an image using Streamlit
uploaded_image = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png"))

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Check if the image is blurred
    is_blurred, variance = is_image_blurred(uploaded_image)

    # Display the result
    if is_blurred:
        st.write(f"The image is blurred. Variance of Laplacian: {variance:.2f}")
    else:
        st.write(f"The image is not blurred. Variance of Laplacian: {variance:.2f}")
