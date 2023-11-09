import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Function to check if an image is blurred
def is_image_blurred(uploaded_image, threshold=550):
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

# Streamlit UI
st.title("Image Blur Detection")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    is_blurred, variance = is_image_blurred(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if is_blurred:
        st.write(f"The uploaded image is blurred with a variance of {variance:.2f}.")
    else:
        st.write(f"The uploaded image is not blurred with a variance of {variance:.2f}.")

st.write("Note: Adjust the threshold value in the code as needed for your use case.")
