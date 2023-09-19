from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
import cv2
import numpy as np
import streamlit as st
from PIL import Image


# Add logo image path
logo_path = "logo.png"  # Update with the path to your logo image

# Load logo image
logo_image = PIL.Image.open(logo_path)

# Custom CSS styles for the navbar and logo
css_styles = """
<style>
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px;
    background-color: #fff;
    width: 100%;
}
.logo {
    display: flex;
    align-items: center;
}

.logo img {
    height: 135px;
    margin-right: 10px;
}

.navbar-title {
    font-size: 35px;
    font-weight: bold;
}

</style>
"""

# Render the custom CSS styles
st.markdown(css_styles, unsafe_allow_html=True)

# Render the navbar
st.markdown('<div class="navbar">'
            '<div class="navbar-title">CokeDetectionAI using YOLOv8</div>'
            '<div class="logo">'
            '<img src="data:image/png;base64,{}" alt="Logo">'
            '</div>'
            '<br>'
          
            '</div>'.format(helper.image_to_base64(logo_image)), unsafe_allow_html=True)

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

st.sidebar.header("MODEL CONFIGURATION")


# slider for the confidence level of the pretrained model over here
pretrained_confidence = float(st.sidebar.slider("Pretrained Model Confidence", 0, 100, 40)) / 100


# Slider for the confidence level of the custom model
custom_confidence = float(st.sidebar.slider("Custom Model Confidence", 0, 100, 40)) / 100

if pretrained_confidence  < 0 or pretrained_confidence  > 1:
    st.error("Please select a valid confidence level between 0 and 100 for the pretrained model.")

model_type = st.sidebar.radio("Select Task", ['Detection',"Can Detection"])

if model_type == 'Detection':
    pretrained_model_path = Path(settings.DETECTION_MODEL)
    custom_model_path = Path(settings.CUSTOM_MODEL)

    # detection_type = st.sidebar.radio("Select Detection Type", ['Default', 'Custom'])
    # if detection_type == 'Default':
    #     model_path = Path(settings.DETECTION_MODEL)
    #     show_input_image = True
    # elif detection_type == 'Custom':
    #     model_path = Path(settings.CUSTOM_MODEL)  # Update with the path to the custom model
    #     show_input_image = False  # Set the flag to False to not display the input image


elif model_type == 'Can Detection':
    # detection_type = st.sidebar.radio("Select Detection Type", ['Default', 'Custom'])
    pretrained_model_path = Path(settings.CanDetectionModel)
    custom_model_path = Path(settings.CUSTOM_MODEL)

    # if detection_type == 'Default':
    #     model_path = Path(settings.CanDetectionModel)
    #     show_input_image = True
    # elif detection_type == 'Custom':
    #     model_path = Path(settings.CUSTOM_MODEL)  # Update with the path to the custom model
    #     show_input_image = False  # Set the flag to False to not display the input image

try:
    pretrained_model = helper.load_model(pretrained_model_path)
    custom_model = helper.load_model(custom_model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {pretrained_model} and {custom_model_path}")
    st.error(ex)



st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_img = None
totalNumberOfBottles = 0  # Initialize the global variable
customNumberOfBottles = 0  # Initialize the variable for custom model

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2, col3 = st.columns(3)  # Create three columns for displaying images

    with col1:
        try:
            if source_img is not None:

                is_blurred, blur_variance = is_image_blurred(source_img)
                if is_blurred:
                    st.error("The uploaded image is blurred. Please upload a clear image.")
                else:
                    uploaded_image = PIL.Image.open(source_img)

                    if uploaded_image is not None:
                        # Display the uploaded image
                        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    if source_img is not None:
        if st.sidebar.button('Detect Objects'):
            res = pretrained_model.predict(uploaded_image, conf=pretrained_confidence )
            num_detected = 0  # Initialize the counter for detected elements

            for i, result in enumerate(res):
                boxes = result.boxes
                class_names = result.names

                # Count the number of detected elements
                num_detected += len(boxes)

                # settting the confidence level of the unessential classes over here
                # Set confidence level of "Person" class to 0
                for j in range(len(class_names)):
                    if class_names[j] == "Person":
                        result.scores[j] = 0.0
  
                res_plotted = result.plot()[:, :, ::-1]
                if i == 0:
                    with col2:
                        st.image(res_plotted, caption=f'Detected Image {i+1}', use_column_width=True)
                        try:
                            with st.expander(f"Detection Results {i+1}"):
                                for box, name in zip(boxes, class_names):
                                    st.write(f"Class: {name}, Box: {box.data}")
                        except Exception as ex:
                            st.write(f"No objects detected in the image {i+1}!")
                elif i == 1:
                    with col3:
                        st.image(res_plotted, caption=f'Detected Image {i+1}', use_column_width=True)
                        try:
                            with st.expander(f"Detection Results {i+1}"):
                                for box, name in zip(boxes, class_names):
                                    st.write(f"Class: {name}, Box: {box.data}")
                        except Exception as ex:
                            st.write(f"No objects detected in the image {i+1}!")

            # Update the global variable
            totalNumberOfBottles = num_detected

            # Display the total count of detected elements
            st.write(f"Total number of detected elements (Detection): {num_detected}")


            custom_model_path = Path(settings.CUSTOM_MODEL)  # Update with the path to the custom model


            if custom_confidence < 0 or custom_confidence > 1:
                st.error("Please select a valid confidence level between 0 and 100 for the custom model.")

            try:
                custom_model = helper.load_model(custom_model_path)
            except Exception as ex:
                st.error(f"Unable to load custom model. Check the specified path: {custom_model_path}")
                st.error(ex)

            if source_radio == settings.IMAGE and source_img is not None:
                res_custom = custom_model.predict(uploaded_image, conf=custom_confidence)
                custom_num_detected = 0  # Initialize the counter for custom model detected elements

                col1, col2 = st.columns(2)  # Create two columns for displaying images

                with col1:
                    st.image(uploaded_image, caption="Input Image", use_column_width=True)

                with col2:
                    for i, result in enumerate(res_custom):
                        boxes = result.boxes
                        class_names = result.names

                        # Count the number of custom model detected elements
                        custom_num_detected += len(boxes)

                        res_plotted = result.plot()[:, :, ::-1]
                        st.image(res_plotted, caption=f'Detected Image (Custom) {i+1}', use_column_width=True)
                        try:
                            with st.expander(f"Detection Results (Custom) {i+1}"):
                                for box, name in zip(boxes, class_names):
                                    st.write(f"Class: {name}, Box: {box.data}")
                        except Exception as ex:
                            st.write(f"No objects detected in the image {i+1} for the custom model!")

                # Calculate the difference between the number of detections
                customNumberOfBottles = custom_num_detected
                difference = totalNumberOfBottles - customNumberOfBottles

                # Display the total count of detected elements for the custom model
                st.write(f"Total number of detected elements (Custom): {custom_num_detected}")

                # Display the difference in detections
                st.write(f"Difference between Detection and Custom: {difference}")

                if difference>0:
                    st.write("Inventory is Polluted")
                else:
                    st.write("Inventory is not Polluted")

                




        


# elif source_radio == settings.VIDEO:
#     helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    st.sidebar.header("Real-time Object Detection")

    # Create a VideoCapture object to capture frames from the webcam
    cap = cv2.VideoCapture(0)  # Use the default webcam (usually index 0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open the webcam.")
    else:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                st.error("Error: Could not read a frame from the webcam.")
                break

            # Perform object detection on the frame (similar to how you did for the image)
            # You can use either the pretrained_model or custom_model here
            # For example, using the pretrained model:
            res = pretrained_model.predict(frame, conf=pretrained_confidence)

            # Display the frame with detected objects
            # You can use the same code for displaying and processing the detections
            # as you did for the image
            for i, result in enumerate(res):
                boxes = result.boxes
                class_names = result.names

                # Display the detected objects on the frame
                for box, name in zip(boxes, class_names):
                    left, top, right, bottom = box
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(frame, name, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the frame with detected objects
                st.image(frame, caption=f'Detected Frame {i+1}', channels='BGR', use_column_width=True)

            # Display the total count of detected elements (you can update this accordingly)
            st.write(f"Total number of detected elements: {len(boxes)}")

            # Check for user input to exit the real-time detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the VideoCapture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")