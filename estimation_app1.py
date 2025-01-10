import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile

# Constants for Pose Estimation
DEMO_IMAGE = 'images/stand.jpg'
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]
inWidth, inHeight = 368, 368

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Function for Pose Detection
@st.cache_data
def pose_detector(frame, threshold=0.2):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = int((frame_width * point[0]) / out.shape[3])
        y = int((frame_height * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame

# Sidebar for navigation buttons
st.sidebar.title("Pose Estimator")
page = st.sidebar.selectbox("", ["Menu","Home", "Pose Estimation", "About", "Help"])

if page == "Menu":
    st.title("Welcome to Human Pose Estimation")
    st.image("images/webImg.png", caption="Example Pose Estimation", use_container_width=True)
    st.markdown("""### What is Pose Estimation? Pose Estimation is a computer vision technique to identify human figures in an image or video by detecting key points on the body.""")
    st.markdown("Navigate to **Pose Estimation** in the menu to try it yourself!")

elif page == "Home":
    st.title("Welcome to Human Pose Estimation")
    st.image("images/stand.jpg", caption="Example Pose Estimation", use_container_width=True)
    st.markdown("""### What is Pose Estimation? Pose Estimation is a computer vision technique to identify human figures in an image or video by detecting key points on the body.""")
    st.markdown("Navigate to **Pose Estimation** in the menu to try it yourself!")

elif page == "Pose Estimation":
    st.title("Pose Estimation")
    
    # Sub-menu for different pose estimation methods (Dropdown for Pose Estimation)
    estimation_method = st.selectbox("Select Estimation Method:", ["Upload Image", "Upload Video", "Live Camera"])

    if estimation_method == "Upload Image":
        st.subheader("Upload Image")
        img_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
        if img_file:
            image = np.array(Image.open(img_file))
            st.image(image, caption="Original Image", use_container_width=True)

            # Pose Estimation
            threshold = st.slider("Threshold for Keypoint Detection", 0, 100, 20, 5) / 100
            output_image = pose_detector(image, threshold)
            st.image(output_image, caption="Pose Estimation", use_container_width=True)

    elif estimation_method == "Upload Video":
        st.subheader("Upload Video")
        video_file = st.file_uploader("Upload a Video:", type=["mp4", "avi", "mov"])
        if video_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(video_file.read())
            cap = cv2.VideoCapture(temp_file.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                output_frame = pose_detector(frame)
                stframe.image(output_frame, channels="BGR", use_container_width=True)

            cap.release()

    elif estimation_method == "Live Camera":
        st.subheader("Live Camera Feed")
        st.write("Live camera feed requires webcam access.")
        cap = cv2.VideoCapture(0)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_frame = pose_detector(frame)
            stframe.image(output_frame, channels="BGR", use_container_width=True)

        cap.release()

elif page == "About":
    st.title("About This Application")
    st.markdown("""
    This application demonstrates **Human Pose Estimation** using a pre-trained TensorFlow model.
    
    - **Key Features**:
      - Upload an image or video to estimate poses.
      - Use your live camera feed for real-time pose estimation.
    - **How It Works**:
      - Detects key body parts using a deep learning model.
      - Visualizes connections between body parts.

    This app is built using **Streamlit** and **OpenCV**.
    """)

elif page == "Help":
    st.title("Help & Troubleshooting")
    st.markdown("""
    - **How to use the app**:
      1. Select "Pose Estimation" from the menu.
      2. Choose an input type: image, video, or live camera.
      3. Follow the instructions to upload or enable your camera.

    - **Common Issues**:
      - If the live camera doesn't work, ensure you have granted webcam permissions.
      - Video uploads may take time depending on the file size.
    """)
