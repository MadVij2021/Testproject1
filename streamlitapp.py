import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image

MODEL_PATH = "pose_landmarker_heavy.task"

# Drawing landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Streamlit UI
st.title("Real-Time Pose Tracking with MediaPipe")
st.write("Using MediaPipe and Streamlit to track poses in real-time.")

# Set up webcam
# run = st.checkbox('Start Webcam')

# PoseLandmarker options and model
base_options = python.BaseOptions(delegate=0,model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # # Pose detection
        detection_result = detector.detect(mp_image)

        # Draw landmarks
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)


        # Display the frame in Streamlit

        # Flip and process image for MediaPipe
    
        return annotated_image


webrtc_streamer(key="example", video_processor_factory=VideoTransformer)






# # STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)

# # # STEP 3: Load the input image.
# # image = mp.Image.create_from_file("image.jpg")


# enable = st.checkbox("Hows it?")
# image = st.camera_input("Take a picture", disabled=not enable)



# # # STEP 4: Detect pose landmarks from the input image.


# # # STEP 5: Process the detection result. In this case, visualize it.

# # # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# if image:
#     st.image(image)


# st.write("Hello World!!!, Lets say whether the model will detect landmarks or not!")