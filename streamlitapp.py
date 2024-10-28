import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2



MODEL_PATH="D:\SIH 2024\pose_landmarker_heavy.task"
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
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
st.title("Real-Time Hand Tracking with MediaPipe")
st.write("Using MediaPipe and Streamlit to track hands in real-time.")

# Set up webcam
run = st.checkbox('Start Webcam')



base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
if run:
    # OpenCV video capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process image for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(rgb_frame)
        annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)

        # Display output in Streamlit

        if cv2.waitKey(1)==ord('q'):
           break
   
        if annotated_image:
           st.image(annotated_image)

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Click the checkbox to start the webcam.")






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