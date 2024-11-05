import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from sample_utils.turn import get_ice_servers
import cv2

st.title("Webcam App with Streamlit WebRTC")

# RTC Configuration for the video stream
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Function to process each video frame
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # (Optional) Process the frame (e.g., apply filters)
    img=cv2.flip(img,0)
    # Return the processed frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer component
webrtc_streamer(
    key="programatic_control",
    rtc_configuration={  # Add this line
        "iceServers": get_ice_servers()
    }
)
