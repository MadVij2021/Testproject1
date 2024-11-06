import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import av
from sample_utils.turn import get_ice_servers
import cv2

st.title("KAISA APP HAI YEH BSDK")

# RTC Configuration for the video stream
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Function to process each video frame

class VideoTransforms(VideoTransformerBase):
    def transform(self,frame):
        img = frame.to_ndarray(format="bgr24")
        # (Optional) Process the frame (e.g., apply filters)
        img=cv2.flip(img,1)
        # Return the processed frame
        return img

# WebRTC streamer component
webrtc_streamer(key="example",video_transformer_factory=VideoTransforms)
