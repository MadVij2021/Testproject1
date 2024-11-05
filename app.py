import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.title("Webcam App with Streamlit WebRTC")

# RTC Configuration for the video stream
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Function to process each video frame
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # (Optional) Process the frame (e.g., apply filters)
    
    # Return the processed frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer component
webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)
