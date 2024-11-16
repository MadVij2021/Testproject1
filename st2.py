from streamlit_webrtc import webrtc_streamer
import av
import cv2


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img=cv2.circle(img,(90,90),80,(255,0,0),9)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
