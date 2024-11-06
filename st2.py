import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Convert to Grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian Blur
        blurred_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
        
        # 3. Edge Detection using Canny
        edges = cv2.Canny(blurred_img, 100, 200)
        
        # Convert edges to BGR to allow for colored overlay
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 4. Overlay Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(edges_colored, "Live Feed", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # 5. Draw Shapes (e.g., Rectangle)
        # Draw a rectangle at a fixed position
        cv2.rectangle(edges_colored, (50, 50), (200, 200), (0, 255, 0), 3)
        
        # 6. Draw Circle
        cv2.circle(edges_colored, (300, 300), 50, (255, 0, 255), -1)
        
        # Return the processed frame
        return edges_colored



webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


