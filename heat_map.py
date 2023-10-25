import cv2
import numpy as np

# Open the video file
video_path = 'slovo_split/val/0a0c3220-ae95-4919-a1df-3f7e538067cb.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
frames = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an empty motion heatmap with the same dimensions as the video frames
motion_heatmap = np.zeros((height, width, 3), dtype=np.uint8)

# Define a threshold for motion detection (adjust as needed)
motion_threshold = 0.5

# Process video frames
for i in range(frame_count - 1):
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        break

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create a mask for areas with motion
    motion_mask = (magnitude > motion_threshold).astype(np.uint8)

    # Map the magnitude to a color using a colormap (e.g., jet)
    color_map = cv2.applyColorMap(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

    # Update the motion heatmap where there is motion
    motion_heatmap = cv2.addWeighted(motion_heatmap, 1.0, color_map, 0.8, 0)
    motion_heatmap = cv2.bitwise_and(motion_heatmap, motion_heatmap, mask=motion_mask)

# Save the motion heatmap where no motion is black and motion is gradient
cv2.imwrite('motion_heatmap_with_black_background.jpg', motion_heatmap)

# Release the video capture object
cap.release()