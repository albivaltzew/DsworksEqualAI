import cv2

# Open the video file
video_path = 'slovo_split/val/0a0f2267-abe1-4be0-a0f5-216c536dd1d7.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get video properties
frame_width = int(cap.get(3))  # Width of frames
frame_height = int(cap.get(4))  # Height of frames

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height), isColor=False)

# Process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # You can apply further processing to the fgmask if needed

    # Write the frame with the foreground mask to the output video
    out.write(fgmask)

    # Display the original frame and the foreground mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture, VideoWriter, and close OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()