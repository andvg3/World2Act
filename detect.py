import cv2
import numpy as np

def get_crop_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read video.")
        return

    # Convert to gray and threshold to find the "real" content
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY) # Higher threshold to ignore noise
    coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        print("-" * 30)
        print(f"DETECTION RESULTS:")
        print(f"X (Left edge): {x}")
        print(f"Y (Top edge):  {y}")
        print(f"Width:         {w}")
        print(f"Height:        {h}")
        print("-" * 30)
        return x, y, w, h
    else:
        print("No content detected!")

# Run this once to see your variables
get_crop_coordinates('output/generated_video_robocasa_model_retrained_4views_PnPpickhotdog_0_8000.mp4')