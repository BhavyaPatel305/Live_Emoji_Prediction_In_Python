# Import Libraries
import mediapipe as mp
import numpy as np
import cv2

# Take video feed as the input
cap = cv2.VideoCapture(0)

while True:
    # Read the feed
    _, frm = cap.read()
    
    # Show it to user
    cv2.imshow("window", frm)
    
    # If user press esc key, destroy all the windows, release the camera and break from the loop
    if cv2.waitKey(1) == 27:
        # Destroy the windows
        cv2.destroyAllWindows()
        # Release the camera
        cap.release()
        # Break from the loop
        break
    
