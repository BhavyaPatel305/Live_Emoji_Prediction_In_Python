# Import Libraries
import mediapipe as mp
import numpy as np
import cv2

# Take video feed as the input
cap = cv2.VideoCapture(0)

# Using Mediapipe Library

# Takes in frames and returns facial keypoints like left hand, right hand, body etc.
holistic = mp.solutions.holistic

# To show visuals
hands = mp.solutions.hands

# Object of holistic class
holis = holistic.Holistic()

# To show visuals
drawing = mp.solutions.drawing_utils

while True:
    # Read the feed
    _, frm = cap.read()
    
    # Flip the frame to avoid mirror effect(Flip left to right, so flip code 1)
    frm = cv2.flip(frm, 1)
    
    # Convert frame from cvtColor to RGB and pass it to the holistic object
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    
    # Draw on the frame: face landmarks, left_hand landmarks, right_hand landmarks
    drawing.draw_landmarks(frm, res.face_landmarks)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
    
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
    
