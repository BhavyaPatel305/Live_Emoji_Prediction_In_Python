# Import Libraries
import mediapipe as mp
import numpy as np
import cv2

# Take video feed as the input
cap = cv2.VideoCapture(0)

# For each emoji, we are collecting 100 frames, so we give name to the data
name = input("Enter the name of the data: ")

# Using Mediapipe Library

# Takes in frames and returns facial keypoints like left hand, right hand, body etc.
holistic = mp.solutions.holistic

# To show visuals
hands = mp.solutions.hands

# Object of holistic class
holis = holistic.Holistic()

# To show visuals
drawing = mp.solutions.drawing_utils

# Numpy list to store the landmarks
# Collection of all the rows(lst)
X = []

# Trigger to stop the data collection
data_size = 0

while True:
    # Row: Will have 1020 columns of landmarks(face landmarks, left_hand landmarks, right_hand landmarks)
    lst = []
    
    # Read the feed
    _, frm = cap.read()
    
    # Flip the frame to avoid mirror effect(Flip left to right, so flip code 1)
    frm = cv2.flip(frm, 1)
    
    # Convert frame from cvtColor to RGB and pass it to the holistic object
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    
    # res can also be none, so better to add some if conditions(Sometimes, it can be none if it doesn't detect anything)
    if res.face_landmarks:
        # Iterate through the face landmarks: If there is a face in the frame
        for i in res.face_landmarks.landmark:
            # i has 2 properties: x position, y position
            # res.face_landmarks.landmark[1] is the reference value
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
            
        # If there is a left hand in the frame
        if res.left_hand_landmarks:
            # Iterate through the left hand landmarks
            for i in res.left_hand_landmarks.landmark:
                # Store left_hand landmarks in the list
                # res.left_hand_landmarks.landmark[8] is the reference value
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # If there is no left hand in the frame, just store 0.0 in place of those 42 points
            # res.left_hand_landmarks.landmark has size of 21
            # So 21*2 = 42 as for x and y
            for i in range(42):
                lst.append(0.0)
        
        # If there is a right hand in the frame
        if res.right_hand_landmarks:
            # Iterate through the right hand landmarks
            for i in res.right_hand_landmarks.landmark:
                # Store right_hand landmarks in the list
                # res.left_hand_landmarks.landmark[8] is the reference value
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # If there is no right hand in the frame, just store 0.0 in place of those 42 points
            # res.right_hand_landmarks.landmark has size of 21
            # So 21*2 = 42 as for x and y
            for i in range(42):
                lst.append(0.0)
        
        # If someone is in frame, than we have collected the data
        # Meaning, lst has some values in it
        # Append lst to array X
        X.append(lst)
        # Increment the data_size
        data_size += 1
            
        
    # Draw on the frame: face landmarks, left_hand landmarks, right_hand landmarks
    drawing.draw_landmarks(frm, res.face_landmarks)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
    # Informing user how much data is collected
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    
    # Show it to user
    cv2.imshow("window", frm)
    
    # If user press esc key, destroy all the windows, release the camera and break from the loop
    # If the value of data_size is greater than 99, than break from the loop because we have enough data
    if cv2.waitKey(1) == 27 or data_size > 99:
        # Destroy the windows
        cv2.destroyAllWindows()
        # Release the camera
        cap.release()
        # Break from the loop
        break
    
# Save the data
np.save(f"{name}.npy", np.array(X))