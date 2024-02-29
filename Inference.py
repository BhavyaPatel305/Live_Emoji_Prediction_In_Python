# Importing the required libraries
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load the model
model = load_model("model.h5")

# Load the labels
label = np.load("labels.npy")

# Same code as Data_Collection.py

# Takes in frames and returns facial keypoints like left hand, right hand, body etc.
holistic = mp.solutions.holistic

# To show visuals
hands = mp.solutions.hands

# Object of holistic class
holis = holistic.Holistic()

# To show visuals
drawing = mp.solutions.drawing_utils

# Capture the video feed from webcam
cap = cv2.VideoCapture(0)

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
        
        # Converting lst to array and re-shape into 2-D array
        lst = np.array(lst).reshape(1,-1)
        
        # Predictions
        # pred will be max index with the prediction, and convert it to a label
        pred = label[np.argmax(model.predict(lst))]
        
        # Print the prediction on the frame
        cv2.putText(frm, pred, (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
        
        
        
    # Draw on the frame: face landmarks, left_hand landmarks, right_hand landmarks
    drawing.draw_landmarks(frm, res.face_landmarks)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
    
    
    
    # Show it to user
    cv2.imshow("window", frm)
    
    # If user press esc key, destroy all the windows, release the camera and break from the loop
    # If the value of data_size is greater than 99, than break from the loop because we have enough data
    if cv2.waitKey(1) == 27:
        # Destroy the windows
        cv2.destroyAllWindows()
        # Release the camera
        cap.release()
        # Break from the loop
        break