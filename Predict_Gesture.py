#--------------------------------Importing Libraries---------------------------------------------#
import cv2
import mediapipe as mp
import pickle
import numpy as np
#------------------------------------------------------------------------------------------------#

#--------------------------------Defining Parameters---------------------------------------------#
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_drawing=mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
#------------------------------------------------------------------------------------------------#

with open("Alphabets_Model-3.pkl", "rb") as f:
    model=pickle.load(f)             #Loading the Model

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Drawing Skeletal Lines
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            #Extracting Landmarks
            data=[]
            wrist=hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                data.extend([lm.x-wrist.x,lm.y-wrist.y,lm.z-wrist.z])

            #Reshaping "data" list as a 2D Array
            data_array=np.array(data)
            data_reshaped=data_array.reshape(1,-1)

            #Model's Prediction
            prediction=model.predict(data_reshaped)
            cv2.putText(frame,f"Prediction:{prediction[0]}",(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

