import cv2
import mediapipe as mp
import csv
import argparse
import os

#------------------------------------------Control Panel-----------------------------------------#
parser=argparse.ArgumentParser()
parser.add_argument("--label",type=str,required=True,help="Name of the Label")
parser.add_argument("--sample",type=int,default=10,help="Sample Number")
args=parser.parse_args()
#------------------------------------------------------------------------------------------------#

#----------------------------------------Defining Parameters-------------------------------------#
cap = cv2.VideoCapture(0)
output_file='Alphabets-3.csv'
count=0
#------------------------------------------------------------------------------------------------#

#----------------------------------Loading 2 Classes in containers-------------------------------#
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_drawing=mp.solutions.drawing_utils
#------------------------------------------------------------------------------------------------#

# This creates a list of strings: ['label', 'x1', 'y1', 'z1', ..., 'x21', 'y21', 'z21']
header = ['label']
for i in range(1, 22):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])
file_exists=os.path.isfile(output_file)


with open(output_file,mode= "a",newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    while count<args.sample:
        ret,frame=cap.read()
        if not ret:
            print("No image")
            break
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                data=[args.label]
                wrist=hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x-wrist.x,lm.y-wrist.y,lm.z-wrist.z])

                writer.writerow(data)
                count+=1
                cv2.putText(frame, f"Sample {count}/{args.sample}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Tracking Test', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Finished adding {args.sample} sample for '{args.label}' to {output_file}")

