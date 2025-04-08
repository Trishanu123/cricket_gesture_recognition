import cv2
import mediapipe as mp
import numpy as np
import csv
import os

data_file = 'umpire_gestures.csv'
if not os.path.exists(data_file):
    with open(data_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'landmark_{i}_x' for i in range(33)] + 
                        [f'landmark_{i}_y' for i in range(33)] + 
                        [f'landmark_{i}_z' for i in range(33)] + ['gesture_label'])

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

valid_gestures = ['six', 'four', 'no_ball']
gesture_counts = {gesture: 0 for gesture in valid_gestures}
max_samples = 500

gesture_label = input("Enter gesture label (six, four, no_ball): ")
while gesture_label not in valid_gestures:
    gesture_label = input("Invalid gesture. Enter a valid gesture (six, four, no_ball): ")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        if gesture_counts[gesture_label] >= max_samples:
            print(f"Collected {max_samples} samples for {gesture_label}. Choose another gesture.")
            gesture_label = input("Enter new gesture label (six, four, no_ball): ")
            while gesture_label not in valid_gestures or gesture_counts[gesture_label] >= max_samples:
                gesture_label = input("Invalid gesture or max samples reached. Enter a valid gesture (six, four, no_ball): ")

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            data_row = [landmark.x for landmark in landmarks] + \
                       [landmark.y for landmark in landmarks] + \
                       [landmark.z for landmark in landmarks] + [gesture_label]
            
            with open(data_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
                gesture_counts[gesture_label] += 1
            
            print(f"Collected {gesture_counts[gesture_label]}/{max_samples} samples for {gesture_label}")

        cv2.putText(image, f"Gesture: {gesture_label} ({gesture_counts[gesture_label]}/{max_samples})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Umpire Gesture Collection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            gesture_label = input("Enter new gesture label (six, four, no_ball): ")
            while gesture_label not in valid_gestures or gesture_counts[gesture_label] >= max_samples:
                gesture_label = input("Invalid gesture or max samples reached. Enter a valid gesture (six, four, no_ball): ")

cap.release()
cv2.destroyAllWindows()
