import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load trained model
model = tf.keras.models.load_model('gesture_model.h5')

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['four', 'no_ball', 'six'])  # Ensure the order matches training

# Load scaler (retrain scaler with dataset for consistency)
data_file = 'umpire_gestures.csv'
df = pd.read_csv(data_file)
scaler = StandardScaler()
X = df.drop(columns=['gesture_label'])
scaler.fit(X)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image
        results = pose.process(image)

        # Convert back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            data_row = np.array([[landmark.x for landmark in landmarks] +
                                 [landmark.y for landmark in landmarks] +
                                 [landmark.z for landmark in landmarks]])
            
            # Normalize data
            data_row = scaler.transform(data_row)
            
            # Predict gesture
            prediction = model.predict(data_row)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            
            # Display prediction on screen
            cv2.putText(image, f'Gesture: {predicted_label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Gesture Prediction', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()