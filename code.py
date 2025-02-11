import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance
import winsound  # For Windows users, use 'playsound' for other OS
import os  # To create and manage the folder for saving images

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\sahil\projects\attention\shape_predictor_68_face_landmarks.dat")

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Landmark indices for left and right eye
LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

# Thresholds and timers
EAR_THRESHOLD = 0.25
CLOSE_DURATION = 1  # in seconds
start_time = None

# Create directory to store images if it doesn't exist
image_folder = r"C:\Users\sahil\projects\attention\drowsiness_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX])

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eye landmarks
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

        # Detect sleep
        if avg_EAR < EAR_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time >= CLOSE_DURATION:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(1000, 1000)  # Play siren for Windows
                
                # Save the image with a unique name
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(image_folder, f"drowsy_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)  # Save the frame as an image

        else:
            start_time = None  # Reset timer if eyes are open

    # Show frame
    cv2.imshow("Sleep Detector", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
