# recognition_opencv.py
import cv2
import numpy as np
import os

# Load OpenCV's Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "faces"
labels, faces = [], []
label_dict = {}

# Load and label face images
print("Training face recognition model...")

current_id = 0
for user_id in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_id)
    
    if os.path.isdir(user_folder):
        label_dict[current_id] = user_id
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if gray_img is not None:
                faces.append(gray_img)
                labels.append(current_id)
    
        current_id += 1

# Convert to NumPy arrays and train the recognizer
if faces:
    recognizer.train(faces, np.array(labels))
    print("Training complete.")
else:
    print("No face data found. Please enroll users first.")
    exit()

# Start real-time face recognition
cap = cv2.VideoCapture(0)
print("Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_crop = gray[y:y + h, x:x + w]

        # Predict the face ID using LBPH recognizer
        label, confidence = recognizer.predict(face_crop)

        if confidence < 70:  # Lower value means better confidence
            user_name = label_dict.get(label, "Unknown")
            print(f"Recognized: {user_name} (Confidence: {confidence:.2f})")
        else:
            user_name = "Unknown"

        # Draw a rectangle and put the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, user_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
