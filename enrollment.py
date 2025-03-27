# enrollment_opencv.py
import cv2
import os

# Create a directory to store face images
dataset_path = "faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load OpenCV's built-in face detector (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize camera
cap = cv2.VideoCapture(0)

user_id = input("Enter your ID (numeric): ").strip()
user_folder = os.path.join(dataset_path, user_id)

# Create a folder for the user if it doesn't exist
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

count = 0
print("Press 's' to capture a face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Enrollment", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            file_name = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(file_name, face_crop)
            print(f"Saved face image: {file_name}")
            count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
