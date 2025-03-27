A robust face matching and recognition system built using OpenCV. The project captures facial data, trains an LBPH model for recognition, and processes real-time video feeds to identify faces. It features intuitive controls, real-time feedback, and an efficient, lightweight design.

Introduction of Project - 
```
**Project Overview: Face Matching and Recognition**

1. **Facial Data Enrollment**  
   - Utilizes a webcam to capture face images in real-time.
   - Employs OpenCV’s Haar Cascade for robust face detection.
   - Saves captured grayscale face images locally, organized by user ID.

2. **Training and Recognition Process**  
   - Loads stored face images to label and train a face recognition model.
   - Uses the LBPH (Local Binary Patterns Histograms) algorithm for efficient and reliable face recognition.
   - Constructs a mapping between numerical labels and user IDs for accurate identification.

3. **Real-Time Processing**  
   - Provides live video feed with highlighted face detection using bounding boxes.
   - Displays the recognized user’s ID along with a confidence score directly on the video stream.
   - Offers a seamless interactive experience with immediate visual feedback.

4. **User-Friendly Interaction**  
   - Simple command inputs: press 's' to capture and save a face image, and 'q' to quit the process.
   - Ensures an intuitive operation for both enrollment and recognition phases.
   - Designed to work optimally with good lighting conditions for best results.

5. **Efficient and Lightweight Implementation**  
   - Combines classical computer vision techniques (Haar Cascades and LBPH) for a resource-efficient solution.
   - Suitable for real-time applications with minimal hardware requirements.
   - Provides a clear structure and modular approach, allowing easy future enhancements.
```


for saving the facial data , run -:

```bash
python run enrollment.py
```
after the camera window is open , while green box on face press S key to save the face data locally and Q when for quiting.

for recognition run :

```bash
python run recognition.py
```
good lighting recomended. 

