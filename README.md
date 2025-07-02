# Face Recognition App

A Python-based Face Recognition application with a PyQt5 GUI, real-time face detection using Haar Cascade, feature extraction using [facenet-pytorch](https://github.com/timesler/facenet-pytorch), and anti-spoofing via eye-blink detection using [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html). All recognized face logs can be exported to CSV/Excel.

## Features

- **Live Face Recognition:** Real-time face detection via webcam and identification using the FaceNet model.
- **Anti-Spoofing (Liveness Detection):** Detects fake faces (photos/images) by monitoring eye blinks (Eye Aspect Ratio with MediaPipe).
- **Face Data Management:**
  - Add new faces via GUI (automatic 10 seconds recording process).
  - List all registered people.
  - Delete any person's data from the database.
  - Refresh/update the model database embeddings.
- **Logging:** Every successful face recognition (not spoofed) will be logged (Excel) with timestamp and capture image path.
- **Export Log:** Export recognition log to CSV file.
- **GUI:** PyQt5-based interface with webcam preview, action buttons, and detection log table.

## System Requirements

- Python 3.7+
- Webcam
- CUDA GPU (optional, for faster inference)

## Installation

1. **Clone the repository and install dependencies**
   ```bash
   pip install opencv-python numpy torch facenet-pytorch mediapipe PyQt5 scipy openpyxl
   ```

2. **Folder Structure**
   - Place the main script in your project folder.
   - Face data will automatically be stored in:  
     `Dataset_Face` (default: `C:/Users/justf/Desktop/Sem 6/KP/FaceRecog/Dataset_Face`)
   - Recognition log images are stored in:  
     `capture_logs`
   - Model embeddings are stored in:  
     `face_embeddings_model.pkl`
   - Detection logs are stored in:  
     `face_recognition_log.xlsx`

3. **Run the application**
   ```bash
   python GUI_Main.py
   ```

## How To Use

1. **Add Data**  
   Click "Add Data", enter a name, and the camera will record and save your face for 10 seconds.

2. **Refresh Model**  
   After adding new data, click "Refresh Model" to update the embeddings database.

3. **List Registered People**  
   See the list of people registered in the database.

4. **Delete Someone's Data**  
   Select and delete a person's face data from the database.

5. **Export Log to CSV**  
   Export all face recognition logs to a CSV file.

6. **Live Recognition & Anti-Spoofing**  
   The camera detects and recognizes faces. If no blink is detected for a set period, the label "[FOTO TERDETEKSI]" ("PHOTO DETECTED") appears as a spoofing warning.

## Technical Notes

- The FaceNet model is loaded once at app startup.
- The EAR threshold for blink detection can be adjusted (`EAR_THRESHOLD = 0.2`).
- The similarity threshold for face recognition can be adjusted at `if max_similarity < 0.7:`.
- Logging occurs only if a person is recognized for at least 4 seconds and with at least 10 seconds delay between logs for the same person.
- Each successful detection saves a frame image in the `capture_logs` folder.

## Dependencies

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [MediaPipe](https://google.github.io/mediapipe/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [SciPy](https://scipy.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)

**This application is suitable for thesis projects, research, or as a prototype for automatic attendance systems with basic anti-spoofing security.**
