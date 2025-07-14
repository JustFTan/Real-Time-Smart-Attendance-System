# Real-Time Smart Attendance System

## Overview

This project is a **Real-Time Smart Attendance System** using face recognition, liveness detection (anti-spoofing with blink detection), and integration with an ESP32 microcontroller for device control (such as unlocking a door). The system is built with Python, using PyQt5 for the GUI, OpenCV for image processing, MediaPipe for facial landmarks, and FaceNet (facenet-pytorch) for face embedding and recognition.

**Key Features:**
- Real-time face detection and recognition via webcam.
- Liveness detection using Eye Aspect Ratio (EAR) to prevent photo spoofing.
- Easy management of face database: add, delete, and refresh user data.
- Attendance logging with export to Excel and CSV.
- Integration with ESP32 for hardware actions (e.g., unlocking doors).
- User-friendly GUI built with PyQt5.

---

## Features

- **Face Recognition:** Uses FaceNet with pre-trained VGGFace2 model for accurate identification.
- **Liveness Detection:** Detects eye blinks using MediaPipe's FaceMesh and EAR calculation to prevent spoofing attacks with photos.
- **Flexible User Management:** Add new users, delete users, refresh the model for new data, and list all registered identities.
- **Attendance Logging:** Records recognized faces with timestamps and corresponding images, exportable to Excel and CSV.
- **ESP32 Integration:** Communicates via serial port to trigger hardware (e.g., open/close doors) based on recognition events.
- **Rich GUI:** Real-time video feedback, simple controls, and detection logs.

---

## Dependencies

- Python 3.7+
- PyQt5
- OpenCV (`opencv-python`)
- NumPy
- Torch (`pytorch`)
- facenet-pytorch
- mediapipe
- scipy
- openpyxl
- pickle (standard library)
- serial (`pyserial`)

Install dependencies using pip:

```bash
pip install pyqt5 opencv-python numpy torch facenet-pytorch mediapipe scipy openpyxl pyserial
```

---

## Folder Structure

- `face_embeddings_model.pkl` : Stores the FaceNet embeddings and names.
- `capture_logs/` : Saved frames of recognized faces for attendance logs.
- `face_recognition_log.xlsx` : Attendance log in Excel format.
- `face_recognition_log.csv` : Attendance log in CSV format (on export).
- `Dataset_Face/` : Directory containing subfolders for each person, each with their face images.

---

## Usage

1. **Connect your ESP32** (ensure the correct serial port is set, default is `COM3`).
2. **Launch the application:**

    ```bash
    python main.py
    ```

3. **GUI Controls:**
   - **Add Data:** Record face images of a new person for 10 seconds.
   - **Refresh Model:** Update the recognition model with new data.
   - **List Daftar Orang:** List all registered users.
   - **Hapus Data Seseorang:** Delete a user and their images.
   - **Export Log ke CSV:** Export the attendance log to CSV file.
   - **Log Table:** View detection history with timestamps, identity, and image path.

4. **Recognition & Logging:**
   - The system recognizes faces in real-time and checks for liveness (blinking).
   - If recognized and real, the system logs the attendance and can trigger actions on the ESP32 (e.g., `UNLOCK:<name>`).

5. **Liveness Detection:**
   - If no blink is detected for more than 3 seconds, the system considers the face as a spoof (photo) and displays a warning.

---

## Customization

- **ESP32 Serial Port:** Change the default serial port (`COM3`) in the code to match your system.
- **Dataset Directory:** The default dataset directory is set at `Dataset_Face`. Update the `base_dataset_dir` variable if needed.
- **Model Thresholds:** You can adjust the similarity threshold or EAR threshold for stricter or more relaxed recognition/liveness detection.

---

## Notes

- Make sure your webcam is connected and accessible.
- For best results, ensure good lighting and clear face visibility during registration and recognition.
- The application runs on CPU or GPU (if available).

---

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/intro)

---

## License

This project is for educational and non-commercial use. Please refer to the licenses of the respective libraries used.
