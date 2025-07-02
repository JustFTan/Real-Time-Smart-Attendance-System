Great, let's create a comprehensive README for your GitHub project titled "Real-time Smart Face Attendance System".

Real-time Smart Face Attendance System
A smart, real-time facial recognition attendance system designed for efficiency and security. This application not only detects and identifies individuals but also incorporates an anti-spoofing feature to prevent fraudulent attendance using photos or videos.

Key Features
Real-time Facial Recognition: Instantly identifies users from a live camera feed.

Face Data Management: Easily add and remove individual face data to and from the model's database.

Intelligent Anti-Spoofing: Utilizes eye blink detection (EAR - Eye Aspect Ratio) via MediaPipe FaceMesh to differentiate real faces from photos/videos.

Automatic Logging System: Automatically logs attendance with a timestamp, identity, and the path to the captured image in an Excel file (.xlsx).

Log Export: Export attendance data from the Excel log to CSV format.

Intuitive Graphical User Interface (GUI): Built with PyQt5 for an easy-to-use experience.

Pre-trained FaceNet Model: Leverages a pre-trained FaceNet model (InceptionResnetV1 with vggface2 weights) for high recognition accuracy.

How Anti-Spoofing Works
The system implements anti-spoofing detection by monitoring a user's eye blinks. If a face is detected but no eye blink occurs for a specified duration (currently 3 seconds), the system will flag it as a potential "photo detected" or spoofing attempt, and attendance will not be logged. This ensures that only genuinely present individuals can be recorded.

Technologies Used
Python: The primary programming language.

PyQt5: For building the graphical user interface (GUI).

OpenCV: Used for image processing and initial face detection (Haar Cascades).

FaceNet-PyTorch: An implementation of FaceNet for generating face embeddings.

Mediapipe: Utilized for FaceMesh to detect facial landmarks, crucial for EAR (Eye Aspect Ratio) calculation and blink detection.

NumPy: For numerical operations, especially embedding manipulation.

SciPy: For cosine similarity calculations between embeddings.

openpyxl: To read from and write to Excel files.

pickle: To save and load face embeddings from disk.

Installation
To get this application up and running, follow these steps:

Clone the Repository:

Bash

git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name
(Replace your_username and your_repo_name with your actual repository details.)

Create and Activate a Virtual Environment (Recommended):

Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install Dependencies:

Bash

pip install -r requirements.txt
Make sure you have a requirements.txt file in your root directory containing:

PyQt5
opencv-python
facenet-pytorch
scipy
numpy
mediapipe
openpyxl
torch # Include if not automatically installed by facenet-pytorch
Usage
Run the Application:

Bash

python GUI_Main.py

Adding New Face Data:

Click the "Add Data" button.

Enter the name of the person you want to add.

The application will start recording your face for 10 seconds. Ensure your face is clearly visible to the camera.

Once done, click "Refresh Model" to incorporate the new face data into the recognition system.

Taking Attendance:

Simply position your face in front of the camera.

The system will attempt to recognize you and perform anti-spoofing detection (requiring eye blinks).

If successfully recognized and not detected as spoofing for 4 seconds, your attendance will be logged.

Other Features:

"Refresh Model": Reloads all face embeddings from the Dataset_Face folder. You must do this after adding or deleting data.

"List Daftar Orang" (List People): Displays a list of names registered in the system.

"Hapus Data Seseorang" (Delete Person Data): Allows you to delete a person's face data from the dataset. Remember to click "Refresh Model" after deletion.

"Export Log ke CSV" (Export Log to CSV): Exports the attendance log file from Excel format to CSV.

Project Structure
.
├── main.py                    # Main PyQt5 application code
├── face_embeddings_model.pkl  # File storing face embeddings (auto-generated)
├── face_recognition_log.xlsx  # Attendance log file (auto-generated)
├── capture_logs/              # Folder to store logged face images
│   └── (timestamp)_(identity).jpg
└── Dataset_Face/              # Folder to store dataset face images
    ├── PersonName1/
    │   ├── PersonName1_000.jpg
    │   └── PersonName1_001.jpg
    └── PersonName2/
        ├── PersonName2_000.jpg
        └── ...
Contributing
Contributions are highly welcome! If you have suggestions, improvements, or want to add new features, please feel free to open a pull request.
