# Face Recognition Attendance System 📸 Real-Time Automated Attendance

A modern, real-time **Face Recognition Attendance System** built using Python, OpenCV, and `face_recognition`. The system automates attendance tracking using facial recognition via webcam feed and records timestamped logs directly into a CSV file.

---

## 🌟 Key Features

- **Automated Student Face Capture (`add_faces.py`)**: Capture face samples using your webcam with live instructions and visual overlays.
- **Deep Learning Face Encoding (`encode_faces.py`)**: Extract 128-dimensional facial encodings using `face_recognition` (dlib ResNet model) for robust match accuracy.
- **Real-Time Live Attendance (`attendance.py`)**: Recognizes faces in real time with high accuracy, displays live bounding boxes with student names, and logs attendance.
- **Duplicate Prevention**: Prevents logging the same student multiple times on the same date.
- **Automated CSV Logging**: Automatically logs student name, date, and time into `attendance.csv`.

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System/
├── add_faces.py        # Module 1: Capture student photos via webcam
├── encode_faces.py     # Module 2: Generate & pickle 128-d face encodings
├── attendance.py       # Module 3: Live face recognition & attendance logging
├── requirements.txt    # Required Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have **Python 3.8+** installed. Clone this repository and install the dependencies:

```bash
git clone https://github.com/abdulsamad1366/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
pip install -r requirements.txt
```

> **Note for `face_recognition` installation**:  
> `face_recognition` requires `dlib` and CMake.
> - **macOS**: `brew install cmake`
> - **Windows**: Install *C++ CMake tools for Windows* via Visual Studio Build Tools.
> - **Linux**: `sudo apt-get install build-essential cmake`

---

## 📋 How to Use

Follow these 3 simple steps to get the system up and running:

### Step 1: Capture Student Faces
Run the script to collect sample photos for a student:
```bash
python add_faces.py
```
- Enter the **Student Name / ID** when prompted.
- Position your face in front of the webcam.
- Press **'S'** to capture a photo (recommended: 10-15 photos per person with varied expressions/lighting).
- Press **'Q'** when done.

### Step 2: Generate Face Encodings
Generate the facial encoding vector embeddings stored in `encodings.pickle`:
```bash
python encode_faces.py
```

### Step 3: Run Live Attendance System
Start real-time face recognition and attendance logging:
```bash
python attendance.py
```
- Recognized students will have a **Green Box** around their face and their attendance will be recorded in `attendance.csv`.
- Unrecognized individuals will show a **Red Box** ("Unknown").
- Press **'Q'** to exit the application.

---

## 📊 Attendance Log Example (`attendance.csv`)

| Name | Date | Time |
| :--- | :--- | :--- |
| John Doe | 2026-08-16 | 09:15:30 |
| Jane Smith | 2026-08-16 | 09:16:02 |

---

## 🛠️ Built With

- **Python** - Core language
- **OpenCV (`cv2`)** - Real-time video processing & frame manipulation
- **`face_recognition`** - Deep learning face detection & 128-d encoding
- **NumPy** - Numerical operations & vector distance comparisons

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
