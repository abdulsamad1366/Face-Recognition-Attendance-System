import cv2
import os
import sys
import pickle
import csv
from datetime import datetime
import numpy as np
import face_recognition

def load_existing_attendance(csv_path):
    """
    Reads existing attendance log to prevent marking duplicate entries
    for the same person on the same date.
    Returns a set of tuples: {(name, date_string)}
    """
    marked_records = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header row
            for row in reader:
                if len(row) >= 2:
                    name, date_str = row[0].strip(), row[1].strip()
                    marked_records.add((name, date_str))
    return marked_records

def mark_attendance(name, csv_path, marked_records):
    """
    Appends a new row to attendance.csv if the student is not already logged today.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Check if student is already marked for today
    if (name, date_str) not in marked_records:
        # Create CSV with header if it doesn't exist yet
        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])  # Write CSV Header
            
            writer.writerow([name, date_str, time_str])
        
        # Track in memory to avoid duplicate disk writes during the session
        marked_records.add((name, date_str))
        print(f"[ATTENDANCE MARKED] Name: {name} | Date: {date_str} | Time: {time_str}")

def main():
    print("==================================================")
    print("      FACE RECOGNITION ATTENDANCE SYSTEM          ")
    print("                Live Attendance                   ")
    print("==================================================")

    pickle_file = "encodings.pickle"
    csv_file = "attendance.csv"
    distance_threshold = 0.5  # Lower distance = stricter match (0.5 is ideal for face_recognition)

    # Step 1: Check if encodings.pickle exists before running attendance
    if not os.path.exists(pickle_file):
        print(f"[ERROR] Encodings file '{pickle_file}' not found!")
        print("[HINT] Please run 'encode_faces.py' first to generate face encodings.")
        sys.exit(1)

    # Step 2: Load known face encodings and student names
    print(f"[INFO] Loading face encodings from '{pickle_file}'...")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    known_encodings = data.get("encodings", [])
    known_names = data.get("names", [])

    if not known_encodings:
        print("[ERROR] No encodings found in pickle file! Re-run encode_faces.py.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(known_encodings)} face encoding(s) for {len(set(known_names))} unique student(s).")

    # Ensure attendance.csv exists with header
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Load existing attendance records into memory
    marked_records = load_existing_attendance(csv_file)

    # Step 3: Open webcam
    print("\n[INFO] Starting webcam for live face recognition...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check camera access permissions.")
        sys.exit(1)

    print("[INFO] Press 'q' or 'Q' to quit.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame from webcam. Exiting.")
            break

        # Resize frame to 0.25x for faster real-time processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate distance between current face and all known encodings
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            name = "Unknown"
            box_color = (0, 0, 255)  # Red box for unknown faces

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                # Check if best match distance is below tolerance threshold (0.5)
                if best_distance < distance_threshold:
                    name = known_names[best_match_index]
                    box_color = (0, 255, 0)  # Green box for recognized student

                    # Mark attendance in CSV file (if not already logged today)
                    mark_attendance(name, csv_file, marked_records)

            # Scale back face location coordinates up 4x to match original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw bounding box around face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Draw filled rectangle background for name text label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display live webcam frame with overlays
        cv2.imshow("Face Recognition Attendance System", frame)

        # Quit when 'q' or 'Q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Stopping live attendance camera.")
            break

    # Step 4: Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session closed. Attendance logged in 'attendance.csv'.")

if __name__ == "__main__":
    main()
