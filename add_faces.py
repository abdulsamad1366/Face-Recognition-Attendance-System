import cv2
import os
import sys

def main():
    print("==================================================")
    print("      FACE RECOGNITION ATTENDANCE SYSTEM          ")
    print("               Add Student Faces                  ")
    print("==================================================")

    # Step 1: Ask the user to enter student name or ID
    student_name = input("Enter Student Name/ID: ").strip()
    
    if not student_name:
        print("[ERROR] Student name cannot be empty. Exiting.")
        sys.exit(1)

    # Step 2: Define dataset folder path and create it if it doesn't exist
    dataset_dir = "dataset"
    student_dir = os.path.join(dataset_dir, student_name)
    os.makedirs(student_dir, exist_ok=True)

    # Count existing images in the student directory to avoid overwriting existing photos
    existing_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(existing_files) + 1

    # Step 3: Open the default webcam (index 0)
    print("\n[INFO] Initializing webcam... Please look at the camera.")
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check camera access permissions.")
        sys.exit(1)

    print("[INFO] Controls:")
    print("       - Press 's' or 'S' to capture and save a face photo.")
    print("       - Press 'q' or 'Q' to quit.")
    print("[INFO] Capture at least 10-15 images per person with different expressions/angles for best accuracy.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame from webcam. Exiting loop.")
            break

        # Create a copy of the frame for display with visual guide instructions
        display_frame = frame.copy()
        
        # Display instructions and captured count directly on the video window
        info_text = f"Student: {student_name} | Photos Saved: {count - 1}"
        controls_text = "Press 'S' to Save Photo | 'Q' to Quit"

        # Draw semi-transparent banner background for readability
        cv2.putText(display_frame, info_text, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, controls_text, (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show live webcam preview
        cv2.imshow("Add Student Faces - Press 'S' to Save", display_frame)

        # Listen for key presses (1 ms delay)
        key = cv2.waitKey(1) & 0xFF

        # If 's' or 'S' is pressed, save the original clean frame
        if key == ord('s') or key == ord('S'):
            img_path = os.path.join(student_dir, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Saved image {count} -> {img_path}")
            count += 1

        # If 'q' or 'Q' is pressed, exit
        elif key == ord('q') or key == ord('Q'):
            print("\n[INFO] Exiting image capture.")
            break

    # Step 4: Release camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Total images captured for '{student_name}': {count - 1}")

if __name__ == "__main__":
    main()
