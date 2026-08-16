import os
import sys
import pickle
import face_recognition

def main():
    print("==================================================")
    print("      FACE RECOGNITION ATTENDANCE SYSTEM          ")
    print("               Encode Dataset Faces               ")
    print("==================================================")

    dataset_dir = "dataset"
    output_pickle = "encodings.pickle"

    # Step 1: Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"[WARNING] '{dataset_dir}' directory does not exist!")
        print("[HINT] Please run 'add_faces.py' first to capture student photos.")
        sys.exit(1)

    # Step 2: Get all subdirectories (each subfolder represents one student)
    student_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    if not student_folders:
        print(f"[WARNING] '{dataset_dir}' is empty! No student folders found.")
        print("[HINT] Please run 'add_faces.py' to add at least one student.")
        sys.exit(1)

    known_encodings = []
    known_names = []
    total_images_processed = 0

    # Step 3: Loop through every student directory
    for student_name in student_folders:
        student_path = os.path.join(dataset_dir, student_name)
        image_files = [f for f in os.listdir(student_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"[WARNING] Skipping student '{student_name}': No images found in folder.")
            continue

        student_encoding_count = 0

        # Loop through each image for the student
        for filename in image_files:
            image_path = os.path.join(student_path, filename)
            
            try:
                # Load image file (RGB format format expected by face_recognition)
                image = face_recognition.load_image_file(image_path)

                # Locate faces in the image
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) == 0:
                    print(f"  [SKIP] No face detected in: {image_path}")
                    continue

                # Generate 128-dimensional face encoding vector
                encodings = face_recognition.face_encodings(image, face_locations)

                # Append face encoding and student name
                known_encodings.append(encodings[0])
                known_names.append(student_name)
                student_encoding_count += 1
                total_images_processed += 1

            except Exception as e:
                print(f"  [ERROR] Failed to process {image_path}: {e}")

        print(f"[INFO] Processing student: {student_name} ({student_encoding_count}/{len(image_files)} images encoded)")

    if total_images_processed == 0:
        print("\n[ERROR] No face encodings could be generated from the dataset.")
        print("[HINT] Make sure image photos contain clear, visible faces.")
        sys.exit(1)

    # Step 4: Save encodings and names into encodings.pickle file
    print(f"\n[INFO] Saving encodings to '{output_pickle}'...")
    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(output_pickle, "wb") as f:
        pickle.dump(data, f)

    print("==================================================")
    print(f"[SUCCESS] Encoding complete!")
    print(f"          Total face encodings saved: {len(known_encodings)}")
    print(f"          Encodings stored in: {output_pickle}")
    print("==================================================")

if __name__ == "__main__":
    main()
