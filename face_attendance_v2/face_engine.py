import os
import pickle
import numpy as np
import face_recognition

BASE_DIR = os.path.dirname(__file__)
DEFAULT_PICKLE = os.path.join(BASE_DIR, "encodings.pickle")

def get_pickle_path():
    if os.path.exists(DEFAULT_PICKLE):
        return DEFAULT_PICKLE
    parent_pickle = os.path.join(os.path.dirname(BASE_DIR), "encodings.pickle")
    if os.path.exists(parent_pickle):
        return parent_pickle
    return DEFAULT_PICKLE

def load_known_encodings(pickle_path=None):
    """Loads face encodings and student names from pickle file."""
    if pickle_path is None:
        pickle_path = get_pickle_path()

    if not os.path.exists(pickle_path):
        return [], []

    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return data.get("encodings", []), data.get("names", [])
    except Exception as e:
        print(f"[FACE ENGINE ERROR] Failed to load encodings from {pickle_path}: {e}")
        return [], []

def save_known_encodings(encodings, names, pickle_path=None):
    """Saves face encodings and student names into pickle file."""
    if pickle_path is None:
        pickle_path = DEFAULT_PICKLE

    data = {
        "encodings": encodings,
        "names": names
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[FACE ENGINE INFO] Saved {len(encodings)} encodings to {pickle_path}")

def match_face(encoding, known_encodings, known_names, threshold=0.5):
    """
    Compares a single face encoding against known encodings.
    Returns (matched_name, distance) if distance < threshold, else (None, distance).
    """
    if len(known_encodings) == 0:
        return None, None

    distances = face_recognition.face_distance(known_encodings, encoding)
    if len(distances) == 0:
        return None, None

    best_idx = np.argmin(distances)
    best_distance = float(distances[best_idx])

    if best_distance < threshold:
        return known_names[best_idx], best_distance

    return None, best_distance

def encode_student_images(student_name, image_paths):
    """
    Encodes a list of image files for a student and appends them to encodings.pickle.
    Returns the count of new encodings created.
    """
    known_encodings, known_names = load_known_encodings()
    new_count = 0

    for path in image_paths:
        try:
            image = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(image)
            if len(boxes) > 0:
                encs = face_recognition.face_encodings(image, known_face_locations=boxes)
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(student_name)
                    new_count += 1
        except Exception as e:
            print(f"[FACE ENGINE WARNING] Error encoding {path}: {e}")

    if new_count > 0:
        save_known_encodings(known_encodings, known_names)

    return new_count

def rebuild_all_encodings(dataset_dir=None):
    """Scans all subfolders in dataset/ and rebuilds encodings.pickle."""
    if dataset_dir is None:
        dataset_dir = os.path.join(BASE_DIR, "dataset")
        if not os.path.exists(dataset_dir):
            dataset_dir = os.path.join(os.path.dirname(BASE_DIR), "dataset")

    if not os.path.exists(dataset_dir):
        return 0

    known_encodings = []
    known_names = []

    student_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    for folder in student_folders:
        folder_path = os.path.join(dataset_dir, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            try:
                img = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(img)
                if len(boxes) > 0:
                    encs = face_recognition.face_encodings(img, known_face_locations=boxes)
                    if len(encs) > 0:
                        known_encodings.append(encs[0])
                        known_names.append(folder)
            except Exception as e:
                print(f"[REBUILD ERROR] Failed {img_path}: {e}")

    save_known_encodings(known_encodings, known_names)
    return len(known_encodings)
