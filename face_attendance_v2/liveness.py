import numpy as np
import time
import face_recognition

def euclidean_dist(pt1, pt2):
    """Computes Euclidean distance between two (x, y) tuple points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye_landmarks):
    """
    Computes Eye Aspect Ratio (EAR) for a single eye landmark set (6 points).
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if len(eye_landmarks) < 6:
        return 0.3  # Default open eye EAR fallback

    # Compute vertical distances
    v1 = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_dist(eye_landmarks[2], eye_landmarks[4])

    # Compute horizontal distance
    h = euclidean_dist(eye_landmarks[0], eye_landmarks[3])

    if h == 0:
        return 0.3

    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_frame_ear(rgb_frame, face_location):
    """
    Extracts facial landmarks for a given face_location and returns average EAR.
    """
    landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_location])
    if not landmarks_list:
        return None

    landmarks = landmarks_list[0]
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")

    if not left_eye or not right_eye:
        return None

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear

class BlinkDetector:
    """
    Tracks Eye Aspect Ratio (EAR) across consecutive frames to detect valid eye blinks.
    Prevents photo / static screen spoofing attempts.
    """
    def __init__(self, ear_threshold=0.22, min_consec_frames=1, reset_timeout=5.0):
        self.ear_threshold = ear_threshold
        self.min_consec_frames = min_consec_frames
        self.reset_timeout = reset_timeout
        self.tracking_data = {}  # key: person_identifier -> dict of state

    def check_liveness(self, person_id, ear_val):
        """
        Updates blink state for a person.
        Returns (is_live: bool, status_message: str).
        """
        now = time.time()

        if ear_val is None:
            return False, "Searching facial landmarks..."

        if person_id not in self.tracking_data:
            self.tracking_data[person_id] = {
                "consec_closed": 0,
                "blink_detected": False,
                "blink_time": None,
                "last_seen": now
            }

        state = self.tracking_data[person_id]
        state["last_seen"] = now

        # Check if blink was already validated within recent reset_timeout window
        if state["blink_detected"] and state["blink_time"]:
            if (now - state["blink_time"]) <= self.reset_timeout:
                return True, "Liveness Verified (Blink detected)"
            else:
                # Reset blink validation after timeout
                state["blink_detected"] = False
                state["blink_time"] = None
                state["consec_closed"] = 0

        # Process frame EAR
        if ear_val < self.ear_threshold:
            state["consec_closed"] += 1
        else:
            if state["consec_closed"] >= self.min_consec_frames:
                # Eye was closed for required frames and now opened -> Blink detected!
                state["blink_detected"] = True
                state["blink_time"] = now
                state["consec_closed"] = 0
                return True, "Liveness Verified (Blink detected)"
            state["consec_closed"] = 0

        return False, "Please blink to verify liveness"

    def reset_person(self, person_id):
        if person_id in self.tracking_data:
            del self.tracking_data[person_id]
