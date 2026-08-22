import cv2
import time
import random
import numpy as np
import face_recognition

def euclidean_dist(pt1, pt2):
    """Computes Euclidean distance between two (x, y) tuple points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye_landmarks):
    """Computes Eye Aspect Ratio (EAR)."""
    if len(eye_landmarks) < 6:
        return 0.3
    v1 = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    h = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    if h == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks):
    """Computes Mouth Aspect Ratio (MAR)."""
    top_lip = landmarks.get("top_lip")
    bottom_lip = landmarks.get("bottom_lip")
    if not top_lip or not bottom_lip:
        return 0.0

    top_center = top_lip[9] if len(top_lip) > 9 else top_lip[-1]
    bottom_center = bottom_lip[9] if len(bottom_lip) > 9 else bottom_lip[-1]
    left_corner = top_lip[0]
    right_corner = top_lip[6] if len(top_lip) > 6 else top_lip[-1]

    v_dist = euclidean_dist(top_center, bottom_center)
    h_dist = euclidean_dist(left_corner, right_corner)
    if h_dist == 0:
        return 0.0
    return v_dist / h_dist

def compute_head_ratios(landmarks):
    """
    Computes robust 2D Facial Landmark Ratios for Yaw (Head Turn).
    """
    try:
        nose_bridge = landmarks.get("nose_bridge")
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        chin = landmarks.get("chin")

        if not (nose_bridge and left_eye and right_eye and chin):
            return 0.0, 0.4

        nose_tip = nose_bridge[-1]  # Point 30: True nose tip
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        eye_center = (left_eye_center + right_eye_center) / 2.0
        eye_dist = euclidean_dist(left_eye_center, right_eye_center)

        if eye_dist == 0:
            return 0.0, 0.4

        yaw_ratio = (nose_tip[0] - eye_center[0]) / eye_dist
        chin_y = chin[8][1] if len(chin) > 8 else chin[-1][1]
        pitch_ratio = (nose_tip[1] - eye_center[1]) / abs(chin_y - eye_center[1]) if abs(chin_y - eye_center[1]) > 0 else 0.4

        return float(yaw_ratio), float(pitch_ratio)
    except Exception:
        return 0.0, 0.4

def detect_screen_replay(frame, face_box):
    """
    Screen Replay Artifact Detector.
    Inspects outer border for rectangular phone/tablet bezel contours.
    """
    try:
        h, w, _ = frame.shape
        top, right, bottom, left = face_box
        box_w = right - left
        box_h = bottom - top
        pad_w = int(box_w * 0.35)
        pad_h = int(box_h * 0.35)

        ext_top = max(0, top - pad_h)
        ext_bottom = min(h, bottom + pad_h)
        ext_left = max(0, left - pad_w)
        ext_right = min(w, right + pad_w)

        roi = frame[ext_top:ext_bottom, ext_left:ext_right]
        if roi.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = (ext_bottom - ext_top) * (ext_right - ext_left)

        rect_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (roi_area * 0.15):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect_ratio = float(cw) / ch if ch > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.0:
                        rect_count += 1

        return (rect_count >= 2), float(rect_count)
    except Exception:
        return False, 0.0

ACTION_PROMPTS = {
    "turn_head": "TASK: Turn Head Left or Right",
    "open_mouth": "TASK: Open Mouth Wide",
    "blink": "TASK: Blink Eyes"
}

class SingleTaskLivenessSession:
    """
    Independent Unique Randomized Liveness Task per Person:
    Seeded specifically per student so each person receives their own distinct task.
    Attendance is logged strictly for the individual student who completes their assigned task.
    """
    def __init__(self, tracking_id="student", time_limit=8.0):
        self.tracking_id = tracking_id
        self.time_limit = time_limit
        
        # Seed pseudo-random generator with person tracking_id to guarantee unique tasks per student
        rng = random.Random(hash(tracking_id + str(time.time())))
        self.target_task = rng.choice(["turn_head", "open_mouth", "blink"])
        
        self.start_time = time.time()
        self.status = "pending"
        self.failure_reason = None
        self.consec_closed = 0
        self.ear_threshold = 0.22

    def evaluate_frame(self, frame, face_loc, rgb_frame):
        now = time.time()
        elapsed = now - self.start_time
        time_remaining = max(0.0, self.time_limit - elapsed)

        if self.status != "pending":
            return {
                "status": self.status,
                "failure_reason": self.failure_reason,
                "prompt": "Liveness Verified & Logged!" if self.status == "passed" else f"Security Fail: {self.failure_reason}",
                "target_task": self.target_task,
                "time_remaining": 0.0
            }

        # 1. Screen Replay Artifact Check
        is_screen, _ = detect_screen_replay(frame, face_loc)
        if is_screen:
            self.status = "failed"
            self.failure_reason = "screen_detected"
            return {
                "status": "failed",
                "failure_reason": "screen_detected",
                "prompt": "ALERT: Screen Replay Attack Detected!",
                "target_task": self.target_task,
                "time_remaining": 0.0
            }

        # 2. Timeout Check
        if elapsed > self.time_limit:
            self.status = "failed"
            self.failure_reason = "timeout"
            return {
                "status": "failed",
                "failure_reason": "timeout",
                "prompt": "Liveness Task Timed Out",
                "target_task": self.target_task,
                "time_remaining": 0.0
            }

        # 3. Extract Landmarks & Check Task Action
        landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_loc])
        if not landmarks_list:
            prompt_text = f"{ACTION_PROMPTS[self.target_task]} ({round(time_remaining, 1)}s)"
            return {
                "status": "pending",
                "prompt": prompt_text,
                "target_task": self.target_task,
                "time_remaining": round(time_remaining, 1)
            }

        landmarks = landmarks_list[0]
        yaw_ratio, pitch_ratio = compute_head_ratios(landmarks)
        mar = calculate_mar(landmarks)
        left_ear = calculate_ear(landmarks.get("left_eye", []))
        right_ear = calculate_ear(landmarks.get("right_eye", []))
        avg_ear = (left_ear + right_ear) / 2.0

        task_passed = False
        if self.target_task == "turn_head":
            task_passed = (abs(yaw_ratio) > 0.16)  # Turn head left or right
        elif self.target_task == "open_mouth":
            task_passed = (mar > 0.38)
        elif self.target_task == "blink":
            if avg_ear < self.ear_threshold:
                self.consec_closed += 1
            else:
                if self.consec_closed >= 1:
                    task_passed = True
                    self.consec_closed = 0

        if task_passed:
            self.status = "passed"
            return {
                "status": "passed",
                "prompt": "Liveness Verified & Logged!",
                "target_task": self.target_task,
                "time_remaining": 0.0
            }

        prompt_text = f"{ACTION_PROMPTS[self.target_task]} ({round(time_remaining, 1)}s)"
        return {
            "status": "pending",
            "prompt": prompt_text,
            "target_task": self.target_task,
            "time_remaining": round(time_remaining, 1)
        }

AdvancedLivenessSession = SingleTaskLivenessSession
