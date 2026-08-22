import cv2
import os
import time
from datetime import datetime
import numpy as np
import face_recognition
import face_engine
from liveness import BlinkDetector, get_frame_ear
import database

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

class CameraStreamManager:
    def __init__(self):
        self.camera = None
        self.blink_detector = BlinkDetector(ear_threshold=0.22, min_consec_frames=1, reset_timeout=5.0)
        self.last_attempt_log_time = {}  # Throttle attempt logging per face (e.g. max once per 2 seconds)

    def get_camera(self):
        if self.camera is None or not self.camera.isOpened():
            for idx in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.camera = cap
                            print(f"[CAMERA INFO] Successfully initialized camera index {idx} via AVFOUNDATION")
                            break
                    cap.release()
                except Exception as e:
                    print(f"[CAMERA WARNING] Failed index {idx}: {e}")

            if self.camera is None or not self.camera.isOpened():
                # Fallback to standard VideoCapture
                self.camera = cv2.VideoCapture(0)
        return self.camera

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def save_snapshot(self, frame, result_label):
        """Saves a frame snapshot for the audit trail log."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"attempt_{result_label}_{timestamp_str}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        return f"snapshots/{filename}"

    def generate_frames(self):
        """MJPEG Live stream generator yielding encoded JPEG frames."""
        cap = self.get_camera()
        known_encodings, known_names = face_engine.load_known_encodings()

        while True:
            # Fetch active class session from database (Layer 1 Security)
            active_session = database.get_active_session()
            session_id = active_session["id"] if active_session else None

            # IF NO ACTIVE SESSION: Release physical camera and yield CAMERA OFF frame
            if not active_session:
                self.release_camera()

                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                # Draw dark background card
                cv2.rectangle(placeholder, (40, 40), (600, 440), (30, 41, 59), cv2.FILLED)
                cv2.rectangle(placeholder, (40, 40), (600, 440), (71, 85, 105), 2)

                cv2.putText(placeholder, "CAMERA OFF", (230, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(placeholder, "No Active Attendance Session", (155, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(placeholder, "Start a class session to turn camera ON", (150, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            # ACTIVE SESSION IS RUNNING: Get camera and read frames
            cap = self.get_camera()
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.03)
                continue

            try:
                # Resize frame to 0.25x for speed
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                now_time = time.time()

                for face_loc, face_enc in zip(face_locations, face_encodings):
                    # 1. Match Face Against Known Encodings
                    matched_name, distance = face_engine.match_face(face_enc, known_encodings, known_names, threshold=0.5)

                    # 2. Check Liveness (Layer 2 Security - Blink EAR Check)
                    ear_val = get_frame_ear(rgb_small_frame, face_loc)
                    tracking_id = matched_name if matched_name else "unknown_face"
                    is_live, liveness_msg = self.blink_detector.check_liveness(tracking_id, ear_val)

                    # Determine Display Attributes & Security Logic
                    if matched_name and is_live:
                        box_color = (0, 255, 0)  # Green
                        status_text = f"{matched_name} (Verified)"
                        attempt_result = "matched"
                    elif matched_name and not is_live:
                        box_color = (0, 255, 255)  # Yellow
                        status_text = f"{matched_name} (Blink to Verify)"
                        attempt_result = "liveness_fail"
                    else:
                        box_color = (0, 0, 255)  # Red
                        status_text = "Unknown Face"
                        attempt_result = "unknown"

                    # Lookup Student Record if Matched
                    student_id = None
                    if matched_name:
                        student_rec = database.get_student_by_identifier(matched_name)
                        if student_rec:
                            student_id = student_rec["id"]

                    # Log Attendance if Session Active + Matched + Live (DB UNIQUE constraint prevents duplicates)
                    if matched_name and is_live and active_session and student_id:
                        database.log_attendance(student_id, session_id)

                    # 3. Log Attempt to Audit Trail (Layer 3 Security - throttled once per 3 sec per tracking_id)
                    last_log = self.last_attempt_log_time.get(tracking_id, 0)
                    if (now_time - last_log) >= 3.0:
                        snapshot_path = self.save_snapshot(frame, attempt_result)
                        database.log_attempt(
                            session_id=session_id,
                            result=attempt_result,
                            student_id=student_id,
                            distance=distance,
                            snapshot_path=snapshot_path
                        )
                        self.last_attempt_log_time[tracking_id] = now_time

                    # Scale face bounding box back up to 100% frame size
                    top, right, bottom, left = [coord * 4 for coord in face_loc]

                    # Draw Bounding Box & Label Overlay
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                    cv2.putText(frame, status_text, (left + 6, bottom - 8),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0) if box_color == (0, 255, 255) else (255, 255, 255), 1)

                # Draw Active Session Banner on Video Feed
                banner_text = f"ACTIVE SESSION: {active_session['class_name']}"
                cv2.putText(frame, banner_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"[STREAM RECOGNITION WARNING] Frame error: {e}")

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

stream_manager = CameraStreamManager()
